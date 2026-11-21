"""CoVoST2 as OWSM utterances.

Ported from ``egs2/owsm_v1/s2t1/local/prepare_covost2.py``, but from the
released manifests rather than that script's input: it reads an espnet2
``{split}.{src}-{tgt}/`` directory, which is itself prepared from these files,
and its utterance ids carry a split and a language pair the published dump does
not have. The dump's ids are ``CoVoST2_{client_id}-{clip}``, so they are minted
from the manifests directly.

Each clip yields two utterances, each its own group: ``<asr>`` over the
CommonVoice transcript and ``<st_{tgt}>`` over the CoVoST2 translation, both
with ``asr_text`` set to the transcript. ``text_prev`` is therefore always
``<na>``.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable, Iterator

from egs3.owsm_v4.esp2_s2t.dataset.builder import OWSMBuilder
from egs3.owsm_v4.esp2_s2t.dataset.utils import (
    Utterance,
    generate_long_utterances,
    run_parallel,
    sub_dataset_config,
)

_CONFIG = sub_dataset_config(__name__)
_AUDIO = _CONFIG["builder"]["audio"]

COMMONVOICE_ENV_VAR = str(_CONFIG["builder"]["commonvoice_env_var"])
CV_SPLITS = [str(x) for x in _CONFIG["builder"]["commonvoice_splits"]]
SPLIT_MAP = {
    str(k): [str(x) for x in v]
    for k, v in _CONFIG["builder"]["split_to_covost_split"].items()
}
READ_HEADERS = bool(_AUDIO["read_headers"])
WORKERS = int(_AUDIO["workers"])
BATCH_SIZE = int(_AUDIO["batch_size"])
CHUNK_SIZE = int(_AUDIO["chunk_size"])

#: ``covost_v2.<src>_<tgt>.tsv``; the tar.gz beside each is the same file.
MANIFEST = re.compile(r"^covost_v2\.([a-zA-Z-]+)_([a-zA-Z-]+)\.tsv$")

#: Some CommonVoice rows carry a whole paragraph in one field.
FIELD_SIZE_LIMIT = 50000 * 1024 * 1024


#: The language-independent half of Moses normalize-punctuation.perl, which
#: egs2/covost2/st1/local/data_prep_covost2.sh runs over every transcript
#: before OWSM reads it. Moses is not installed here, so its table is ported;
#: the language-specific rules (number and comma handling for en/de/fr/cs) are
#: not, because nothing in the measured difference needs them.
PUNCTUATION = {
    "\u201e": '"', "\u201c": '"', "\u201d": '"',   # „ “ ”
    "\u00ab": '"', "\u00bb": '"',                  # « »
    "\u2018": "'", "\u2019": "'",                  # ‘ ’
    "\u00b4": "'", "\u0060": "'",                  # ´ `
    "\u2013": "-", "\u2014": "-",                  # – —
    "\u2026": "...",                                # …
    "\u00a0": " ",                                  # non-breaking space
}


def normalize_punctuation(text: str) -> str:
    """Fold typographic punctuation to ASCII, as Moses does.

    Without it the dump's French transcripts match on 58% of clips rather than
    71%, and Russian on 94% rather than 99.6%: Common Voice writes a curly
    apostrophe where the dump has a straight one.
    """
    for source, target in PUNCTUATION.items():
        text = text.replace(source, target)
    return text


def base_language(locale: str) -> str:
    """Return the language a locale is labelled with: ``zh-CN`` is ``zh``."""
    return locale.split("-")[0]


def read_tsv(path: Path) -> Iterator[dict]:
    """Yield the rows of one tab-separated manifest, quoting left alone.

    QUOTE_NONE because these are tab-separated, not comma-separated: a double
    quote is ordinary text, and the default reader strips it. A sentence
    written with quotation marks reaches the dump as the raw field, doubled
    quotes and all, so unquoting it here would rewrite 20% of the transcripts.
    CommonVoice's own sub-dataset does parse the quoting, because
    prepare_commonvoice.py does.
    """
    csv.field_size_limit(FIELD_SIZE_LIMIT)
    with path.open(encoding="utf-8") as stream:
        yield from csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)


def _probe_chunk(task):
    """Measure a run of clips, returning ``[(index, duration, error), ...]``.

    Module-level and argument-only so it pickles to a worker process.
    """
    import soundfile as sf

    first_index, paths = task
    out = []
    for offset, path in enumerate(paths):
        try:
            info = sf.info(str(path))
            out.append((first_index + offset, info.frames / info.samplerate, None))
        except Exception as exc:  # noqa: BLE001 - recorded per file, not raised
            out.append((first_index + offset, 0.0, repr(exc)))
    return out


COVOST_MIRROR = "https://dl.fbaipublicfiles.com/covost"
#: The pairs OWSM builds, as the dump has them.
COVOST_PAIRS = (
    "ar_en", "ca_en", "cy_en", "de_en", "en_ar", "en_ca", "en_cy", "en_de",
    "en_et", "en_fa", "en_id", "en_ja", "en_lv", "en_mn", "en_sl", "en_sv-SE",
    "en_ta", "en_tr", "en_zh-CN", "es_en", "et_en", "fa_en", "fr_en", "id_en",
    "it_en", "ja_en", "lv_en", "mn_en", "nl_en", "pt_en", "ru_en", "sl_en",
    "sv-SE_en", "ta_en", "tr_en", "zh-CN_en",
)


class CoVoST2Builder(OWSMBuilder):
    """Build CoVoST2 split caches from the tsv manifests and CommonVoice."""

    CORPUS = "covost2"

    def prepare_source(self, corpora=None, **_kwargs) -> None:
        """Download CoVoST 2's translation manifests (CC0, ~250 MB).

        Only the manifests: the audio is Common Voice's, and the builder reads
        it from there. Pairs not listed here simply are not fetched.
        """
        from egs3.owsm_v4.esp2_s2t.dataset import download

        dest = self.source_destination(corpora)
        for pair in COVOST_PAIRS:
            url = f"{COVOST_MIRROR}/covost_v2.{pair}.tsv.tar.gz"
            download.fetch_and_extract(url, dest)
        download.require(dest, f"CoVoST 2 download left nothing in {dest}")

    def is_valid_source_root(self, candidate: Path) -> bool:
        """Expect at least one ``covost_v2.<src>_<tgt>.tsv``."""
        return any(MANIFEST.match(p.name) for p in candidate.glob("covost_v2.*.tsv"))

    def pairs(self, source_root: Path, languages: Iterable[str] | None) -> list:
        """Return ``(src, tgt, manifest)`` for each pair, in a stable order.

        ``languages`` filters on the source side, which is how the mixture is
        reduced for a trial build.
        """
        wanted = {str(x) for x in languages} if languages else None
        found = []
        for path in sorted(source_root.glob("covost_v2.*.tsv")):
            match = MANIFEST.match(path.name)
            if match and (wanted is None or match.group(1) in wanted):
                found.append((match.group(1), match.group(2), path))
        return found

    def clip_index(self, commonvoice_root: Path, locale: str) -> dict:
        """Return ``{clip stem: (client_id, transcript)}`` for one locale.

        CoVoST2 names a clip and nothing else, so the speaker and the source
        transcript are read from CommonVoice. Built once per locale and reused
        across that locale's pairs -- all fourteen ``en_X`` pairs share it.
        """
        out: dict = {}
        for cv_split in CV_SPLITS:
            path = commonvoice_root / locale / f"{cv_split}.tsv"
            if not path.is_file():
                continue
            for row in read_tsv(path):
                clip = row.get("path", "").removesuffix(".mp3")
                if clip and clip not in out:
                    out[clip] = (
                        row.get("client_id", ""),
                        normalize_punctuation(row.get("sentence", "")),
                    )
        return out

    def iter_rows(
        self,
        source_root: Path,
        split: str,
        failures: Iterable | None = None,
        limit: int | None = None,
        read_headers: bool | None = None,
        languages: Iterable[str] | None = None,
        commonvoice_dir: str | None = None,
        workers: int | None = None,
        batch_size: int | None = None,
        **_options,
    ) -> Iterator[dict]:
        """Yield cache rows for ``split``, one language pair at a time."""
        import os

        check_header = READ_HEADERS if read_headers is None else read_headers
        n_workers = WORKERS if workers is None else workers
        size = BATCH_SIZE if batch_size is None else batch_size
        commonvoice = Path(
            commonvoice_dir or os.environ.get(COMMONVOICE_ENV_VAR, "")
        )
        if not commonvoice.is_dir():
            raise RuntimeError(
                f"{COMMONVOICE_ENV_VAR} is not a directory: {commonvoice!r}. "
                "CoVoST2 carries no transcripts or speakers of its own; see "
                "the sub-dataset's config.yaml."
            )

        count = 0
        indexes: dict = {}
        durations: dict = {}
        # Every en_X pair carries the same English clips, so the <asr> side is
        # produced once per pair and upstream collapses the copies by utterance
        # id. Doing it here keeps the task ratio right: without it the ASR side
        # is fourteen times too large for English, and the build comes out 69%
        # over the dump. The surviving copy is the first pair in sorted order.
        seen_asr: set = set()

        for src, tgt, manifest in self.pairs(source_root, languages):
            if src not in indexes:
                indexes[src] = self.clip_index(commonvoice, src)
            index = indexes[src]
            clips = commonvoice / src / "clips"
            batch: list = []

            for row in read_tsv(manifest):
                if row.get("split") not in SPLIT_MAP.get(split, [split]):
                    continue
                clip = row["path"].removesuffix(".mp3")
                known = index.get(clip)
                if known is None:
                    if failures is not None:
                        failures.write(
                            json.dumps(
                                {
                                    "utt_id": f"{self.PREFIX}_{clip}",
                                    "error": "clip not in CommonVoice validated.tsv",
                                }
                            )
                            + "\n"
                        )
                    continue
                client_id, transcript = known
                batch.append(
                    (
                        clip,
                        client_id,
                        transcript,
                        normalize_punctuation(row["translation"]),
                        src,
                        tgt,
                    )
                )
                if len(batch) < size:
                    continue
                for out in self._emit(
                    batch, clips, durations, check_header, n_workers,
                    failures, seen_asr,
                ):
                    yield out
                    count += 1
                    if limit is not None and count >= limit:
                        return
                batch = []

            for out in self._emit(
                batch, clips, durations, check_header, n_workers, failures, seen_asr
            ):
                yield out
                count += 1
                if limit is not None and count >= limit:
                    return

    def _emit(
        self, batch, clips, durations, check_header, n_workers, failures, seen_asr
    ):
        """Probe the clips this batch has not measured yet, then yield its rows.

        ``durations`` is keyed by clip across the whole build, so a clip shared
        by several pairs is opened once.
        """
        if not batch:
            return

        if check_header:
            unseen = []
            for clip, *_rest in batch:
                if clip not in durations and clip not in unseen:
                    unseen.append(clip)
            paths = [clips / f"{clip}.mp3" for clip in unseen]
            tasks = [
                (start, paths[start : start + CHUNK_SIZE])
                for start in range(0, len(paths), CHUNK_SIZE)
            ]
            for chunk in run_parallel(
                _probe_chunk, tasks, default_workers=n_workers
            ):
                for index, duration, error in chunk:
                    durations[unseen[index]] = (duration, error)

        for clip, client_id, transcript, translation, src, tgt in batch:
            duration, error = durations.get(clip, (0.0, None))
            if error is None and duration <= 0:
                error = f"duration={duration}"
            if error is not None:
                if failures is not None:
                    failures.write(
                        json.dumps({"utt_id": f"{self.PREFIX}_{clip}", "error": error})
                        + "\n"
                    )
                continue

            wav_id = f"{self.PREFIX}_{client_id}-{clip}"
            wav_path = str(clips / f"{clip}.mp3")
            lang = base_language(src)
            for task, text in (
                ("asr", transcript),
                (f"st_{base_language(tgt)}", translation),
            ):
                if task == "asr":
                    if wav_id in seen_asr:
                        continue
                    seen_asr.add(wav_id)
                utterance = Utterance(
                    utt_id=wav_id,
                    wav_id=wav_id,
                    wav_path=wav_path,
                    start_time=0.0,
                    end_time=duration,
                    lang=f"<{lang}>",
                    task=f"<{task}>",
                    text=text,
                    asr_text=transcript,
                )
                for span in generate_long_utterances([utterance]):
                    yield {
                        "utt_id": span.utt_id,
                        "wav_path": span.wav_path,
                        "start_time": span.start_time,
                        "end_time": span.end_time,
                        "lang": lang,
                        # The utterance id carries the target -- the dump's ST
                        # ids end in st_<tgt> -- but the cache column holds the
                        # bare task, because the <st_xxx> token is composed
                        # from task and tgt_lang when a row is read.
                        "task": "asr" if task == "asr" else "st",
                        "tgt_lang": base_language(tgt) if task != "asr" else "",
                        "text": span.text_with_time,
                        "text_prev": span.prev_text,
                        "text_ctc": span.asr_text,
                    }
