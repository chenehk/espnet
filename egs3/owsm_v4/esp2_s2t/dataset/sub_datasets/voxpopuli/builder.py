"""VoxPopuli as OWSM utterances.

Ported from ``egs2/owsm_v3/s2t1/local/prepare_voxpopuli.py``. A parliamentary
session is one long recording segmented by the annotation, so a session is one
group and ``text_prev`` carries the span before.

Two upstream behaviours are reproduced deliberately rather than corrected,
because the published data was built with them and the dump is the oracle:

* the ST path logs "the utterance is too short. Skip" and then keeps the
  utterance -- the ``continue`` is missing, where the ASR path has it;
* ST reads the audio of the *target* language channel,
  ``raw_audios/<tgt>/<year>/<event>_<tgt>.ogg``, while labelling the row with
  the source language. ASR reads ``raw_audios/original/`` as expected.
"""

from __future__ import annotations

import csv
import gzip
import json
from ast import literal_eval
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

from egs3.owsm_v4.esp2_s2t.dataset.builder import OWSMBuilder
from egs3.owsm_v4.esp2_s2t.dataset.utils import (
    Utterance,
    generate_long_utterances,
    iso3,
    sub_dataset_config,
)

_CONFIG = sub_dataset_config(__name__)

ASR_LANGUAGES = tuple(str(x) for x in _CONFIG["builder"]["asr_languages"])
ST_TARGET_LANGUAGES = tuple(str(x) for x in _CONFIG["builder"]["st_target_languages"])
READ_HEADERS = bool(_CONFIG["builder"]["audio"]["read_headers"])

#: Some annotation rows carry a whole session's transcript in one field.
FIELD_SIZE_LIMIT = 50000 * 1024 * 1024

#: Shorter than this is not a usable span. Applied on the ASR path only, as
#: upstream applies it.
MIN_DURATION = 0.1


def preprocess_text(text: str) -> str:
    """Collapse whitespace, replacing any stray tag brackets with a space.

    Upstream neither drops the segment nor raises -- it warns and substitutes,
    so a transcript that genuinely contains ``<`` survives in altered form.
    """
    if "<" in text or ">" in text:
        text = text.replace("<", " ").replace(">", " ")
    return " ".join(text.split())


def parse_src_id(id_: str) -> tuple[str, str, str]:
    """Split an s2s reference id into ``(event_id, lang, utt_id)``."""
    event_id, utt_id = id_.split("_", 1)
    event_id, lang = event_id.rsplit("-", 1)
    return event_id, lang, utt_id


def read_asr_rows(path: Path) -> Iterator[dict]:
    """Yield the annotation rows of one language's ``asr_<lang>.tsv.gz``."""
    csv.field_size_limit(FIELD_SIZE_LIMIT)
    with gzip.open(path, "rt") as stream:
        yield from csv.DictReader(stream, delimiter="|")


def read_s2s_rows(path: Path) -> Iterator[dict]:
    """Yield the reference translations of one target language."""
    csv.field_size_limit(FIELD_SIZE_LIMIT)
    with gzip.open(path, "rt") as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def annotation_text(row: dict) -> str:
    """Return the row's transcript, preferring the original over the normed."""
    if row["original_text"].strip():
        return row["original_text"]
    return row["normed_text"]


VOXPOPULI_BASE = "https://dl.fbaipublicfiles.com/voxpopuli"
#: The European Parliament sessions VoxPopuli covers.
VOXPOPULI_YEARS = tuple(range(2009, 2021))


class VoxPopuliBuilder(OWSMBuilder):
    """Build VoxPopuli split caches from the shipped annotations."""

    CORPUS = "voxpopuli"

    def asr_annotation(self, source_root: Path, lang: str) -> Path:
        """Return the ASR annotation shipped for ``lang``."""
        return source_root / "transcribed_data" / lang / f"asr_{lang}.tsv.gz"

    def s2s_annotation(self, source_root: Path, tgt: str) -> Path:
        """Return the reference translations shipped for ``tgt``."""
        return source_root / "transcribed_data" / tgt / f"s2s_{tgt}_ref.tsv.gz"

    def prepare_source(self, corpora=None, **_kwargs) -> None:
        """Download VoxPopuli's original audio and ASR annotations (CC0).

        The original, unsegmented audio is what OWSM cuts spans from: one tar
        per year, about 5.7 GB each, 2009 to 2020. The annotations are a few
        megabytes per language. Roughly 70 GB in all.
        """
        from egs3.owsm_v4.esp2_s2t.dataset import download

        options = self.corpus_options(corpora)
        dest = self.source_destination(corpora)
        wanted = options.get("languages")
        years = options.get("years") or VOXPOPULI_YEARS
        audio = dest / "raw_audios"
        for year in years:
            download.fetch_and_extract(
                f"{VOXPOPULI_BASE}/audios/original_{year}.tar", audio
            )
        for lang in ASR_LANGUAGES:
            if wanted and lang not in wanted:
                continue
            download.fetch(
                f"{VOXPOPULI_BASE}/annotations/asr/asr_{lang}.tsv.gz",
                self.asr_annotation(dest, lang),
            )
        download.require(dest, f"VoxPopuli download left nothing in {dest}")

    def is_valid_source_root(self, candidate: Path) -> bool:
        """Expect the original audio tree and at least one ASR annotation."""
        return (candidate / "raw_audios" / "original").is_dir() and any(
            self.asr_annotation(candidate, lang).is_file() for lang in ASR_LANGUAGES
        )

    def _check_audio(self, path: Path, read_header: bool) -> str | None:
        """Return an error string for ``path``, or None when it is usable."""
        if not path.is_file():
            return f"audio not found: {path}"
        if not read_header:
            return None
        import soundfile as sf

        try:
            rate = sf.info(str(path)).samplerate
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            return repr(exc)
        if rate != self.EXPECTED_FS:
            return f"sample rate is {rate}, expected {self.EXPECTED_FS}"
        return None

    def asr_groups(
        self,
        source_root: Path,
        split: str,
        lang: str,
        failures: Iterable | None,
        read_header: bool,
    ) -> Iterator[tuple[str, list[Utterance]]]:
        """Yield ``(session_id, utterances)`` for one language's ASR side."""
        annotation = self.asr_annotation(source_root, lang)
        if not annotation.is_file():
            return
        audio_root = source_root / "raw_audios" / "original"
        iso_src = iso3(lang)
        groups: dict[str, list[Utterance]] = defaultdict(list)
        checked: dict[Path, str | None] = {}

        for row in read_asr_rows(annotation):
            if row["split"] != split:
                continue
            event_id = row["session_id"]
            audio_path = audio_root / event_id[:4] / f"{event_id}_original.ogg"

            if audio_path not in checked:
                checked[audio_path] = self._check_audio(audio_path, read_header)
            error = checked[audio_path]
            if error is not None:
                if failures is not None:
                    failures.write(
                        json.dumps({"wav_id": event_id, "error": error}) + "\n"
                    )
                continue

            text = preprocess_text(annotation_text(row))
            if not text:
                continue

            # Upstream takes the VAD bounds rather than start_time/end_time.
            vad = literal_eval(row["vad"])
            start_time, end_time = float(vad[0][0]), float(vad[-1][-1])
            if end_time - start_time < MIN_DURATION:
                continue

            groups[event_id].append(
                Utterance(
                    utt_id=f"{self.PREFIX}_asr_{event_id}_{row['id_']}",
                    wav_id=f"{self.PREFIX}_asr_{event_id}",
                    # Upstream stores an `ffmpeg ... |` Kaldi pipe; soundfile
                    # reads the ogg directly and can seek into it.
                    wav_path=str(audio_path),
                    start_time=start_time,
                    end_time=end_time,
                    lang=f"<{iso_src}>",
                    task="<asr>",
                    text=text,
                    asr_text=text,
                )
            )

        for event_id, utterances in groups.items():
            yield event_id, utterances

    def st_groups(
        self,
        source_root: Path,
        split: str,
        src: str,
        tgt: str,
        failures: Iterable | None,
        read_header: bool,
    ) -> Iterator[tuple[str, list[Utterance]]]:
        """Yield ``(session_id, utterances)`` for one source/target pair."""
        asr_path = self.asr_annotation(source_root, src)
        s2s_path = self.s2s_annotation(source_root, tgt)
        if not asr_path.is_file() or not s2s_path.is_file():
            return

        source_text = {
            f"{row['session_id']}-{row['id_']}": (
                annotation_text(row),
                row["split"],
            )
            for row in read_asr_rows(asr_path)
        }

        # The ST audio is the target language's channel, not the original.
        audio_root = source_root / "raw_audios" / tgt
        iso_src, iso_tgt = iso3(src), iso3(tgt)
        groups: dict[str, list[Utterance]] = defaultdict(list)
        checked: dict[Path, str | None] = {}

        for row in read_s2s_rows(s2s_path):
            event_id, row_src, utt_id = parse_src_id(row["id"])
            if row_src != src:
                continue

            src_text, row_split = source_text.get(row["id"], (None, None))
            if src_text is None or row_split != split:
                continue
            src_text = preprocess_text(src_text)
            if not src_text:
                continue

            tgt_text = preprocess_text(row["tgt_text"])
            if not tgt_text:
                continue

            audio_path = audio_root / event_id[:4] / f"{event_id}_{tgt}.ogg"
            if audio_path not in checked:
                checked[audio_path] = self._check_audio(audio_path, read_header)
            error = checked[audio_path]
            if error is not None:
                if failures is not None:
                    failures.write(
                        json.dumps({"wav_id": event_id, "error": error}) + "\n"
                    )
                continue

            # No minimum duration here: upstream logs one and keeps the row.
            groups[event_id].append(
                Utterance(
                    utt_id=(
                        f"{self.PREFIX}_st_{iso_src}2{iso_tgt}_{event_id}_{utt_id}"
                    ),
                    wav_id=f"{self.PREFIX}_st_{iso_src}2{iso_tgt}_{event_id}",
                    wav_path=str(audio_path),
                    start_time=float(row["start_time"]),
                    end_time=float(row["end_time"]),
                    lang=f"<{iso_src}>",
                    task=f"<st_{iso_tgt}>",
                    text=tgt_text,
                    asr_text=src_text,
                )
            )

        for event_id, utterances in groups.items():
            yield event_id, utterances

    def iter_rows(
        self,
        source_root: Path,
        split: str,
        failures: Iterable | None = None,
        limit: int | None = None,
        read_headers: bool | None = None,
        asr_languages: Iterable[str] | None = None,
        st_target_languages: Iterable[str] | None = None,
        **_options,
    ) -> Iterator[dict]:
        """Yield cache rows for ``split``: every ASR language, then every pair.

        ``limit`` caps the rows yielded, so the offline dump comparison can run
        without a built cache.
        """
        check_header = READ_HEADERS if read_headers is None else read_headers
        sources = sorted(str(x) for x in (asr_languages or ASR_LANGUAGES))
        targets = sorted(str(x) for x in (st_target_languages or ST_TARGET_LANGUAGES))
        count = 0

        def emit(groups, lang_code, task, tgt_code):
            nonlocal count
            for _event_id, utterances in groups:
                for span in generate_long_utterances(utterances):
                    # Counted before the yield: the caller's limit check runs
                    # while this generator is suspended here.
                    count += 1
                    yield {
                        "utt_id": span.utt_id,
                        "wav_path": span.wav_path,
                        "start_time": span.start_time,
                        "end_time": span.end_time,
                        "lang": lang_code,
                        "task": task,
                        "tgt_lang": tgt_code,
                        "text": span.text_with_time,
                        "text_prev": span.prev_text,
                        "text_ctc": span.asr_text,
                    }

        for src in sources:
            groups = self.asr_groups(source_root, split, src, failures, check_header)
            for row in emit(groups, iso3(src), "asr", ""):
                yield row
                if limit is not None and count >= limit:
                    return

        for src in sources:
            for tgt in targets:
                groups = self.st_groups(
                    source_root, split, src, tgt, failures, check_header
                )
                for row in emit(groups, iso3(src), "st", iso3(tgt)):
                    yield row
                    if limit is not None and count >= limit:
                        return
