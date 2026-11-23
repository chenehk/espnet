"""CommonVoice as OWSM utterances.

Ported from ``egs2/owsm_v3/s2t1/local/prepare_commonvoice.py``. Each clip is
one read sentence with one transcript, so every utterance forms its own group
and ``text_prev`` is always ``<na>``.

The corpus ships ``validated``, ``dev`` and ``test``; OWSM derives train as
validated minus every speaker appearing in dev or test. That is a plain set
difference on ``client_id``, with no shuffle and no seed, so unlike
ReazonSpeech the published partition is reproducible exactly.
"""

from __future__ import annotations

import csv
import json
import logging
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
_AUDIO_CFG = _CONFIG["builder"]["audio"]

TASK = str(_CONFIG["dataset"]["task"])
LANGUAGE_MAP = {str(k): str(v) for k, v in _CONFIG["builder"]["language_map"].items()}
MAX_TEXT_LENGTH = int(_CONFIG["builder"]["max_text_length"])
HELD_OUT_FROM_TRAIN = [str(x) for x in _CONFIG["builder"]["held_out_from_train"]]
READ_HEADERS = bool(_AUDIO_CFG["read_headers"])
WORKERS = int(_AUDIO_CFG["workers"])
BATCH_SIZE = int(_AUDIO_CFG["batch_size"])
CHUNK_SIZE = int(_AUDIO_CFG["chunk_size"])

#: Some annotation rows carry a whole paragraph in one field.
FIELD_SIZE_LIMIT = 50000 * 1024 * 1024


def preprocess_text(text: str) -> str:
    """Collapse whitespace, replacing any stray tag brackets with a space.

    Upstream neither drops the row nor raises -- it warns and substitutes, so
    a transcript that genuinely contains ``<`` survives in altered form.
    """
    if "<" in text or ">" in text:
        text = text.replace("<", " ").replace(">", " ")
    return " ".join(text.split())


def read_tsv(path: Path) -> Iterator[dict]:
    """Yield the rows of one CommonVoice tsv."""
    csv.field_size_limit(FIELD_SIZE_LIMIT)
    with path.open(encoding="utf-8") as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def _probe_chunk(task):
    """Measure a run of clips, returning ``[(index, duration, error), ...]``.

    A chunk rather than a file because a header read costs about as much as
    scheduling it. Module-level and argument-only so it pickles to a worker.

    Args:
        task: ``(first_index, [path, ...], expected_fs)``.
    """
    import soundfile as sf

    first_index, paths, expected_fs = task
    out = []
    for offset, path in enumerate(paths):
        try:
            info = sf.info(str(path))
            # The clips are mp3 at the corpus' own rate, which the read path
            # resamples; only the duration is read here.
            out.append((first_index + offset, info.frames / info.samplerate, None))
        except Exception as exc:  # noqa: BLE001 - recorded per file, not raised
            out.append((first_index + offset, 0.0, repr(exc)))
    return out


logger = logging.getLogger(__name__)

HF_DATASET = str(_CONFIG["builder"]["hf_dataset"])
RELEASE = str(_CONFIG["builder"]["release"])


class CommonVoiceBuilder(OWSMBuilder):
    """Build CommonVoice split caches from the per-language tsv files."""

    CORPUS = "commonvoice"

    def corpus_root(self, candidate: Path) -> Path:
        """Return the directory holding ``<lang>/validated.tsv``.

        An archive unpacks to ``<release>/<lang>/``, so a tree built by
        ``prepare_source`` carries one level the reader does not want. Both
        that tree and a release directory pointed at directly are accepted.
        """
        nested = candidate / RELEASE
        if any((nested / lang).is_dir() for lang in LANGUAGE_MAP):
            return nested
        return candidate

    def prepare_source(self, corpora=None, **_kwargs) -> None:
        """Use the corpus where it already is, else unpack it from the mirror.

        Mozilla serves each language from a signed URL that cannot be
        scripted, so a machine that already holds a release should use it.
        The fallback is a mirror of the release archives, one per language,
        which are unpacked here and then deleted: the reader wants the
        extracted tsv tree, and keeping both would double 585 GB.
        """
        from egs3.owsm_v4.esp2_s2t.dataset import download

        dest = self.source_destination(corpora)
        if dest.is_dir() and self.is_valid_source_root(dest):
            logger.info("%s already holds Common Voice", dest)
            return

        options = self.corpus_options(corpora)
        repo = str(options.get("hf_dataset") or HF_DATASET)
        wanted = options.get("languages")
        names = sorted(str(x) for x in wanted) if wanted else sorted(LANGUAGE_MAP)
        patterns = [
            f"{RELEASE}-{name}.tar.gz"
            for name in names
            if not (dest / f".{RELEASE}-{name}.tar.gz.extracted").is_file()
        ]

        if patterns:
            try:
                from huggingface_hub import snapshot_download

                snapshot_download(
                    repo_id=repo,
                    repo_type="dataset",
                    local_dir=str(dest),
                    allow_patterns=patterns,
                )
            except Exception as error:
                raise FileNotFoundError(
                    f"Common Voice is not at {dest} and the mirror could not "
                    f"be fetched ({type(error).__name__}: {error}).\n"
                    f"Either point {self.SOURCE_ENV_VAR} at an existing "
                    f"release, or get access to "
                    f"https://huggingface.co/datasets/{repo} and log in "
                    "(huggingface-cli login, or $HF_TOKEN)."
                ) from error

            if not any(dest.glob(f"{RELEASE}-*.tar.gz")):
                raise FileNotFoundError(
                    f"{repo} served no {RELEASE}-*.tar.gz for {len(patterns)} "
                    f"requested languages. Check the codes against the "
                    "repository's file list."
                )

        for archive in sorted(dest.glob(f"{RELEASE}-*.tar.gz")):
            download.extract(archive, dest)
            archive.unlink()

        download.require(dest, f"Common Voice download left nothing in {dest}")

    def is_valid_source_root(self, candidate: Path) -> bool:
        """Expect at least one mapped language directory with a validated.tsv."""
        root = self.corpus_root(candidate)
        return any(
            (root / lang / "validated.tsv").is_file() for lang in LANGUAGE_MAP
        )

    def languages(self, source_root: Path, languages: Iterable[str] | None) -> list:
        """Return the language directories to build, in a stable order."""
        root = self.corpus_root(source_root)
        wanted = (
            sorted(str(x) for x in languages) if languages else sorted(LANGUAGE_MAP)
        )
        return [
            lang
            for lang in wanted
            if lang in LANGUAGE_MAP and (root / lang / "validated.tsv").is_file()
        ]

    def split_rows(self, lang_dir: Path, split: str, held_out=None) -> list:
        """Return the annotation rows of one language's ``split``.

        ``dev`` is the corpus' own file. ``train`` is validated minus every
        speaker appearing in the ``held_out`` splits.

        The two sources disagree on which those are.
        ``prepare_commonvoice.py`` holds out dev *and* test, then writes three
        sets; the published dump carries only dev and train, and its train
        holds the test speakers -- for Catalan, all 16,380 test clips plus the
        151 further validated clips by those speakers. Holding out dev alone
        reproduces the dump; adding ``test`` reproduces the egs2 script.
        """
        if split != "train":
            path = lang_dir / f"{split}.tsv"
            return list(read_tsv(path)) if path.is_file() else []

        speakers = set()
        for other in held_out or HELD_OUT_FROM_TRAIN:
            path = lang_dir / f"{other}.tsv"
            if path.is_file():
                speakers.update(row["client_id"] for row in read_tsv(path))
        return [
            row
            for row in read_tsv(lang_dir / "validated.tsv")
            if row["client_id"] not in speakers
        ]

    def iter_rows(
        self,
        source_root: Path,
        split: str,
        failures: Iterable | None = None,
        limit: int | None = None,
        read_headers: bool | None = None,
        languages: Iterable[str] | None = None,
        workers: int | None = None,
        batch_size: int | None = None,
        held_out_from_train: Iterable[str] | None = None,
        **_options,
    ) -> Iterator[dict]:
        """Yield cache rows for ``split``, one language directory at a time.

        Durations are read in batches so an eleven-million-row build holds
        only one batch of tasks and results at a time.
        """
        check_header = READ_HEADERS if read_headers is None else read_headers
        n_workers = WORKERS if workers is None else workers
        size = BATCH_SIZE if batch_size is None else batch_size
        count = 0

        root = self.corpus_root(source_root)
        for lang in self.languages(root, languages):
            iso = LANGUAGE_MAP[lang]
            lang_dir = root / lang
            clips = lang_dir / "clips"
            batch: list = []

            for row in self.split_rows(lang_dir, split, held_out_from_train):
                text = preprocess_text(row["sentence"])
                if not text or len(text) > MAX_TEXT_LENGTH:
                    continue
                batch.append((row["path"], text, clips / row["path"], iso))
                if len(batch) < size:
                    continue
                for out in self._emit(batch, check_header, n_workers, failures):
                    yield out
                    count += 1
                    if limit is not None and count >= limit:
                        return
                batch = []

            for out in self._emit(batch, check_header, n_workers, failures):
                yield out
                count += 1
                if limit is not None and count >= limit:
                    return

    def _emit(self, batch, check_header, n_workers, failures) -> Iterator[dict]:
        """Probe one batch of clips and yield their cache rows."""
        if not batch:
            return

        if check_header:
            paths = [path for _name, _text, path, _iso in batch]
            tasks = [
                (start, paths[start : start + CHUNK_SIZE], self.EXPECTED_FS)
                for start in range(0, len(paths), CHUNK_SIZE)
            ]
            measured = {
                index: (duration, error)
                for chunk in run_parallel(
                    _probe_chunk, tasks, default_workers=n_workers
                )
                for index, duration, error in chunk
            }
        else:
            measured = {index: (0.0, None) for index in range(len(batch))}

        for index, (name, text, path, iso) in enumerate(batch):
            duration, error = measured[index]
            if error is None and duration <= 0:
                error = f"duration={duration}"
            if error is not None:
                if failures is not None:
                    failures.write(
                        json.dumps({"utt_id": name, "error": error}) + "\n"
                    )
                continue

            utterance = Utterance(
                # The id upstream passes is the speaker, but the packer mints
                # the final one from wav_id, so the clip name is what ends up
                # in it -- which is what the dump carries.
                utt_id=f"{self.PREFIX}_{name}",
                wav_id=f"{self.PREFIX}_{name}",
                # Upstream stores an `ffmpeg ... |` Kaldi pipe to resample;
                # libsndfile reads the mp3 directly and the read path
                # resamples.
                wav_path=str(path),
                start_time=0.0,
                end_time=duration,
                lang=f"<{iso}>",
                task=f"<{TASK}>",
                text=text,
                asr_text=text,
            )
            for span in generate_long_utterances([utterance]):
                yield {
                    "utt_id": span.utt_id,
                    "wav_path": span.wav_path,
                    "start_time": span.start_time,
                    "end_time": span.end_time,
                    "lang": iso,
                    "task": TASK,
                    "tgt_lang": "",
                    "text": span.text_with_time,
                    "text_prev": span.prev_text,
                    "text_ctc": span.asr_text,
                }
