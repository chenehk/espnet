"""FLEURS as OWSM utterances.

Ported from ``egs2/owsm_v3/s2t1/local/prepare_fleurs.py``. The corpus is
distributed through HuggingFace rather than as a directory of files, so the
cache is the source; nothing is downloaded at build time.

Every clip is one read sentence with one transcript, so each utterance forms
its own group and ``text_prev`` is always ``<na>``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Iterator

from egs3.owsm_v4.esp2_s2t.dataset.builder import OWSMBuilder
from egs3.owsm_v4.esp2_s2t.dataset.utils import (
    Utterance,
    generate_long_utterances,
    sub_dataset_config,
)

_CONFIG = sub_dataset_config(__name__)
_BUILDER = _CONFIG["builder"]

TASK = str(_CONFIG["dataset"]["task"])
HF_DATASET = str(_BUILDER["hf_dataset"])
HF_CONFIG = str(_BUILDER["hf_config"])
HF_SPLITS = {str(k): str(v) for k, v in _BUILDER["hf_splits"].items()}
PATH_MARKER = str(_BUILDER["path_marker"])
SAMPLES_PER_SECOND = int(_BUILDER["samples_per_second"])
LANGUAGE_MAP = {str(k): str(v) for k, v in _BUILDER["language_map"].items()}


def preprocess_text(text: str) -> str:
    """Collapse whitespace, replacing any stray tag brackets with a space.

    Upstream neither drops the row nor raises -- it warns and substitutes.
    """
    if "<" in text or ">" in text:
        text = text.replace("<", " ").replace(">", " ")
    return " ".join(text.split())


def rehang(path: str, source_root: Path) -> str:
    """Return ``path`` with everything left of the marker replaced.

    The cache records absolute paths from wherever the corpus was first
    downloaded. Only the part from ``/downloads/extracted/`` rightwards
    identifies the file.
    """
    index = path.find(PATH_MARKER)
    if index < 0:
        return path
    return str(source_root) + path[index:]


class FLEURSBuilder(OWSMBuilder):
    """Build FLEURS split caches from the HuggingFace cache."""

    CORPUS = "fleurs"

    def is_valid_source_root(self, candidate: Path) -> bool:
        """Expect the dataset's cache directory under this root."""
        owner, _, name = HF_DATASET.partition("/")
        return (candidate / f"{owner}___{name}").is_dir()

    def load_split(self, source_root: Path, split: str):
        """Return the cached split, with the audio column left undecoded.

        Decoding would need a codec package and read every clip; the manifest
        already carries ``num_samples``.
        """
        import datasets

        datasets.disable_progress_bars()
        loaded = datasets.load_dataset(
            HF_DATASET, HF_CONFIG, cache_dir=str(source_root)
        )
        name = HF_SPLITS.get(split, split)
        return loaded[name].cast_column("audio", datasets.Audio(decode=False))

    def iter_rows(
        self,
        source_root: Path,
        split: str,
        failures: Iterable | None = None,
        limit: int | None = None,
        **_options,
    ) -> Iterator[dict]:
        """Yield cache rows for ``split``, one clip at a time."""
        table = self.load_split(source_root, split)
        names = table.features["lang_id"].names
        count = 0

        columns = ["path", "lang_id", "num_samples", "raw_transcription"]
        for row in table.select_columns(columns):
            speech_id = os.path.basename(row["path"]).removesuffix(".wav")
            utt_id = f"{self.PREFIX}_{speech_id}"
            text = preprocess_text(row["raw_transcription"])
            path = rehang(row["path"], source_root)
            lang = LANGUAGE_MAP.get(names[row["lang_id"]])

            error = None
            if not text:
                error = "empty text"
            elif lang is None:
                error = f"unknown language {names[row['lang_id']]}"
            elif not Path(path).is_file():
                error = f"no audio at {path}"

            if error is not None:
                if failures is not None:
                    failures.write(
                        json.dumps({"utt_id": utt_id, "error": error}) + "\n"
                    )
                continue

            utterance = Utterance(
                utt_id=utt_id,
                wav_id=utt_id,
                wav_path=path,
                start_time=0.0,
                end_time=row["num_samples"] / SAMPLES_PER_SECOND,
                lang=f"<{lang}>",
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
                    "lang": lang,
                    "task": TASK,
                    "tgt_lang": "",
                    "text": span.text_with_time,
                    "text_prev": span.prev_text,
                    "text_ctc": span.asr_text,
                }
                count += 1
                if limit is not None and count >= limit:
                    return
