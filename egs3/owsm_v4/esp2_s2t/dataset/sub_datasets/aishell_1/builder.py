"""AISHELL-1 as OWSM utterances.

Ported from ``egs2/owsm_v1/s2t1/local/prepare_aishell.py``. Each wav holds one
read sentence with one transcript, so every utterance forms its own group and
``text_prev`` is always ``<na>``.
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
    run_parallel,
    sub_dataset_config,
)

_CONFIG = sub_dataset_config(__name__)
_AUDIO_CFG = _CONFIG["builder"]["audio"]

TASK = str(_CONFIG["dataset"]["task"])
HEADER_BYTES = int(_AUDIO_CFG["header_bytes"])
BYTES_PER_SECOND = int(_AUDIO_CFG["bytes_per_second"])
READ_HEADERS = bool(_AUDIO_CFG["read_headers"])

TRANSCRIPT = Path("data_aishell") / "transcript" / "aishell_transcript_v0.8.txt"
WAV_DIR = Path("data_aishell") / "wav"


def duration_from_filesize(size: int) -> float:
    """Seconds of audio in a ``size``-byte file of this corpus' fixed format."""
    return (size - HEADER_BYTES) / BYTES_PER_SECOND


def read_transcripts(source_root: Path) -> dict[str, str]:
    """Return ``{utterance name: text}`` for the whole corpus, splits together.

    The text is joined without spaces, as upstream does: the file is written
    word-segmented, but OWSM scores Mandarin by character.
    """
    name2text = {}
    with (source_root / TRANSCRIPT).open(encoding="utf-8") as stream:
        for line in stream:
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            name, text = parts
            name2text[name] = "".join(text.split())
    if not name2text:
        raise RuntimeError(f"No transcripts in {source_root / TRANSCRIPT}")
    return name2text


def _probe_file(task):
    """Measure one file, returning ``(index, size, error)``.

    Module-level and argument-only so it can be pickled to a worker.

    Args:
        task: ``(index, path, read_header, expected_fs)``.
    """
    index, path, read_header, expected_fs = task
    try:
        size = os.stat(path).st_size
        if read_header:
            import soundfile as sf

            info = sf.info(str(path))
            if info.samplerate != expected_fs:
                raise ValueError(
                    f"sample rate is {info.samplerate}, expected {expected_fs}"
                )
            from_header = info.frames / info.samplerate
            from_size = duration_from_filesize(size)
            if round(1000 * from_header) != round(1000 * from_size):
                raise ValueError(
                    f"header says {from_header:.3f}s, filesize says {from_size:.3f}s"
                )
    except Exception as exc:  # noqa: BLE001 - recorded per file, not raised
        return index, 0, repr(exc)
    return index, size, None


OPENSLR = "https://openslr.trmal.net/resources"


class AISHELLBuilder(OWSMBuilder):
    """Build AISHELL-1 split caches from the raw transcript and audio tree."""

    CORPUS = "aishell_1"

    def prepare_source(self, corpora=None, **_kwargs) -> None:
        """Download AISHELL-1 from OpenSLR (Apache-2.0, ~15 GB).

        The speech archive nests a second layer of per-speaker tars, which
        the corpus expects unpacked in place.
        """
        from egs3.owsm_v4.esp2_s2t.dataset import download

        dest = self.source_destination(corpora)
        for name in ("data_aishell.tgz", "resource_aishell.tgz"):
            download.fetch_and_extract(f"{OPENSLR}/33/{name}", dest)
        for inner in sorted((dest / "data_aishell" / "wav").glob("*.tar.gz")):
            download.extract(inner, inner.parent)
            inner.unlink()
        download.require(dest, f"AISHELL-1 download left nothing in {dest}")

    def is_valid_source_root(self, candidate: Path) -> bool:
        """Expect the transcript file and a ``wav/<split>/`` tree per split."""
        return (candidate / TRANSCRIPT).is_file() and all(
            (candidate / WAV_DIR / split).is_dir() for split in self.SPLITS
        )

    def iter_rows(
        self,
        source_root: Path,
        split: str,
        failures: Iterable | None = None,
        limit: int | None = None,
        read_headers: bool | None = None,
        **_options,
    ) -> Iterator[dict]:
        """Yield cache rows for ``split``, in utterance-id order.

        Upstream globs the speaker directories and lets ``sort -k1,1 -u`` order
        the result afterwards; sorting here gives the same order without the
        second pass, and keeps the cache stable across filesystems.

        ``limit`` takes the first N files, so the offline dump comparison can
        run without a built cache.
        """
        name2text = read_transcripts(source_root)
        # The corpus ships audio for utterances the transcript omits. Upstream
        # drops them silently, so they are not failures.
        paths = sorted(
            path
            for path in (source_root / WAV_DIR / split).glob("*/*.wav")
            if path.name.removesuffix(".wav") in name2text
        )
        if limit is not None:
            paths = paths[:limit]

        tasks = [
            (
                index,
                path,
                READ_HEADERS if read_headers is None else read_headers,
                self.EXPECTED_FS,
            )
            for index, path in enumerate(paths)
        ]
        measured = {
            index: (size, error)
            for index, size, error in run_parallel(_probe_file, tasks)
        }

        for index, path in enumerate(paths):
            name = path.name.removesuffix(".wav")
            utt_id = f"{self.PREFIX}_{name}"
            size, error = measured[index]
            duration = duration_from_filesize(size) if error is None else 0.0
            text = name2text[name]
            if error is None and (duration <= 0 or not text):
                error = f"duration={duration}, text={text!r}"
            if error is not None:
                if failures is not None:
                    failures.write(
                        json.dumps({"utt_id": utt_id, "error": error}) + "\n"
                    )
                continue

            utterance = Utterance(
                utt_id=utt_id,
                wav_id=utt_id,
                wav_path=str(path.resolve()),
                start_time=0.0,
                end_time=duration,
                lang=f"<{self.LANG}>",
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
                    "lang": self.LANG,
                    "task": TASK,
                    "tgt_lang": "",
                    "text": span.text_with_time,
                    "text_prev": span.prev_text,
                    "text_ctc": span.asr_text,
                }
