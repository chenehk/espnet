"""VCTK as OWSM utterances.

There is no ``prepare_vctk.py``: OWSM runs the corpus through
``egs2/vctk/asr1/local/data_prep_speaker_closed.sh`` and then
``egs2/owsm_v3/s2t1/local/kaldi_to_whisper.py``. Both layers are reproduced
here against the raw corpus -- the Kaldi directory is never materialised.

Each utterance is its own recording, trimmed to its speech, so each forms its
own group and ``text_prev`` is always ``<na>``.
"""

from __future__ import annotations

import json
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

LANG = str(_CONFIG["dataset"]["lang"])
TASK = str(_CONFIG["dataset"]["task"])
CORPUS_SUBDIR = str(_BUILDER["corpus_subdir"])
EXCLUDE = {str(x) for x in _BUILDER["exclude_speakers"]}
NUM_DEV = int(_BUILDER["num_dev"])
NUM_EVAL = int(_BUILDER["num_eval"])
LABEL_SUBDIR = str(_BUILDER["label_subdir"])
UNITS_PER_SECOND = int(_BUILDER["label_units_per_second"])
PAUSE = str(_BUILDER["pause_symbol"])


def speech_bounds(label_path: Path) -> tuple:
    """Return ``(start, end)`` in seconds, trimmed to the speech.

    The label is ``start end symbol`` per line in 100 ns units. Upstream walks
    forward while the *next* symbol is a pause and takes the current line's
    end, then walks backward while the *previous* symbol is a pause and takes
    the current line's start -- so a recording that is pause throughout has no
    bounds.
    """
    rows = []
    with label_path.open(encoding="utf-8") as stream:
        for line in stream:
            parts = line.split()
            if len(parts) >= 3:
                rows.append((int(parts[0]), int(parts[1]), parts[2]))
    if not rows:
        return None

    start = None
    for index in range(len(rows) - 1):
        if rows[index + 1][2] != PAUSE:
            start = rows[index][1]
            break
    end = None
    for index in range(len(rows) - 1, 0, -1):
        if rows[index - 1][2] != PAUSE:
            end = rows[index][0]
            break
    if start is None or end is None:
        return None
    return start / UNITS_PER_SECOND, end / UNITS_PER_SECOND


VCTK_URL = "https://datashare.ed.ac.uk/download/DS_10283_3443.zip"


class VCTKBuilder(OWSMBuilder):
    """Build VCTK split caches from wav48, txt and the mono labels."""

    CORPUS = "vctk"

    def corpus_root(self, source_root: Path) -> Path:
        """Return the directory holding ``wav48``, ``txt`` and ``lab``."""
        nested = source_root / CORPUS_SUBDIR
        return nested if (nested / "wav48").is_dir() else source_root

    def prepare_source(self, corpora=None, **_kwargs) -> None:
        """Download VCTK from Edinburgh DataShare (ODC-By, 11.7 GB)."""
        from egs3.owsm_v4.esp2_s2t.dataset import download

        dest = self.source_destination(corpora)
        download.fetch_and_extract(VCTK_URL, dest)
        download.require(dest, f"VCTK download left nothing in {dest}")

    def is_valid_source_root(self, candidate: Path) -> bool:
        """Expect the three per-speaker trees beside each other."""
        root = self.corpus_root(candidate)
        return all(
            (root / name).is_dir() for name in ("wav48", "txt", LABEL_SUBDIR)
        )

    def speakers(self, root: Path) -> list:
        """Return the speakers to build, in the order upstream walks them.

        A speaker without a label directory ends the upstream script, so one
        missing here is a corpus problem rather than something to skip over.
        """
        found = sorted(
            path.name
            for path in (root / "wav48").iterdir()
            if path.is_dir() and path.name.startswith("p")
        )
        return [name for name in found if name not in EXCLUDE]

    def split_utterances(self, stems: list, split: str) -> list:
        """Return the stems belonging to ``split`` for one speaker.

        ``subset_data_dir.sh --last`` then ``--first``/``--last``: the last
        ``num_dev + num_eval`` are held out, the first of those are eval and
        the last are dev, and everything before them is train.
        """
        held = NUM_DEV + NUM_EVAL
        if split == "train":
            return stems[: max(0, len(stems) - held)]
        deveval = stems[-held:] if len(stems) >= held else stems
        if split == "dev":
            return deveval[-NUM_DEV:]
        return deveval[:NUM_EVAL]

    def iter_rows(
        self,
        source_root: Path,
        split: str,
        failures: Iterable | None = None,
        limit: int | None = None,
        **_options,
    ) -> Iterator[dict]:
        """Yield cache rows for ``split``, one speaker at a time."""
        root = self.corpus_root(source_root)
        count = 0

        for speaker in self.speakers(root):
            labels = root / LABEL_SUBDIR / speaker
            texts = root / "txt" / speaker
            # upstream walks the wav files and skips any without both a label
            # and a transcript, so the split boundaries are over what survives.
            # The two companions are listed once per speaker rather than
            # stat-ed per utterance: that is 2 reads instead of 2 x 40,000,
            # which on Lustre is the difference between seconds and an hour.
            have_lab = {path.stem for path in labels.glob("*.lab")}
            have_txt = {path.stem for path in texts.glob("*.txt")}
            stems = [
                path.stem
                for path in sorted((root / "wav48" / speaker).glob("*.wav"))
                if path.stem in have_lab and path.stem in have_txt
            ]

            for stem in self.split_utterances(stems, split):
                utt_id = f"{self.PREFIX}_{stem}"
                bounds = speech_bounds(labels / f"{stem}.lab")
                text = " ".join(
                    (texts / f"{stem}.txt").read_text(encoding="utf-8").split()
                )

                error = None
                if not text:
                    error = "empty text"
                elif bounds is None:
                    error = "no speech in the label"
                elif bounds[1] <= bounds[0]:
                    error = f"empty span {bounds}"

                if error is not None:
                    if failures is not None:
                        failures.write(
                            json.dumps({"utt_id": utt_id, "error": error}) + "\n"
                        )
                    continue

                utterance = Utterance(
                    utt_id=utt_id,
                    wav_id=utt_id,
                    wav_path=str(root / "wav48" / speaker / f"{stem}.wav"),
                    start_time=bounds[0],
                    end_time=bounds[1],
                    lang=f"<{LANG}>",
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
                        "lang": LANG,
                        "task": TASK,
                        "tgt_lang": "",
                        "text": span.text_with_time,
                        "text_prev": span.prev_text,
                        "text_ctc": span.asr_text,
                    }
                    count += 1
                    if limit is not None and count >= limit:
                        return
