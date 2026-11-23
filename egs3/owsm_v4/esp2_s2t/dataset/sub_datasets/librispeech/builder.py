"""LibriSpeech as OWSM utterances.

Ported from ``egs2/owsm_v1/s2t1/local/prepare_librispeech.py``. One chapter is
one group, so ``generate_long_utterances`` packs its consecutive sentences
into 30 s spans along the chapter timeline.

It needs the separate "original mp3" release for the chapter audio and the
sentence offsets; the per-utterance flac release has neither, and without the
offsets the spans cannot be reconstructed from it.
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
MP3_ENV_VAR = str(_BUILDER["mp3_env_var"])
MP3_SUBDIR = str(_BUILDER["mp3_subdir"])
SPLIT_SOURCES = {
    str(k): [str(x) for x in v] for k, v in _BUILDER["split_sources"].items()
}
SEGMENTS_SUFFIX = str(_BUILDER["segments_suffix"])
TRANSCRIPT_SUFFIX = str(_BUILDER["transcript_suffix"])
LOWERCASE = bool(_BUILDER["lowercase"])


def read_pairs(segments: Path, transcripts: Path) -> Iterator[tuple]:
    """Yield ``(utterance id, start, end, text)`` for one chapter.

    The two files are one line per sentence in the same order; upstream
    asserts the ids line up, so a disagreement is a corpus problem rather than
    something to skip past quietly.
    """
    with segments.open(encoding="utf-8") as seg_f, transcripts.open(
        encoding="utf-8"
    ) as trans_f:
        for seg, trans in zip(seg_f, trans_f):
            seg, trans = seg.strip(), trans.strip()
            if not seg or not trans:
                continue
            utt_id, start, end = seg.split()
            trans_id, _, text = trans.partition(" ")
            if utt_id != trans_id:
                raise RuntimeError(
                    f"{segments} and {transcripts} disagree: "
                    f"{utt_id!r} against {trans_id!r}"
                )
            yield utt_id, float(start), float(end), text


OPENSLR = "https://openslr.trmal.net/resources"


class LibriSpeechBuilder(OWSMBuilder):
    """Build LibriSpeech split caches from the chapter mp3s and sentence files."""

    CORPUS = "librispeech"

    def flac_root(self, source_root: Path) -> Path:
        """Return the directory holding the ``<split>-<condition>`` trees."""
        nested = source_root / "LibriSpeech"
        return nested if (nested / "dev-clean").is_dir() else source_root

    def mp3_root(self, source_root: Path, mp3_dir: str | None = None) -> Path:
        """Return the directory holding ``mp3/<speaker>/<chapter>/``.

        ``original-mp3.tar.gz`` unpacks beside the flac release's split
        directories, so the corpus root is the last candidate rather than an
        error: a machine that downloaded both has them in one tree.
        """
        import os

        for candidate in (mp3_dir, os.environ.get(MP3_ENV_VAR), source_root):
            if not candidate:
                continue
            root = Path(candidate)
            nested = root / "LibriSpeech"
            if (nested / MP3_SUBDIR).is_dir():
                return nested / MP3_SUBDIR
            if (root / MP3_SUBDIR).is_dir():
                return root / MP3_SUBDIR
        return Path(source_root) / MP3_SUBDIR

    def mp3_destination(self, corpora=None) -> Path:
        """Where the original-mp3 release should be downloaded to."""
        import os

        configured = self.corpus_options(corpora).get("mp3_dir") or os.environ.get(
            MP3_ENV_VAR
        )
        return Path(configured) if configured else self.source_destination(corpora)

    def prepare_source(self, corpora=None, **_kwargs) -> None:
        """Download both LibriSpeech releases from OpenSLR (about 140 GiB).

        The flac release says which speakers are in which split; only
        ``original-mp3`` carries the chapter audio and the sentence offsets
        OWSM cuts its 30 s spans at. Neither alone is enough, and both unpack
        into a shared ``LibriSpeech/`` directory.
        """
        from egs3.owsm_v4.esp2_s2t.dataset import download

        dest = self.source_destination(corpora)
        for names in SPLIT_SOURCES.values():
            for name in names:
                download.fetch_and_extract(f"{OPENSLR}/12/{name}.tar.gz", dest)
        download.fetch_and_extract(
            f"{OPENSLR}/12/original-mp3.tar.gz", self.mp3_destination(corpora)
        )
        download.require(dest, f"LibriSpeech download left nothing in {dest}")

    def is_source_prepared(self, corpora=None, **_kwargs) -> bool:
        """Require the flac split directories *and* the mp3 tree."""
        if not super().is_source_prepared(corpora=corpora, **_kwargs):
            return False
        options = self.corpus_options(corpora)
        source_root = self.resolve_source_root(options.get("source_dir"))
        return self.mp3_root(source_root, options.get("mp3_dir")).is_dir()

    def is_valid_source_root(self, candidate: Path) -> bool:
        """Expect the flac release's split directories."""
        root = self.flac_root(candidate)
        return all(
            (root / name).is_dir()
            for names in SPLIT_SOURCES.values()
            for name in names
        )

    def speakers(self, source_root: Path, split: str) -> list:
        """Return the speakers of ``split``, from the flac release.

        The split only names the speakers; every chapter they have in the mp3
        tree is taken, which is why the counts do not match the flac release's.
        """
        root = self.flac_root(source_root)
        found = []
        for name in SPLIT_SOURCES.get(split, [split]):
            directory = root / name
            if not directory.is_dir():
                continue
            found.extend(
                path.name for path in sorted(directory.iterdir()) if path.is_dir()
            )
        return sorted(set(found))

    def iter_rows(
        self,
        source_root: Path,
        split: str,
        failures: Iterable | None = None,
        limit: int | None = None,
        mp3_dir: str | None = None,
        **_options,
    ) -> Iterator[dict]:
        """Yield cache rows for ``split``, one chapter at a time."""
        mp3 = self.mp3_root(source_root, mp3_dir)
        if not mp3.is_dir():
            raise RuntimeError(
                f"No {MP3_SUBDIR}/ tree at {mp3!r}. It is the original-mp3 "
                f"release; set {MP3_ENV_VAR} or create_dataset.corpora."
                "librispeech.mp3_dir, or let prepare_source fetch it."
            )
        count = 0

        for speaker in self.speakers(source_root, split):
            speaker_dir = mp3 / speaker
            if not speaker_dir.is_dir():
                continue
            for chapter_dir in sorted(speaker_dir.iterdir()):
                if not chapter_dir.is_dir():
                    continue
                chapter = chapter_dir.name
                stem = f"{speaker}-{chapter}"
                segments = chapter_dir / f"{stem}{SEGMENTS_SUFFIX}"
                transcripts = chapter_dir / f"{stem}{TRANSCRIPT_SUFFIX}"
                audio = chapter_dir / f"{chapter}.mp3"
                wav_id = f"{self.PREFIX}_{speaker}_{chapter}"

                missing = [
                    path.name
                    for path in (segments, transcripts, audio)
                    if not path.is_file()
                ]
                if missing:
                    if failures is not None:
                        failures.write(
                            json.dumps(
                                {"utt_id": wav_id, "error": f"missing {missing}"}
                            )
                            + "\n"
                        )
                    continue

                utterances = []
                for utt_id, start, end, text in read_pairs(segments, transcripts):
                    text = " ".join(text.split())
                    if LOWERCASE:
                        text = text.lower()
                    if not text or end <= start:
                        continue
                    utterances.append(
                        Utterance(
                            utt_id=f"{self.PREFIX}_{utt_id}",
                            wav_id=wav_id,
                            wav_path=str(audio),
                            start_time=start,
                            end_time=end,
                            lang=f"<{LANG}>",
                            task=f"<{TASK}>",
                            text=text,
                            asr_text=text,
                        )
                    )
                if not utterances:
                    continue

                for span in generate_long_utterances(utterances):
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
