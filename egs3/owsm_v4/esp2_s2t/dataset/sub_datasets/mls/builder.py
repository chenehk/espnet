"""Multilingual LibriSpeech as OWSM utterances.

Ported from ``egs2/owsm_v2/s2t1/local/prepare_multilingual_librispeech.py``.
``segments.txt`` cuts each LibriVox recording into utterances; they are grouped
back by recording so ``generate_long_utterances`` packs consecutive ones into
30 s spans, which is why OWSM needs the long audio rather than MLS's own
per-utterance files.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Iterable, Iterator

from egs3.owsm_v4.esp2_s2t.dataset.builder import OWSMBuilder
from egs3.owsm_v4.esp2_s2t.dataset.utils import (
    LANGUAGES,
    run_parallel,
    Utterance,
    generate_long_utterances,
    sub_dataset_config,
)

_CONFIG = sub_dataset_config(__name__)
_BUILDER = _CONFIG["builder"]

TASK = str(_CONFIG["dataset"]["task"])
LANGS = [str(x) for x in _BUILDER["languages"]]
LONG_AUDIO_ENV_VAR = str(_BUILDER["long_audio_env_var"])
LONG_AUDIO_SUFFIX = str(_BUILDER["long_audio_suffix"])
LONG_AUDIO_RATE = int(_BUILDER["long_audio_rate"])
AUDIO_KEY = str(_BUILDER["audio_key"])


def recording_key(url: str, how: str = AUDIO_KEY) -> str:
    """Return the name a recording's audio is stored and keyed under.

    Args:
        url: The LibriVox mp3 URL from ``segments.txt``.
        how: ``stem`` for the mp3 basename, as upstream and the dump use, or
            ``url`` for a digest of the whole URL, which is unique.
    """
    stem = url.rsplit("/", 1)[-1].removesuffix(".mp3")
    if how == "stem":
        return stem
    return f"{stem}_{hashlib.blake2b(url.encode(), digest_size=4).hexdigest()}"


def read_transcripts(path: Path) -> dict:
    """Return ``{utterance id: transcript}`` for one split."""
    out = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            utt_id, transcript = line.strip().split(maxsplit=1)
            out[utt_id] = " ".join(transcript.split())
    return out


def read_segments(path: Path) -> Iterator[tuple]:
    """Yield ``(utterance id, url, start, end)`` for one split, in file order."""
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            utt_id, url, start, end = line.split()
            yield utt_id, url, float(start), float(end)


logger = logging.getLogger(__name__)


def build_long_recording(task) -> int:
    """Rebuild one recording from MLS's own utterances; return how many it laid.

    Must stay module-level and single-argument: ``run_parallel`` pickles it.
    """
    import os

    import numpy as np
    import soundfile as sf

    out_path, segments, audio_dir = Path(task[0]), task[1], Path(task[2])
    if out_path.is_file():
        return 0

    total = int(round(max(end for _utt, _start, end in segments) * LONG_AUDIO_RATE))
    timeline = np.zeros(total, dtype="float32")
    laid = 0
    for utt_id, start, _end in segments:
        speaker, book, _ = utt_id.split("_")
        source = audio_dir / speaker / book / f"{utt_id}.flac"
        try:
            chunk, rate = sf.read(str(source), dtype="float32")
        except Exception:
            continue
        if rate != LONG_AUDIO_RATE:
            continue
        offset = int(round(start * LONG_AUDIO_RATE))
        chunk = chunk[: max(0, total - offset)]
        timeline[offset : offset + len(chunk)] = chunk
        laid += 1

    temporary = out_path.with_suffix(f"{LONG_AUDIO_SUFFIX}.{os.getpid()}.tmp")
    sf.write(str(temporary), timeline, LONG_AUDIO_RATE, format="FLAC",
             subtype="PCM_16")
    os.replace(temporary, out_path)
    return laid


MLS_MIRROR = "https://dl.fbaipublicfiles.com/mls"
# The archives are named for the language, not its code.
MLS_ARCHIVE_LANGUAGES = (
    "english",
    "german",
    "dutch",
    "french",
    "spanish",
    "italian",
    "portuguese",
    "polish",
)


class MLSBuilder(OWSMBuilder):
    """Build MLS split caches from segments.txt and the long audio."""

    CORPUS = "mls"

    def prepare_source(self, corpora=None, **_kwargs) -> None:
        """Download MLS from Facebook's mirror (CC BY 4.0, ~2.4 TB for all eight).

        This brings the shipped per-utterance audio. The long audio OWSM cuts
        its spans from is rebuilt from it separately -- see MLS_LONG_AUDIO.
        """
        from egs3.owsm_v4.esp2_s2t.dataset import download

        dest = self.source_destination(corpora)
        wanted = self.corpus_options(corpora).get("languages")
        # LANGUAGES maps a code to its archive name; the fetch iterates names.
        codes = {name: code for code, name in LANGUAGES.items()}
        for language in MLS_ARCHIVE_LANGUAGES:
            if wanted and codes.get(language, language) not in wanted:
                continue
            download.fetch_and_extract(f"{MLS_MIRROR}/mls_{language}.tar.gz", dest)
        download.require(dest, f"MLS download left nothing in {dest}")
        self.build_long_audio(corpora)

    def long_audio_destination(self, corpora=None) -> Path:
        """Where the rebuilt recordings go."""
        import os

        configured = self.corpus_options(corpora).get("long_audio_dir") or os.environ.get(
            LONG_AUDIO_ENV_VAR
        )
        if not configured:
            raise FileNotFoundError(
                f"Nowhere to write MLS long audio. Set {LONG_AUDIO_ENV_VAR} or "
                "pass create_dataset.corpora.mls.long_audio_dir."
            )
        return Path(configured)

    def long_audio_plan(self, corpora=None):
        """Yield ``(split_dir, out_dir, tasks)`` for each language and split.

        The plan is also the completeness test: a recording is one task, and
        one already on disk is skipped by the worker, so a partial tree costs
        only what it has left.
        """
        options = self.corpus_options(corpora)
        source_root = self.resolve_source_root(options.get("source_dir"))
        out_root = self.long_audio_destination(corpora)
        how = options.get("audio_key") or AUDIO_KEY

        for lang in self.languages(source_root, options.get("languages")):
            name = f"mls_{LANGUAGES[lang]}"
            for split in self.requested_splits(options):
                split_dir = source_root / name / split
                if not (split_dir / "segments.txt").is_file():
                    continue
                by_recording: dict = {}
                for utt_id, url, start, end in read_segments(
                    split_dir / "segments.txt"
                ):
                    by_recording.setdefault(recording_key(url, how), []).append(
                        (utt_id, start, end)
                    )
                out_dir = out_root / name / split
                tasks = []
                for key, rows in by_recording.items():
                    rows.sort(key=lambda row: row[1])
                    target = out_dir / f"{key}{LONG_AUDIO_SUFFIX}"
                    if not target.is_file():
                        tasks.append((str(target), rows, str(split_dir / "audio")))
                yield split_dir, out_dir, tasks

    def build_long_audio(self, corpora=None) -> None:
        """Rebuild the recordings OWSM cuts its spans from.

        Upstream re-downloads 203,626 LibriVox originals from archive.org;
        this lays MLS's own utterances on a silent timeline at their
        ``segments.txt`` offsets instead, which reproduces each recording
        wherever a span can fall. English alone is 180k recordings, so this
        runs through ``run_parallel`` and honours the recipe's ``parallel``
        block.
        """
        for split_dir, out_dir, tasks in self.long_audio_plan(corpora):
            if not tasks:
                logger.info("%s: long audio is complete", out_dir)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            logger.info("%s: rebuilding %d recordings", out_dir, len(tasks))
            laid = sum(run_parallel(build_long_recording, tasks))
            logger.info(
                "%s: %d recordings written, %d utterances laid down",
                out_dir,
                len(tasks),
                laid,
            )

    def is_source_prepared(self, corpora=None, **_kwargs) -> bool:
        """Require the corpus *and* every recording it cuts spans from."""
        if not super().is_source_prepared(corpora=corpora, **_kwargs):
            return False
        try:
            return not any(tasks for _dir, _out, tasks in self.long_audio_plan(corpora))
        except FileNotFoundError:
            return False

    def is_valid_source_root(self, candidate: Path) -> bool:
        """Expect at least one prepared language directory."""
        return any(
            (candidate / f"mls_{LANGUAGES[lang]}" / "dev" / "segments.txt").is_file()
            for lang in LANGS
            if lang in LANGUAGES
        )

    def languages(self, source_root: Path, languages: Iterable[str] | None) -> list:
        """Return the languages to build, in upstream's order."""
        wanted = [str(x) for x in languages] if languages else LANGS
        return [
            lang
            for lang in wanted
            if lang in LANGUAGES
            and (source_root / f"mls_{LANGUAGES[lang]}").is_dir()
        ]

    def groups(self, split_dir: Path, lang: str, audio_dir: Path, how: str):
        """Yield one recording's utterances at a time, in first-seen order.

        A recording whose long audio is missing is skipped whole: its spans
        cannot be cut, and emitting them against an absent file would move the
        failure to training time.
        """
        transcripts = read_transcripts(split_dir / "transcripts.txt")
        by_recording: dict = {}
        for utt_id, url, start, end in read_segments(split_dir / "segments.txt"):
            by_recording.setdefault(recording_key(url, how), []).append(
                (utt_id, start, end)
            )

        for key, segments in by_recording.items():
            segments.sort(key=lambda row: row[1])
            path = audio_dir / f"{key}{LONG_AUDIO_SUFFIX}"
            yield key, path, [
                Utterance(
                    utt_id=f"{self.PREFIX}_{lang}_{utt_id}",
                    wav_id=f"{self.PREFIX}_{lang}_{key}",
                    wav_path=str(path),
                    start_time=start,
                    end_time=end,
                    lang=f"<{lang}>",
                    task=f"<{TASK}>",
                    text=transcripts[utt_id],
                    asr_text=transcripts[utt_id],
                )
                for utt_id, start, end in segments
                if utt_id in transcripts
            ]

    def iter_rows(
        self,
        source_root: Path,
        split: str,
        failures: Iterable | None = None,
        limit: int | None = None,
        languages: Iterable[str] | None = None,
        long_audio_dir: str | None = None,
        audio_key: str | None = None,
        **_options,
    ) -> Iterator[dict]:
        """Yield cache rows for ``split``, one language at a time."""
        import os

        how = audio_key or AUDIO_KEY
        root = Path(long_audio_dir or os.environ.get(LONG_AUDIO_ENV_VAR, ""))
        if not root.is_dir():
            raise RuntimeError(
                f"{LONG_AUDIO_ENV_VAR} is not a directory: {root!r}. "
                "prepare_source writes the long audio there; see the "
                "sub-dataset's config.yaml."
            )
        count = 0

        for lang in self.languages(source_root, languages):
            name = f"mls_{LANGUAGES[lang]}"
            split_dir = source_root / name / split
            if not (split_dir / "segments.txt").is_file():
                continue
            audio_dir = root / name / split

            for key, path, utterances in self.groups(split_dir, lang, audio_dir, how):
                if not utterances:
                    continue
                if not path.is_file():
                    if failures is not None:
                        failures.write(
                            json.dumps(
                                {
                                    "utt_id": f"{self.PREFIX}_{lang}_{key}",
                                    "error": f"no long audio at {path}",
                                }
                            )
                            + "\n"
                        )
                    continue
                for span in generate_long_utterances(utterances):
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
