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
from pathlib import Path
from typing import Iterable, Iterator

from egs3.owsm_v4.esp2_s2t.dataset.builder import OWSMBuilder
from egs3.owsm_v4.esp2_s2t.dataset.utils import (
    LANGUAGES,
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
                f"{LONG_AUDIO_ENV_VAR} is not a directory: {root!r}. The long "
                "audio is built by submit/build_mls_audio_long.sh; see the "
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
