"""Cleaned YODAS, from metadata published on the Hugging Face Hub.

The cleaning pipeline (``egs2/owsm_v4/s2t1/local/yodas3*.py``) keeps only
metadata: one gzipped JSONL per language and shard, each line a long-form
utterance with its span, its three text streams and a ``wav_path`` of the form
``<source>://<lang>/<shard>/<stem>``. That is what gets published, and it is
small -- the whole YODAS2 production run is 12 GB.

The audio is not published with it. YODAS' own shards are on the Hub, but not
in a form this recipe can read: YODAS v3 ships WebM, YODAS2 ships 24 kHz WAV,
and OWSM reads 16 kHz mono. So ``prepare_source`` fetches each referenced
shard, extracts only the recordings the metadata names, and transcodes those
to 16 kHz mono Opus. That is the cleaning's own pass 2, run by whoever
consumes the metadata rather than by whoever produced it, and it is the
expensive step: the metadata downloads in minutes, the audio in days.

``sub_datasets/yodas3`` is the sibling of this builder for the case where the
cleaning ran locally and pass 2 has already written the audio.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable, Iterator

from egs3.owsm_v4.esp2_s2t.dataset.builder import OWSMBuilder
from egs3.owsm_v4.esp2_s2t.dataset.utils import sub_dataset_config

_CONFIG = sub_dataset_config(__name__)
_BUILDER = _CONFIG["builder"]

TASK = str(_CONFIG["dataset"]["task"])
METADATA_REPO = str(_BUILDER["metadata_repo"])
SOURCE = str(_BUILDER["source"])
RULE = str(_BUILDER["rule"])
AUDIO_REPO = {
    "yodas2": "espnet/yodas2",
    "yodas3": "espnet/yodas3",
}

logger = logging.getLogger(__name__)


def split_wav_path(wav_path: str) -> tuple[str, str, str]:
    """``yodas2://de/000_00000000/abc`` -> ``("de", "000_00000000", "abc")``."""
    _, _, rest = wav_path.partition("://")
    lang, shard, stem = rest.split("/", 2)
    return lang, shard, stem


def shard_file(lang: str, shard: str) -> str:
    """Path of a shard's audio archive inside its Hub repository.

    YODAS2 splits a language across numbered subsets and the cleaning names a
    shard ``<NNN>_<name>`` after them, so ``de/000_00000000`` is
    ``data/de000/audio/00000000.tar.gz``. YODAS v3 does not split.
    """
    if SOURCE == "yodas2":
        number, name = shard.split("_", 1)
        return f"data/{lang}{number}/audio/{name}.tar.gz"
    return f"data/{lang}/audio/{shard}.tar"


def ffmpeg_binary() -> str:
    """Return an ffmpeg to transcode with.

    The cluster module fails to load on some GPU nodes, so the static build
    that comes with imageio-ffmpeg is preferred where it is installed; it is
    equivalent to rounding on this audio.
    """
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg") or "ffmpeg"


class YODASCleanedBuilder(OWSMBuilder):
    """Build the cleaned-YODAS cache from published metadata plus fetched audio."""

    CORPUS = "yodas_cleaned"

    def is_valid_source_root(self, candidate: Path) -> bool:
        """Expect the published metadata; the audio is checked per shard."""
        return (candidate / "rows").is_dir()

    def iter_shards(self, source_root: Path, langs=None):
        """``(lang, shard, rows_file)`` for every published shard, sorted."""
        for rows_file in sorted((source_root / "rows").glob("*/*.jsonl.gz")):
            lang = rows_file.parent.name
            if langs is None or lang in langs:
                yield lang, rows_file.name.removesuffix(".jsonl.gz"), rows_file

    def prepare_source(self, corpora=None, **_kwargs) -> None:
        """Download the metadata, then the audio each shard needs.

        Audio is fetched shard by shard and a shard is marked ``DONE`` only
        once every recording it names has been transcoded, so a job killed at
        its walltime resumes rather than restarts.
        """
        from huggingface_hub import hf_hub_download, snapshot_download

        options = self.corpus_options(corpora)
        dest = self.source_destination(corpora)
        languages = options.get("languages")

        rows = dest / "rows"
        if not rows.is_dir():
            logger.info("downloading %s", METADATA_REPO)
            snapshot_download(
                repo_id=METADATA_REPO,
                repo_type="dataset",
                local_dir=str(dest),
                allow_patterns=[f"{RULE}/*"] if RULE else None,
            )
            if not rows.is_dir() and (dest / RULE).is_dir():
                (dest / RULE).rename(rows)

        for lang, shard, rows_file in self.iter_shards(dest, languages):
            audio_dir = dest / "audio" / lang / shard
            if (audio_dir / "DONE").exists():
                continue
            stems = {
                split_wav_path(row["wav_path"])[2]
                for row in self.read_rows(rows_file)
            }
            logger.info("%s/%s: %d recordings", lang, shard, len(stems))
            archive = hf_hub_download(
                repo_id=AUDIO_REPO[SOURCE],
                repo_type="dataset",
                filename=shard_file(lang, shard),
            )
            self.transcode(Path(archive), stems, audio_dir)
            (audio_dir / "DONE").touch()

    @staticmethod
    def read_rows(rows_file: Path) -> Iterator[dict]:
        """Yield each published utterance of one shard."""
        with gzip.open(rows_file, "rt", encoding="utf-8") as stream:
            for line in stream:
                yield json.loads(line)

    def transcode(self, archive: Path, stems: set[str], audio_dir: Path) -> None:
        """Extract the named recordings and write them as 16 kHz mono Opus."""
        audio_dir.mkdir(parents=True, exist_ok=True)
        ffmpeg = ffmpeg_binary()
        with tarfile.open(archive) as bundle:
            for member in bundle:
                stem = Path(member.name).stem
                if not member.isfile() or stem not in stems:
                    continue
                target = audio_dir / f"{stem}.opus"
                if target.is_file():
                    continue
                source = bundle.extractfile(member)
                if source is None:
                    continue
                with tempfile.NamedTemporaryFile(
                    dir=audio_dir, suffix=Path(member.name).suffix
                ) as raw:
                    shutil.copyfileobj(source, raw)
                    raw.flush()
                    partial = target.with_suffix(".opus.tmp")
                    result = subprocess.run(
                        [ffmpeg, "-nostdin", "-y", "-i", raw.name,
                         "-ac", "1", "-ar", str(self.EXPECTED_FS),
                         "-c:a", "libopus", "-b:a", "28k", str(partial)],
                        capture_output=True,
                    )
                if result.returncode != 0:
                    partial.unlink(missing_ok=True)
                    tail = result.stderr.decode(errors="replace")[-300:]
                    raise RuntimeError(f"ffmpeg failed on {member.name}: {tail}")
                os.replace(partial, target)

    def iter_rows(
        self,
        source_root: Path,
        split: str,
        failures: Iterable | None = None,
        langs: list[str] | None = None,
        limit: int | None = None,
        **_options,
    ) -> Iterator[dict]:
        """Yield cache rows, shard by shard, skipping shards without audio.

        The published ``wav_path`` is a ``<source>://`` reference, not a file;
        it is resolved here against the audio ``prepare_source`` wrote, so the
        metadata stays portable between machines.
        """
        source_root = source_root.resolve()
        count = 0
        for lang, shard, rows_file in self.iter_shards(source_root, langs):
            audio_dir = source_root / "audio" / lang / shard
            if not (audio_dir / "DONE").exists():
                if failures is not None:
                    failures.write(
                        json.dumps(
                            {
                                "shard": f"{lang}/{shard}",
                                "error": "audio not prepared for this shard",
                            }
                        )
                        + "\n"
                    )
                continue
            for row in self.read_rows(rows_file):
                _, _, stem = split_wav_path(row["wav_path"])
                yield {
                    "utt_id": row["utt_id"],
                    "wav_path": str(audio_dir / f"{stem}.opus"),
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "lang": row["lang"].strip("<>"),
                    "task": TASK,
                    "tgt_lang": "",
                    "text": row["text_with_time"],
                    "text_prev": row["prev_text"],
                    "text_ctc": row["asr_text"],
                }
                count += 1
                if limit is not None and count >= limit:
                    return
