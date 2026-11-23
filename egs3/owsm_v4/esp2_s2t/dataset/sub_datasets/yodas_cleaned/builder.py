"""Cleaned YODAS, from metadata published on the Hugging Face Hub.

The cleaning pipeline (``egs2/owsm_v4/s2t1/local/yodas3*.py``) keeps only
metadata: gzipped JSONL under ``data/<lang>/``, each line a long-form
utterance with its span, its three text streams and a ``wav_path`` of the form
``<source>://<lang>/<shard>/<stem>`` naming the audio shard the recording is
in. That is what gets published, and it is small -- 23 GB for the YODAS v3
production run, against the corpus' hundreds of terabytes of audio.

The audio is not published with it. YODAS' own shards are on the Hub, but not
in a form this recipe can read: YODAS v3 ships WebM, YODAS2 ships 24 kHz WAV,
and OWSM reads 16 kHz mono. So ``prepare_source`` fetches each referenced
shard, extracts only the recordings the metadata names, and transcodes those
to 16 kHz mono Opus. That is the cleaning's own pass 2, run by whoever
consumes the metadata rather than by whoever produced it, and it is the
expensive step: the metadata downloads in minutes, the audio in days.

The published files are re-sharded to about a gigabyte each, so one of them
covers several audio shards and an audio shard can straddle two of them. Audio
is therefore fetched and marked done per *audio* shard, named by ``wav_path``,
after a pass over the whole language's metadata -- never per published file.
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
from egs3.owsm_v4.esp2_s2t.dataset.utils import run_parallel, sub_dataset_config

#: Where each source's audio lives. espnet/yodas3 was renamed to
#: espnet/yodas3.1 and now 307-redirects there; the canonical id is used so a
#: fetch does not depend on the redirect outliving it.
AUDIO_REPO = {
    "yodas2": "espnet/yodas2",
    "yodas3": "espnet/yodas3.1",
}

logger = logging.getLogger(__name__)


def split_wav_path(wav_path: str) -> tuple[str, str, str]:
    """``yodas3://de/0007/abc`` -> ``("de", "0007", "abc")``."""
    _, _, rest = wav_path.partition("://")
    lang, shard, stem = rest.split("/", 2)
    return lang, shard, stem


def transcode(archive: Path, stems: set[str], audio_dir: Path, fs: int) -> None:
    """Extract the named recordings and write them as 16 kHz mono Opus.

    YODAS v3 ships each recording's subtitles as a ``.json`` beside it
    under the same stem, so the member's suffix decides, not its stem.
    """
    audio_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_binary()
    written = set()
    with tarfile.open(archive) as bundle:
        for member in bundle:
            stem = Path(member.name).stem
            if not member.isfile() or stem not in stems:
                continue
            if member.name.endswith(".json"):
                continue
            written.add(stem)
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
                     "-ac", "1", "-ar", str(fs),
                     "-c:a", "libopus", "-b:a", "28k",
                     # The muxer cannot be guessed from the .tmp the write
                     # is staged through.
                     "-f", "opus", str(partial)],
                    capture_output=True,
                )
            if result.returncode != 0:
                partial.unlink(missing_ok=True)
                tail = result.stderr.decode(errors="replace")[-300:]
                raise RuntimeError(f"ffmpeg failed on {member.name}: {tail}")
            os.replace(partial, target)
    missing = sorted(stems - written)
    if missing:
        raise RuntimeError(
            f"{archive} has no audio for {len(missing)} recording(s) the "
            f"metadata names, e.g. {missing[:3]}"
        )


def shard_archive(source: str, name: str, audio_root=None) -> Path:
    """Return a local copy of one shard's archive, fetching it if needed.

    ``audio_root`` is a local mirror of the audio repository laid out the way
    the Hub lays it out, so a cluster that already holds the corpus does not
    download it again. The upstream YODAS2 corpus on disk is exactly this
    shape. A mirror that is set but incomplete raises rather than quietly
    falling back, because the fallback is terabytes.
    """
    from huggingface_hub import hf_hub_download

    if audio_root:
        local = Path(audio_root) / name
        if local.is_file():
            return local
        raise FileNotFoundError(
            f"{source} audio root {audio_root} has no {name}. "
            "Unset it to fetch from the Hub instead."
        )
    return Path(
        hf_hub_download(
            repo_id=AUDIO_REPO[source], repo_type="dataset", filename=name
        )
    )


def prepare_shard(task) -> tuple:
    """Transcode one shard's recordings and mark it done.

    Returns ``(key, error)``; a task that raises returns its message rather
    than propagating, so one unreadable shard out of forty thousand does not
    abandon the rest. ``run_parallel`` pickles this, so it stays module-level
    and single-argument.
    """
    source, name, audio_root, stems, audio_dir, fs, key = task
    audio_dir = Path(audio_dir)
    try:
        archive = shard_archive(source, name, audio_root)
        transcode(archive, set(stems), audio_dir, fs)
    except Exception as error:
        return key, f"{type(error).__name__}: {error}"
    (audio_dir / "DONE").touch()
    return key, None


def ffmpeg_binary() -> str:
    """Return an ffmpeg to transcode with.

    The cluster module fails to load on some GPU nodes, so the static build
    that comes with imageio-ffmpeg is preferred where it is installed; it is
    equivalent to rounding on this audio. ``$FFMPEG`` overrides both.
    """
    override = os.environ.get("FFMPEG")
    if override:
        return override
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise FileNotFoundError(
        "No ffmpeg: neither $FFMPEG, nor imageio-ffmpeg (pip install "
        "imageio-ffmpeg), nor one on PATH. Delta's login and compute nodes "
        "ship none, so this is the usual state of a fresh environment."
    )


class YODASCleanedBuilder(OWSMBuilder):
    """Build a cleaned-YODAS cache from published metadata plus fetched audio.

    This is the shared implementation, not a sub-dataset: it has no
    ``config.yaml`` and is not registered as a ``data_src``. Each corpus is a
    package of its own naming its metadata repository and its source, so that
    YODAS2 and YODAS v3 can appear in one mixture -- which they cannot if the
    source is a module constant.
    """

    #: Filled from the subclass's own config.yaml by __init_subclass__.
    METADATA_REPO: str = ""
    SOURCE: str = ""
    ROWS_DIR: str = ""
    TASK: str = ""

    def __init_subclass__(cls, **kwargs) -> None:
        """Read the per-corpus keys this builder adds to the shared ones."""
        super().__init_subclass__(**kwargs)
        config = sub_dataset_config(cls.__module__)
        if config is None:
            return
        builder = config["builder"]
        cls.METADATA_REPO = str(builder["metadata_repo"])
        cls.SOURCE = str(builder["source"])
        cls.ROWS_DIR = str(builder["rows_dir"])
        cls.TASK = str(config["dataset"]["task"])

    def shard_file(self, lang: str, shard: str) -> str:
        """Path of a shard's audio archive inside its Hub repository.

        YODAS2 splits a language across numbered subsets and the cleaning
        names a shard ``<NNN>_<name>`` after them, so ``de/000_00000000`` is
        ``data/de000/audio/00000000.tar.gz``. YODAS v3 does not split.
        """
        if self.SOURCE == "yodas2":
            number, name = shard.split("_", 1)
            return f"data/{lang}{number}/audio/{name}.tar.gz"
        return f"data/{lang}/audio/{shard}.tar"

    def rows_root(self, source_root: Path) -> Path:
        """Where the published JSONL lives under the source root."""
        return source_root / self.ROWS_DIR if self.ROWS_DIR else source_root

    def is_valid_source_root(self, candidate: Path) -> bool:
        """Expect the published metadata; the audio is checked per shard."""
        return self.rows_root(candidate).is_dir()

    def iter_rows_files(self, source_root: Path, languages=None):
        """``(lang, rows_file)`` for every published file, sorted."""
        wanted = {str(x) for x in languages} if languages else None
        for rows_file in sorted(self.rows_root(source_root).glob("*/*.jsonl.gz")):
            lang = rows_file.parent.name
            if wanted is None or lang in wanted:
                yield lang, rows_file

    def pending_shards(self, source_root: Path, languages=None) -> dict:
        """``{lang: {shard: {stem, ...}}}`` for the audio not yet prepared.

        Only shards without a ``DONE`` are collected, which is what keeps the
        set small enough to hold for a language the size of English.
        """
        pending: dict[str, dict[str, set[str]]] = {}
        done: dict[tuple[str, str], bool] = {}
        for _, rows_file in self.iter_rows_files(source_root, languages):
            for row in self.read_rows(rows_file):
                lang, shard, stem = split_wav_path(row["wav_path"])
                key = (lang, shard)
                if key not in done:
                    done[key] = (source_root / "audio" / lang / shard / "DONE").exists()
                if not done[key]:
                    pending.setdefault(lang, {}).setdefault(shard, set()).add(stem)
        return pending

    def prepare_source(self, corpora=None, **_kwargs) -> None:
        """Download the metadata, then the audio each shard it names needs.

        A shard is marked ``DONE`` only once every recording it names has been
        transcoded, so a job killed at its walltime resumes rather than
        restarts.
        """
        from huggingface_hub import snapshot_download

        options = self.corpus_options(corpora)
        dest = self.source_destination(corpora)
        languages = options.get("languages")
        audio_root = options.get("audio_root") or os.environ.get(
            f"{self.SOURCE.upper()}_AUDIO"
        )

        prefix = f"{self.ROWS_DIR}/" if self.ROWS_DIR else ""
        patterns = (
            [f"{prefix}{lang}/*" for lang in languages]
            if languages
            else [f"{prefix}*"]
        )
        logger.info("downloading %s %s", self.METADATA_REPO, patterns)
        snapshot_download(
            repo_id=self.METADATA_REPO,
            repo_type="dataset",
            local_dir=str(dest),
            allow_patterns=patterns,
        )

        pending = self.pending_shards(dest, languages)
        tasks = [
            (
                self.SOURCE,
                self.shard_file(lang, shard),
                audio_root,
                sorted(stems),
                str(dest / "audio" / lang / shard),
                self.EXPECTED_FS,
                f"{lang}/{shard}",
            )
            for lang in sorted(pending)
            for shard, stems in sorted(pending[lang].items())
        ]
        if not tasks:
            logger.info("%s: every shard is already prepared", self.CORPUS)
            return

        logger.info(
            "%s: %d shards, %d recordings to transcode",
            self.CORPUS,
            len(tasks),
            sum(len(task[3]) for task in tasks),
        )
        failures = [
            (key, error) for key, error in run_parallel(prepare_shard, tasks) if error
        ]
        for key, error in failures[:10]:
            logger.error("%s: %s", key, error)
        if failures:
            raise RuntimeError(
                f"{len(failures)} of {len(tasks)} shards failed, e.g. "
                f"{failures[0][0]}: {failures[0][1]}. The shards that "
                "succeeded are marked DONE, so a rerun resumes."
            )

    @staticmethod
    def read_rows(rows_file: Path) -> Iterator[dict]:
        """Yield each published utterance of one file."""
        with gzip.open(rows_file, "rt", encoding="utf-8") as stream:
            for line in stream:
                yield json.loads(line)

    def iter_rows(
        self,
        source_root: Path,
        split: str,
        failures: Iterable | None = None,
        languages: list[str] | None = None,
        limit: int | None = None,
        **_options,
    ) -> Iterator[dict]:
        """Yield cache rows, skipping the audio shards that are not prepared.

        The published ``wav_path`` is a ``<source>://`` reference, not a file;
        it is resolved here against the audio ``prepare_source`` wrote, so the
        metadata stays portable between machines.
        """
        source_root = source_root.resolve()
        count = 0
        done: dict[tuple[str, str], bool] = {}
        for _, rows_file in self.iter_rows_files(source_root, languages):
            for row in self.read_rows(rows_file):
                lang, shard, stem = split_wav_path(row["wav_path"])
                key = (lang, shard)
                audio_dir = source_root / "audio" / lang / shard
                if key not in done:
                    done[key] = (audio_dir / "DONE").exists()
                    if not done[key] and failures is not None:
                        failures.write(
                            json.dumps(
                                {
                                    "shard": f"{lang}/{shard}",
                                    "error": "audio not prepared for this shard",
                                }
                            )
                            + "\n"
                        )
                if not done[key]:
                    continue
                yield {
                    "utt_id": row["utt_id"],
                    "wav_path": str(audio_dir / f"{stem}.opus"),
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "lang": row["lang"].strip("<>"),
                    "task": self.TASK,
                    "tgt_lang": "",
                    "text": row["text_with_time"],
                    "text_prev": row["prev_text"],
                    "text_ctc": row["asr_text"],
                }
                count += 1
                if limit is not None and count >= limit:
                    return
