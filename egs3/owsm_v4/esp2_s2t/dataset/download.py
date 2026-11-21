"""Fetching the corpora whose licence lets us redistribute a model trained on them.

``OWSMBuilder.prepare_source`` raises by default, because most of OWSM's
corpora are licence-gated or shipped out of band. The permissive ones can be
fetched, and the builders that can do so call into here.

Everything is resumable and idempotent: a complete file is skipped rather than
refetched, a partial one continues with a Range request, and an archive is
unpacked once, recorded by a marker beside it. A job killed at its walltime
therefore costs only what it had left to do.

Nothing here runs on import, and no builder's ``iter_rows`` touches the network.
"""

from __future__ import annotations

import logging
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

# A download of tens of gigabytes over a cluster's shared link is slow enough
# that a half-hour stall is not obviously a hang; fail instead of waiting.
TIMEOUT = 600
CHUNK = 1 << 20


def remote_size(url: str) -> int | None:
    """Return the server's content length, or None when it will not say."""
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            length = response.headers.get("Content-Length")
    except OSError as error:
        logger.warning("HEAD %s failed: %s", url, error)
        return None
    return int(length) if length else None


def fetch(url: str, target: Path) -> Path:
    """Download ``url`` to ``target``, resuming a partial file.

    A server that ignores the Range header restarts the file rather than
    appending to it, which would corrupt the result silently.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = remote_size(url)
    have = target.stat().st_size if target.is_file() else 0
    if expected is not None and have == expected:
        logger.info("%s is already complete (%d bytes)", target.name, have)
        return target

    headers = {}
    mode = "wb"
    if have and expected is not None and have < expected:
        headers["Range"] = f"bytes={have}-"
        mode = "ab"
        logger.info("resuming %s at %d of %d bytes", target.name, have, expected)

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        if mode == "ab" and response.status != 206:
            logger.warning("%s ignored the Range header; restarting", url)
            mode, have = "wb", 0
        with target.open(mode) as handle:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                handle.write(block)

    size = target.stat().st_size
    if expected is not None and size != expected:
        raise OSError(f"{target} is {size} bytes, expected {expected}")
    logger.info("fetched %s (%d bytes)", target.name, size)
    return target


def extract(archive: Path, dest: Path) -> None:
    """Unpack ``archive`` into ``dest`` once, recorded by a marker.

    The marker, not the directory, says an extraction finished: a job killed
    at its walltime leaves a partial tree behind, and skipping on the directory
    alone would call that corpus done.
    """
    marker = dest / f".{archive.name}.extracted"
    if marker.is_file():
        logger.info("%s is already unpacked", archive.name)
        return
    dest.mkdir(parents=True, exist_ok=True)
    logger.info("unpacking %s", archive.name)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(dest)
    else:
        with tarfile.open(archive) as bundle:
            # Untrusted input: it unpacks into its own directory and nothing
            # runs from it. `filter="data"` refuses members with absolute or
            # parent paths, which tarfile allowed by default before 3.12.
            try:
                bundle.extractall(dest, filter="data")
            except TypeError:
                bundle.extractall(dest)
    marker.touch()


def fetch_and_extract(url: str, dest: Path, keep_archive: bool = False) -> None:
    """Download one archive into ``dest`` and unpack it there.

    The marker is checked before the download, not only before the unpack:
    the archive is deleted once it has been unpacked, so a run that looked
    only at the file would fetch all 82 GiB of LibriSpeech again to discover
    there was nothing left to do.
    """
    name = Path(url).name
    if (dest / f".{name}.extracted").is_file():
        logger.info("%s is already unpacked", name)
        return
    archive = fetch(url, dest / name)
    extract(archive, dest)
    if not keep_archive:
        archive.unlink(missing_ok=True)


def fetch_many(urls: dict[str, Path]) -> None:
    """Download ``{url: target}`` in order, skipping what is already complete."""
    for index, (url, target) in enumerate(urls.items(), start=1):
        logger.info("[%d/%d] %s", index, len(urls), url)
        fetch(url, target)


def require(root: Path, message: str) -> None:
    """Raise with ``message`` unless something was written under ``root``."""
    if not root.is_dir() or not any(root.iterdir()):
        raise FileNotFoundError(message)


def disk_free(path: Path) -> int:
    """Bytes free on the filesystem holding ``path``, for a pre-flight check."""
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free
