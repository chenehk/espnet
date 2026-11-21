#!/usr/bin/env python3
r"""Publish a prepared corpus or its metadata to a Hugging Face dataset repo.

Two things get published from this recipe, for two different reasons:

  yodas-metadata  the cleaning's filtered rows -- one gzipped JSONL per
                  language and shard, no audio. 12 GB for the YODAS2
                  production run, and the input ``sub_datasets/yodas_cleaned``
                  reads.
  commonvoice     a Common Voice release a machine already holds, so a cluster
                  that cannot reach Mozilla's signed download URLs can still
                  get it.

  upload_to_hub.py yodas-metadata --src /work/.../yodas2_all/filtered \\
      --repo espnet/yodas_cleaned_owsm_v5 --rule filtered
  upload_to_hub.py commonvoice --src /work/.../cv-corpus-13.0-2023-03-09 \\
      --repo <org>/common_voice_13_0_mirror

Needs a token with write access (``huggingface-cli login`` or ``$HF_TOKEN``).
Nothing is uploaded without ``--yes``: these are public by default, and a
corpus whose licence forbids redistribution must not be pushed. Of OWSM's
corpora only the permissive ones may be -- see local/corpus_sources.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Licences that allow redistributing the corpus or a derivative of it. The
# verdict comes from the same registry the audit table is built from, so a
# corpus demoted there cannot be uploaded from here by accident.
ALLOWED_VERDICTS = {"usable"}


def check_licence(corpus: str) -> None:
    """Refuse to publish a corpus whose licence does not allow it."""
    from egs3.owsm_v4.esp2_s2t.local.corpus_sources import LICENCES

    entry = LICENCES.get(corpus)
    if entry is None:
        raise SystemExit(f"{corpus} has no licence entry; refusing to upload")
    licence, verdict = entry[0], entry[1]
    if verdict not in ALLOWED_VERDICTS:
        raise SystemExit(
            f"{corpus} is {licence} ({verdict}); redistributing it is not "
            f"allowed. Nothing was uploaded."
        )
    print(f"{corpus}: {licence} ({verdict}) -- redistribution allowed")


def upload(src: Path, repo: str, path_in_repo: str | None, private: bool) -> None:
    """Create the repository if needed and push ``src`` into it."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    api.upload_large_folder(
        folder_path=str(src),
        repo_id=repo,
        repo_type="dataset",
        path_in_repo=path_in_repo,
    )
    print(f"uploaded {src} to https://huggingface.co/datasets/{repo}")


def main() -> int:
    """Check the licence, show what would go, and upload once confirmed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("what", choices=("yodas-metadata", "commonvoice"))
    parser.add_argument("--src", required=True, help="directory to publish")
    parser.add_argument("--repo", required=True, help="<org>/<name> on the Hub")
    parser.add_argument(
        "--rule",
        help="yodas-metadata: the LID rule this run used, which becomes the "
        "directory it lands in so several rules can share one repository",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--yes", action="store_true", help="actually upload")
    args = parser.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        raise SystemExit(f"{src} is not a directory")

    corpus = "yodas3" if args.what == "yodas-metadata" else "commonvoice"
    check_licence(corpus)

    if args.what == "yodas-metadata":
        shards = sorted(src.glob("*/*.jsonl.gz"))
        languages = sorted({p.parent.name for p in shards})
        total = sum(p.stat().st_size for p in shards)
        print(
            f"{len(shards):,} shards over {len(languages)} languages, "
            f"{total / 1024 ** 3:.1f} GB"
        )
        if not shards:
            raise SystemExit(f"no <lang>/<shard>.jsonl.gz under {src}")
        path_in_repo = args.rule or None
    else:
        path_in_repo = None
        print(f"publishing the tree at {src}")

    if not args.yes:
        print("\ndry run; pass --yes to upload")
        return 0
    upload(src, args.repo, path_in_repo, args.private)
    return 0


if __name__ == "__main__":
    sys.exit(main())
