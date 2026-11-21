#!/usr/bin/env python3
"""Report how far this recipe's corpora have got, measured against the espnet2 dump.

Every number comes off disk: the dump's ``text`` files give the espnet2 row
counts, the built caches give ours. Nothing here is typed in by hand, so the
table can be regenerated at any point and will disagree with a stale page.

  dataset_status.py --json results.json      # the paper's results.json shape
  dataset_status.py                          # the same table on stdout

``--json`` rewrites only the ``status`` table and the ``sources`` block of an
existing file, so hand-written tables beside it survive.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# corpus -> how it appears on both sides. ``ours`` and ``dump`` override the
# split names where they are not simply "dev" and "train"; ``prefix``
# restricts the dump rows to one of several corpora sharing a directory.
CORPORA = {
    "spgispeech": {"dir": "SPGISpeech", "ours": {"dev": "val"}},
    "aishell_1": {"dir": "AISHELL-1"},
    "tedlium3_filtered": {"dir": "TEDLIUM3_filtered"},
    "gigaspeech": {"dir": "GigaSpeech", "ours": {"dev": "DEV", "train": "XL"}},
    "gigast": {"dir": "GigaST", "ours": {"dev": None, "train": "XL"}},
    "voxpopuli": {"dir": "VoxPopuli"},
    "reazonspeech": {"dir": "ReazonSpeech"},
    "commonvoice": {"dir": "commonvoice"},
    "must_c_v1.2": {"dir": "MuST-C", "prefix": "MuST-C_v1.2_"},
    "must_c_v2": {"dir": "MuST-C", "prefix": "MuST-C_v2_"},
    "must_c_v3": {"dir": "MuST-C", "prefix": "MuST-C_v3_"},
    "mls": {"dir": "MLS"},
    "covost2": {"dir": "CoVoST2"},
    "fleurs": {"dir": "FLEURS"},
    "wenetspeech": {"dir": "WenetSpeech"},
    "laborotv": {
        "dir": "LaboroTVSpeech",
        "dump": {"dev": "laborotv_dev_owsm", "train": "laborotv_train_owsm"},
    },
    "aidatatang": {"dir": "aidatatang"},
    "magicdata": {"dir": "magicdata"},
    "vctk": {"dir": "vctk"},
    "librispeech": {"dir": "LibriSpeech"},
    "ami": {"dir": "ami"},
    "babel": {"dir": "babel"},
    "fisher_callhome": {"dir": "fisher_callhome"},
    "ksponspeech": {"dir": "ksponspeech"},
    "swbd": {"dir": "swbd"},
    "voxforge": {"dir": "voxforge"},
    "openslr": {"dir": "openslr"},
    "ru_open_stt": {"dir": "ru_open_stt"},
    "jtubespeech": {
        "dir": "JtubeSpeech",
        "dump": {
            "dev": "jtubespeech_test_normal_jun21_owsm",
            "train": "jtubespeech_train_owsm",
        },
    },
    "wsj": {"dir": "wsj"},
}

SPLITS = ("dev", "train")


def split_name(spec: dict, side: str, split: str) -> str | None:
    """Return ``side``'s name for ``split`` ("ours" or "dump"), None if it has none."""
    return spec.get(side, {}).get(split, split)


def dump_rows(dump_root: Path, spec: dict, split: str) -> int | None:
    """Count the dump's rows for one split, or None when it has no such split."""
    name = split_name(spec, "dump", split)
    if name is None:
        return None
    text = dump_root / "datasets" / spec["dir"] / name / "text"
    if not text.is_file():
        return None
    prefix = spec.get("prefix")
    with text.open(encoding="utf-8", errors="replace") as handle:
        if prefix is None:
            return sum(1 for _ in handle)
        return sum(1 for line in handle if line.startswith(prefix))


def cache_rows(cache_root: Path, corpus: str, split: str) -> int | None:
    """Return a built split's row count, or None when it has not been built."""
    info = cache_root / corpus / "hf_audio_index" / split / "dataset_info.json"
    if not info.is_file():
        return None
    # save_to_disk keeps the count under the generator's own split name, which
    # is always "train" however the split is called here.
    splits = json.loads(info.read_text()).get("splits") or {}
    return sum(part["num_examples"] for part in splits.values()) or None


def collect(dump_root: Path, cache_root: Path) -> list[list]:
    """Return one row per corpus: dump and cache counts for dev and train."""
    rows = []
    for corpus, spec in CORPORA.items():
        record = [corpus]
        for split in SPLITS:
            theirs = dump_rows(dump_root, spec, split)
            name = split_name(spec, "ours", split)
            ours = cache_rows(cache_root, corpus, name) if name else None
            record += [theirs, ours, None if None in (theirs, ours) else ours - theirs]
        rows.append(record)
    return rows


def main() -> int:
    """Print the table, or fold it into a paper exploration's ``results.json``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", default=os.environ.get("OWSM_DUMP"))
    parser.add_argument("--cache", default=os.environ.get("OWSM_CACHE"))
    parser.add_argument("--json", help="results.json to update in place")
    args = parser.parse_args()
    if not args.dump or not args.cache:
        parser.error("--dump/--cache or $OWSM_DUMP/$OWSM_CACHE are required")

    rows = collect(Path(args.dump), Path(args.cache))
    columns = [
        "Corpus",
        "espnet2 dev",
        "ours dev",
        "Δ dev",
        "espnet2 train",
        "ours train",
        "Δ train",
    ]
    table = {
        "caption": "Rows per corpus in the espnet2 dump and in this recipe's cache. "
        "A blank cell means the split does not exist there yet.",
        "label": "tbl-exp-001-status",
        "columns": columns,
        "format": [None, ",d", ",d", ",d", ",d", ",d", ",d"],
        "rows": rows,
    }
    if not args.json:
        print(" | ".join(columns))
        for row in rows:
            print(" | ".join("" if v is None else str(v) for v in row))
        return 0

    path = Path(args.json)
    results = json.loads(path.read_text()) if path.is_file() else {}
    results.setdefault("tables", {})["status"] = table
    results.setdefault("metrics", {})["corpora_built"] = sum(
        1 for row in rows if row[2] is not None or row[5] is not None
    )
    results["metrics"]["corpora_total"] = len(rows)
    results.setdefault("sources", {})["status"] = {
        "dump": str(args.dump),
        "cache": str(args.cache),
        "generated_by": "egs3/owsm_v4/esp2_s2t/local/dataset_status.py",
    }
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} corpora to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
