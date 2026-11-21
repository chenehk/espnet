#!/usr/bin/env python3
"""Hours and languages per corpus, read off the espnet2 dump's utterance ids.

Every id ends ``_<start_ms>_<end_ms>_<lang>_<task>``, so the span is in the id
and no audio has to be opened. Rows that share a span -- one ASR row and a
translation row per target language -- are counted once for hours, because the
audio is the same; ``row_hours`` keeps the inflated figure for comparison.

  dump_stats.py --dump $OWSM_DUMP --out stats.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

ID = re.compile(r"^(?P<span>.*_(?P<start>\d{9})_(?P<end>\d{9}))_(?P<rest>.+)$")
LANG = re.compile(r"^<([a-z]{3})>")


def scan(split_dir: Path) -> dict:
    """Return hours, deduplicated hours and languages for one split."""
    spans: set[str] = set()
    row_seconds = 0.0
    span_seconds = 0.0
    langs: dict[str, int] = {}
    tasks: set[str] = set()
    rows = 0
    with (split_dir / "text").open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            utt_id, _, text = line.rstrip("\n").partition(" ")
            found = ID.match(utt_id)
            if found is None:
                continue
            rows += 1
            seconds = (int(found["end"]) - int(found["start"])) / 1000.0
            row_seconds += seconds
            if found["span"] not in spans:
                spans.add(found["span"])
                span_seconds += seconds
            language = LANG.match(text)
            if language:
                langs[language.group(1)] = langs.get(language.group(1), 0) + 1
            if "<st_" in text[:24]:
                tasks.add("st")
            elif "<asr>" in text[:24]:
                tasks.add("asr")
    return {
        "rows": rows,
        "row_hours": round(row_seconds / 3600, 1),
        "hours": round(span_seconds / 3600, 1),
        "languages": langs,
        "tasks": sorted(tasks),
    }


def main() -> int:
    """Scan every corpus of the dump and write one record each."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", default=os.environ.get("OWSM_DUMP"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.dump) / "datasets"
    out: dict[str, dict] = {}
    for corpus_dir in sorted(root.iterdir()):
        if not corpus_dir.is_dir():
            continue
        merged = {
            "rows": 0,
            "row_hours": 0.0,
            "hours": 0.0,
            "languages": {},
            "tasks": set(),
        }
        for split_dir in sorted(corpus_dir.iterdir()):
            if not (split_dir / "text").is_file():
                continue
            part = scan(split_dir)
            merged["rows"] += part["rows"]
            merged["row_hours"] += part["row_hours"]
            merged["hours"] += part["hours"]
            merged["tasks"] |= set(part["tasks"])
            for code, count in part["languages"].items():
                merged["languages"][code] = merged["languages"].get(code, 0) + count
        merged["tasks"] = sorted(merged["tasks"])
        merged["row_hours"] = round(merged["row_hours"], 1)
        merged["hours"] = round(merged["hours"], 1)
        merged["n_languages"] = len(merged["languages"])
        out[corpus_dir.name] = merged
        print(
            f"{corpus_dir.name:20} {merged['hours']:>10,.1f} h  "
            f"{merged['n_languages']:>3} languages  {merged['tasks']}",
            flush=True,
        )

    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
