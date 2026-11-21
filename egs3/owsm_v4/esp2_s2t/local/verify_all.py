#!/usr/bin/env python3
"""Check one corpus' built cache against the espnet2 dump and record the result.

  verify_all.py --corpus openslr --out <dir>

Writes ``<dir>/<corpus>.json``: row counts both ways and per-stream mismatch
counts for every split. The acceptance criterion is the dump being a subset of
ours — ``s2t.sh`` stage 4 drops utterances below ``min_wav_duration`` after
``local/prepare_*.py`` has run, so rows the dump lacks are expected and rows it
has that we lack are not.

One corpus per process: the biggest hold tens of millions of rows on each side.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from egs3.owsm_v4.esp2_s2t.local.check_against_dump import (  # noqa: E402
    CORPORA,
    STREAMS,
    load_dump,
    our_rows,
)


def verify(corpus: str, splits: list[str], dump_root: Path, recipe_dir, cache) -> dict:
    """Compare the cache against the dump for ``splits`` taken together."""
    dump = {stream: {} for stream in STREAMS}
    for split in splits:
        part = load_dump(dump_root, corpus, split, None)
        for stream in STREAMS:
            dump[stream].update(part[stream])

    our_ids: set[str] = set()
    extra = 0
    mismatches = {stream: 0 for stream in STREAMS}
    examples: list[str] = []
    for split in splits:
        for utt_id, streams in our_rows(corpus, split, None, None, recipe_dir, cache):
            our_ids.add(utt_id)
            if utt_id not in dump["text"]:
                extra += 1
                continue
            for stream in STREAMS:
                if streams[stream] != dump[stream].get(utt_id):
                    mismatches[stream] += 1
                    if len(examples) < 5 and stream == "text":
                        examples.append(utt_id)

    matched = len(dump["text"].keys() & our_ids)
    return {
        "splits": splits,
        "ours": len(our_ids),
        "dump": len(dump["text"]),
        "in_dump_not_ours": len(dump["text"]) - matched,
        "in_ours_not_dump": extra,
        "stream_mismatches": mismatches,
        "examples": examples,
        "pass": (len(dump["text"]) - matched) == 0 and not any(mismatches.values()),
    }


def main() -> int:
    """Verify one corpus and write its record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    parser.add_argument("--dump", default=os.environ.get("OWSM_DUMP"))
    parser.add_argument("--cache", default=os.environ.get("OWSM_CACHE", "data/hf"))
    parser.add_argument("--recipe-dir", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    spec = CORPORA[args.corpus]
    cache = {"cache_dir": args.cache}
    started = time.perf_counter()
    record: dict = {"corpus": args.corpus}
    try:
        if spec.get("compare_union"):
            # The dev/train partition is ours, not upstream's, so only the
            # union of the two splits is comparable.
            record["union"] = verify(
                args.corpus,
                sorted(spec["dump_splits"]),
                Path(args.dump),
                args.recipe_dir,
                cache,
            )
        else:
            for split in sorted(spec["dump_splits"]):
                record[split] = verify(
                    args.corpus, [split], Path(args.dump), args.recipe_dir, cache
                )
    except Exception as error:  # a missing cache or dump split is a result too
        record["error"] = f"{type(error).__name__}: {error}"
    record["seconds"] = round(time.perf_counter() - started, 1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.corpus}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
