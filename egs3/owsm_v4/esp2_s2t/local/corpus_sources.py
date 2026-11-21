#!/usr/bin/env python3
"""Report, per corpus, how its data is obtained and whether its licence allows use.

  corpus_sources.py --json results.json      # the paper's results.json shape
  corpus_sources.py                          # the same table on stdout
  corpus_sources.py --licence-free --json results.json \
      --dump-stats stats.json --yodas-results exp-002/results.json

The "fetches" column is derived, not declared: a builder counts as fetching its
own data only if it overrides the base class' ``prepare_source`` (which just
raises) or calls ``datasets.load_dataset``. It therefore stays true as the
builders change.

The licence column is a registry, because a licence is not visible from the
code. Each entry names where it was read and when, so a stale one is obvious.

``--licence-free`` drops every corpus whose verdict is not ``usable`` and adds
the three things a reader of that list wants: the commit that added the builder,
and the corpus' hours and language count. Hours come from the espnet2 dump,
which is why they are the v3 figures and not what a fresh build would produce;
YODAS has no dump row and is read from the cleaning exploration's results.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from datetime import date
from pathlib import Path

SUB_DATASETS = Path(__file__).resolve().parents[1] / "dataset" / "sub_datasets"

# A verdict is about one goal: a released model anyone may use, including
# commercially. NC, ND, SA and copyleft all fail it, as does anything whose
# terms forbid redistribution.
OK, NC, ND, SA, GATED, LDC, MIXED, UNKNOWN = (
    "usable",
    "non-commercial",
    "no-derivatives",
    "share-alike",
    "gated",
    "LDC",
    "mixed",
    "unknown",
)

# corpus -> (licence, verdict, where it was read, when)
LICENCES = {
    "aishell_1": ("Apache-2.0", OK, "openslr.org/33", "2026-11-17"),
    "aidatatang": (
        "CC BY-NC-ND 4.0",
        NC,
        "secondary sources only; openslr.org/62 states no licence",
        "2026-11-17",
    ),
    "magicdata": ("CC BY-NC-ND 4.0", NC, "openslr.org/68", "2026-11-17"),
    "must_c": ("CC BY-NC-ND 4.0", NC, "mt.fbk.eu", "2026-11-17"),
    "must_c_v2": ("CC BY-NC-ND 4.0", NC, "mt.fbk.eu", "2026-11-17"),
    "must_c_v3": ("CC BY-NC-ND 4.0", NC, "mt.fbk.eu", "2026-11-17"),
    "spgispeech": (
        "Kensho user agreement",
        GATED,
        "huggingface.co/datasets/kensho/spgispeech; non-commercial, "
        "no redistribution",
        "2026-11-17",
    ),
    "reazonspeech": (
        "CDLA-Sharing-1.0",
        SA,
        "huggingface.co/datasets/reazon-research/reazonspeech; also limited to "
        "Japanese Copyright Act Article 30-4",
        "2026-11-17",
    ),
    "covost2": (
        "CC0 (translations)",
        OK,
        "github.com/facebookresearch/covost; audio comes from Common Voice",
        "2026-11-17",
    ),
    "voxforge": (
        "GPL (v2 or later, v3 for later submissions)",
        SA,
        "voxforge.org; copyleft over the audio itself",
        "2026-11-17",
    ),
    "gigast": (
        "CC BY-NC 4.0",
        NC,
        "st-benchmark.github.io/resources/GigaST",
        "2026-11-17",
    ),
    "gigaspeech": (
        "non-commercial terms of access",
        NC,
        "TERMS_OF_ACCESS in the corpus directory",
        "2026-11-17",
    ),
    "wenetspeech": (
        "non-commercial terms of access",
        NC,
        "TERMS_OF_ACCESS in the corpus directory",
        "2026-11-17",
    ),
    "voxpopuli": (
        "CC0 (data)",
        OK,
        "the repo's own README licence table: data CC0, code and models "
        "CC BY-NC 4.0. The LICENSE file in the checkout is the CODE licence, "
        "which this registry previously mistook for the data's",
        "2026-11-20",
    ),
    "tedlium3_filtered": ("CC BY-NC-ND 3.0", ND, "openslr.org/51", "2026-11-16"),
    "swbd": ("LDC97S62", LDC, "catalog.ldc.upenn.edu", "2026-11-16"),
    "fisher_callhome": (
        "LDC2010S01 and others",
        LDC,
        "catalog.ldc.upenn.edu",
        "2026-11-16",
    ),
    "babel": ("IARPA Babel, LDC", LDC, "catalog.ldc.upenn.edu", "2026-11-16"),
    "ru_open_stt": (
        "CC BY-NC 4.0",
        NC,
        "github.com/snakers4/open_stt",
        "2026-11-17",
    ),
    "jtubespeech": (
        "video-id list only; audio is YouTube's",
        UNKNOWN,
        "github.com/sarulab-speech/jtubespeech states no licence for the audio",
        "2026-11-17",
    ),
    "laborotv": (
        "custom",
        UNKNOWN,
        "arxiv.org/abs/2412.17847 records it as Custom; no primary page found",
        "2026-11-17",
    ),
    "openslr": (
        "CC BY-SA 4.0 for 23 of the 29 sets; the other 6 state none",
        SA,
        "LICENSE files in each SLR directory; SLR32/37/41/42/43/44 have none",
        "2026-11-17",
    ),
    "librispeech": ("CC BY 4.0", OK, "openslr.org/12", "2026-11-16"),
    "mls": ("CC BY 4.0", OK, "openslr.org/94", "2026-11-16"),
    "vctk": ("ODC-By 1.0", OK, "COPYING in the corpus directory", "2026-11-16"),
    "commonvoice": ("CC0", OK, "commonvoice.mozilla.org", "2026-11-17"),
    "fleurs": ("CC BY 4.0", OK, "huggingface.co/datasets/google/fleurs", "2026-11-17"),
    "ami": (
        "CC BY 4.0",
        OK,
        "groups.inf.ed.ac.uk/ami/download",
        "2026-11-17",
    ),
    "ksponspeech": (
        "AIHub registration, KOGL Type 4",
        GATED,
        "aihub.or.kr; OWSM's own license_free list calls this free, which the "
        "AIHub terms do not support",
        "2026-11-17",
    ),
    "yodas3": ("CC BY 3.0", OK, "huggingface.co/datasets/espnet/yodas2", "2026-11-17"),
    "yodas_cleaned": (
        "CC BY 3.0",
        OK,
        "huggingface.co/datasets/espnet/yodas2; the cleaning publishes metadata "
        "only, and the audio stays under the source dataset's licence",
        "2026-11-21",
    ),
}

# The page the licence was read on. Separate from the registry above so that
# adding one cannot silently reword a verdict; collect() refuses a corpus that
# has a licence but no URL.
URLS = {
    "aishell_1": "https://openslr.org/33",
    "aidatatang": "https://openslr.org/62",
    "magicdata": "https://openslr.org/68",
    "must_c": "https://mt.fbk.eu/must-c/",
    "must_c_v2": "https://mt.fbk.eu/must-c/",
    "must_c_v3": "https://mt.fbk.eu/must-c/",
    "spgispeech": "https://huggingface.co/datasets/kensho/spgispeech",
    "reazonspeech": "https://huggingface.co/datasets/reazon-research/reazonspeech",
    "covost2": "https://github.com/facebookresearch/covost",
    "voxforge": "https://www.voxforge.org/",
    "gigast": "https://st-benchmark.github.io/resources/GigaST",
    "gigaspeech": "https://huggingface.co/datasets/speechcolab/gigaspeech",
    "wenetspeech": "https://github.com/wenet-e2e/WenetSpeech",
    "voxpopuli": "https://github.com/facebookresearch/voxpopuli",
    "tedlium3_filtered": "https://openslr.org/51",
    "swbd": "https://catalog.ldc.upenn.edu/LDC97S62",
    "fisher_callhome": "https://catalog.ldc.upenn.edu/LDC2010S01",
    "babel": "https://catalog.ldc.upenn.edu/",
    "ru_open_stt": "https://github.com/snakers4/open_stt",
    "jtubespeech": "https://github.com/sarulab-speech/jtubespeech",
    "laborotv": "https://github.com/laboroai/LaboroTVSpeech",
    "openslr": "https://openslr.org/",
    "librispeech": "https://openslr.org/12",
    "mls": "https://openslr.org/94",
    "vctk": "https://datashare.ed.ac.uk/handle/10283/3443",
    "commonvoice": "https://commonvoice.mozilla.org/en/datasets",
    "fleurs": "https://huggingface.co/datasets/google/fleurs",
    "ami": "https://groups.inf.ed.ac.uk/ami/download/",
    "ksponspeech": "https://aihub.or.kr",
    "yodas3": "https://huggingface.co/datasets/espnet/yodas2",
    "yodas_cleaned": "https://huggingface.co/datasets/espnet/yodas2",
}

# A clean licence is not the same as an open download. These two are CC0, but
# the audio only comes through a Common Voice account that accepts the terms.
AGREEMENT = {
    "commonvoice": "Common Voice account; accept the dataset terms to download",
    "covost2": "translations are open, but the audio is Common Voice's",
}

# corpus -> the key dump_stats.py writes for it. Absent means the dump has no
# row for that corpus.
DUMP_KEY = {
    "aishell_1": "AISHELL-1",
    "ami": "ami",
    "commonvoice": "commonvoice",
    "covost2": "CoVoST2",
    "fleurs": "FLEURS",
    "librispeech": "LibriSpeech",
    "mls": "MLS",
    "vctk": "vctk",
    "voxpopuli": "VoxPopuli",
}

# OpenSLR is 29 sets under separate terms, so it expands into one row per
# licence group rather than one row for the corpus.
OPENSLR_SETS = {
    "CC BY-SA 4.0": [35, 36, 52, 53, 54, 61, 63, 64, 65, 66, 69, 70, 71, 72,
                     73, 74, 75, 76, 77, 78, 79, 83, 86],
    "no licence file": [32, 37, 41, 42, 43, 44],
}

# The one step still outside the repository: MLS ships no long-form archive, so
# the long audio the builder reads is rebuilt by a script in the gitignored
# submit/, and a new cluster has neither it nor the rebuilt audio.
SCRIPTED = {
    "mls": "submit/build_mls_audio_long.sh (reconstructs long audio)",
}


def fetches_own_data(corpus: str) -> bool:
    """Check whether this builder can obtain the corpus rather than find it."""
    source = (SUB_DATASETS / corpus / "builder.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_source":
            return True
        if isinstance(node, ast.Attribute) and node.attr == "load_dataset":
            return True
    return False


def collect() -> list[list]:
    """Return one row per builder on disk."""
    rows = []
    for path in sorted(SUB_DATASETS.iterdir()):
        if not (path / "builder.py").is_file():
            continue
        corpus = path.name
        licence, verdict, where, when = LICENCES.get(
            corpus, ("—", UNKNOWN, "not looked up", "—")
        )
        if corpus in LICENCES and corpus not in URLS:
            raise SystemExit(f"{corpus} has a licence but no URL in URLS")
        rows.append(
            [
                corpus,
                "yes" if fetches_own_data(corpus) else "no",
                SCRIPTED.get(corpus, "—"),
                licence,
                verdict,
                where,
                when,
            ]
        )
    return rows


def commit_of(corpus: str) -> str:
    """Return the short hash of the commit that last touched this builder."""
    result = subprocess.run(
        ["git", "log", "-1", "--format=%h", "--", str(SUB_DATASETS / corpus)],
        capture_output=True,
        text=True,
        cwd=SUB_DATASETS,
    )
    return result.stdout.strip() or "uncommitted"


def dump_stats(path: str | None) -> dict:
    """Return dump_stats.py's output, or an empty mapping when not given."""
    return json.loads(Path(path).read_text()) if path else {}


def yodas_stats(path: str | None) -> tuple[float | None, int | None]:
    """Return hours and languages for the cleaned YODAS we would publish.

    The dump has no YODAS row, so the figures come from the cleaning
    exploration's own results -- the production rule, which is the one the
    published metadata is built from.
    """
    if not path:
        return None, None
    coverage = json.loads(Path(path).read_text())["tables"]["coverage"]
    columns = coverage["columns"]
    hours = columns.index("Hours kept")
    langs = columns.index("Languages >=1 h")
    for row in coverage["rows"]:
        if row[0] == "production":
            return row[hours], row[langs]
    raise SystemExit(f"{path} has no 'production' row in its coverage table")


def licence_free(stats: dict, yodas: tuple) -> tuple[list[list], list[list]]:
    """Return the open corpora and, separately, the ones behind an agreement."""
    open_rows, gated_rows = [], []
    for path in sorted(SUB_DATASETS.iterdir()):
        if not (path / "builder.py").is_file():
            continue
        corpus = path.name
        licence, verdict, _, _ = LICENCES.get(corpus, ("—", UNKNOWN, "", ""))
        if verdict != OK:
            continue
        if corpus.startswith("yodas"):
            hours, languages = yodas
        else:
            entry = stats.get(DUMP_KEY.get(corpus, ""), {})
            hours = entry.get("hours")
            languages = entry.get("n_languages")
        row = [
            corpus,
            f"[{licence}]({URLS[corpus]})",
            commit_of(corpus),
            hours,
            languages,
            AGREEMENT.get(corpus, "open download"),
        ]
        (gated_rows if corpus in AGREEMENT else open_rows).append(row)
    return open_rows, gated_rows


def main() -> int:
    """Print the table, or fold it into a paper exploration's results.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="results.json to update in place")
    parser.add_argument(
        "--csv", help="write a CSV with openslr expanded into its licence groups"
    )
    parser.add_argument(
        "--licence-free",
        action="store_true",
        help="only the usable corpora, with commit, hours and languages",
    )
    parser.add_argument("--dump-stats", help="dump_stats.py output, for hours")
    parser.add_argument(
        "--yodas-results", help="exp-002 results.json, for the YODAS row"
    )
    args = parser.parse_args()

    if args.licence_free:
        return write_licence_free(args)

    rows = collect()
    columns = [
        "Corpus",
        "Builder fetches it",
        "Script (uncommitted)",
        "Licence",
        "Verdict",
        "Read from",
        "Checked",
    ]
    table = {
        "caption": "How each corpus is obtained, and whether its licence allows "
        "a model anyone may use. A verdict other than *usable* excludes the "
        "corpus from a licence-clean build.",
        "label": "tbl-exp-003-sources",
        "columns": columns,
        "rows": rows,
    }
    if args.csv:
        import csv as csv_module

        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv_module.writer(handle)
            writer.writerow(columns)
            for row in rows:
                if row[0] != "openslr":
                    writer.writerow(row)
                    continue
                for licence, sets in OPENSLR_SETS.items():
                    names = ", ".join(f"SLR{n}" for n in sets)
                    verdict = SA if licence.startswith("CC BY-SA") else UNKNOWN
                    writer.writerow(
                        [f"openslr ({names})", row[1], row[2], licence, verdict]
                        + row[5:]
                    )
        print(f"wrote {args.csv}")
        return 0

    if not args.json:
        for row in rows:
            print(" | ".join(str(value) for value in row))
        return 0

    path = Path(args.json)
    results = json.loads(path.read_text()) if path.is_file() else {}
    results.setdefault("tables", {})["sources"] = table
    verdicts = [row[4] for row in rows]
    results.setdefault("metrics", {}).update(
        corpora=len(rows),
        usable=verdicts.count(OK),
        fetch_themselves=sum(1 for row in rows if row[1] == "yes"),
        usable_that_fetch_themselves=sum(
            1 for row in rows if row[4] == OK and row[1] == "yes"
        ),
    )
    results.setdefault("sources", {})["sources"] = {
        "generated_by": "egs3/owsm_v4/esp2_s2t/local/corpus_sources.py",
        "generated_on": str(date.today()),
    }
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} corpora to {path}")
    return 0


def write_licence_free(args) -> int:
    """Emit the usable-corpora table, open sets first and gated ones last."""
    open_rows, gated_rows = licence_free(
        dump_stats(args.dump_stats), yodas_stats(args.yodas_results)
    )
    rows = open_rows + gated_rows
    table = {
        "caption": "The corpora a licence-clean OWSM may ship: every one allows "
        "a model anyone may use, commercially included. Hours and languages are "
        "the espnet2 v3 dump's, except YODAS, which has no dump row and is the "
        "production rule's output from @sec-exp-002-yodas2-recleaning. The last "
        "two rows are open by licence but not by access: their audio comes "
        "through a Common Voice account.",
        "label": "tbl-exp-003-licence-free",
        "columns": [
            "Corpus",
            "Licence",
            "Commit",
            "Hours",
            "Languages",
            "How it is obtained",
        ],
        "format": [None, None, None, ",.1f", ",d", None],
        "rows": rows,
    }
    if not args.json:
        for row in rows:
            print(" | ".join("" if value is None else str(value) for value in row))
        return 0

    path = Path(args.json)
    results = json.loads(path.read_text()) if path.is_file() else {}
    results.setdefault("tables", {})["licence_free"] = table
    results.setdefault("metrics", {}).update(
        licence_free_corpora=len(rows),
        licence_free_open_access=len(open_rows),
        licence_free_hours=sum(row[3] for row in rows if row[3] is not None),
    )
    results.setdefault("sources", {})["licence_free"] = {
        "generated_by": "egs3/owsm_v4/esp2_s2t/local/corpus_sources.py --licence-free",
        "hours_from": args.dump_stats or "not supplied",
        "yodas_from": args.yodas_results or "not supplied",
        "generated_on": str(date.today()),
    }
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} licence-free corpora to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
