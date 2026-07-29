"""Enumerate every UNIMOD accession the PROSPECT catalog actually contains.

    uv run python tools/sweep_prospect_mods.py               # default: the 3 PTM-heavy records
    uv run python tools/sweep_prospect_mods.py --records all  # full sweep, all 5 records

Network + cache: downloads each record's metadata parquet through the normal FileCache, so a
second run is local. Prints one row per accession: count, resolved name, and whether it has a
valid 6-element composition. Any 'NO' row is a mod the student cannot encode.

WARNING: a partial sweep (the default, or any explicit subset of ``--records``) can only ever
under-report the accession set relative to a full sweep. It is fine for a quick local check, but
it must NEVER be used to shrink `tests/test_prospect.py::test_every_prospect_accession_is_encodable`
-- "this run only saw 12 of the 22 frozen accessions" is expected and not evidence the list is
stale. Only a `--records all` run, covering every key in `RECORDS`, is authoritative for that.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter

import pandas as pd
import pepdistill_rs as _rs

from pepdistill.data.prospect import RECORDS, ProspectSource
from pepdistill.data.prospect_catalog import load_catalog

ACCESSION = re.compile(r"\[UNIMOD:(\d+)\]")

# The base `prospect`/`tmt` records are large (~3 GB combined) and their modification vocabulary
# is almost entirely a subset of the PTM-enrichment records below. Default to the cheap subset;
# `--records all` opts into the full (authoritative) sweep.
DEFAULT_RECORDS = ("multi_ptm", "tmt_ptm", "test_ptm")


def _parse_records(arg: str) -> list[str]:
    if arg == "all":
        return list(RECORDS)
    chosen = [r.strip() for r in arg.split(",") if r.strip()]
    unknown = [r for r in chosen if r not in RECORDS]
    if unknown:
        raise SystemExit(f"unknown record(s) {unknown}; known: {sorted(RECORDS)}")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        default=",".join(DEFAULT_RECORDS),
        help=(
            "comma-separated subset of RECORDS keys, or 'all' for every record "
            f"(default: {','.join(DEFAULT_RECORDS)})"
        ),
    )
    args = parser.parse_args()
    records = _parse_records(args.records)

    is_full = set(records) == set(RECORDS)
    print(f"sweeping records: {records}" + ("" if is_full else " (PARTIAL -- see warning below)"))
    if not is_full:
        print(
            "WARNING: partial sweep. This can only under-report accessions vs. a full sweep; "
            "do not use it to shrink the frozen test list. Run with --records all for the "
            "authoritative sweep."
        )

    counts: Counter[int] = Counter()
    catalog = load_catalog()
    for record in records:
        source = ProspectSource(record)
        files = catalog["records"][record]["files"]
        for name in files:
            if "meta" not in name or not name.endswith(".parquet"):
                continue
            path = source.resolve_file(name)
            df = pd.read_parquet(path, columns=["modified_sequence"])
            for seq in df["modified_sequence"].unique():
                counts.update(int(a) for a in ACCESSION.findall(str(seq)))

    print(f"{'acc':>8}  {'count':>9}  {'name':<28}  encodable")
    for acc, n in counts.most_common():
        name = _rs.unimod_name(acc)
        if name is None:
            print(f"{acc:>8}  {n:>9}  {'<UNRESOLVED>':<28}  NO")
            continue
        try:
            _rs.mod_element_comp(name)
            ok = "yes"
        except ValueError as e:
            ok = f"NO ({e})"
        print(f"{acc:>8}  {n:>9}  {name:<28}  {ok}")


if __name__ == "__main__":
    main()
