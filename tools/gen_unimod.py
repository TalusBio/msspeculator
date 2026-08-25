"""Regenerate the vendored UNIMOD tables. Network; run manually, commit the output.

    uv run python tools/gen_unimod.py

Pass `--expect-digest HEX` to fail unless upstream still hashes to what the vendored tables
were extracted from; the exact command is recorded in each file's own header.

Emits two TSVs under rust/core/data/, each opening with a '#'-prefixed provenance/license
header (see `_header` below) that travels with the file itself -- not just this docstring --
since the TSVs, not this script, are the redistributed/compiled artifact:
  elements.tsv  symbol \t mono_mass          (from <umod:elem>)
  unimod.tsv    accession \t title \t composition \t mono_mass   (from <umod:mod>)

The mono_mass column of unimod.tsv is a TEST FIXTURE, not a source of truth: pepdistill
computes mass from the composition and asserts agreement across every row, so a bad element
mass or a parser bug fails loudly instead of hiding in one unused modification.

Source: https://www.unimod.org/xml/unimod.xml
Terms:  The header of unimod.xml itself (fetched 2026-07-28) reads, verbatim:

            Copyright (C) 2002-2006 Unimod; this information may be copied, distributed and/or
            modified under certain conditions, but it comes WITHOUT ANY WARRANTY; see the
            accompanying Design Science License for more details

        The referenced Design Science License (https://www.unimod.org/dsl.txt, mirrored at
        https://www.gnu.org/licenses/dsl.html) grants, verbatim:

            "Permission is granted to distribute, publish or otherwise present verbatim copies
            of the entire Source Data of the Work, in any medium, provided that full copyright
            notice and disclaimer of warranty, where applicable, is conspicuously published on
            all copies, and a copy of this License is distributed along with the Work."

        and, for derivative works:

            "Permission is granted to modify or sample from a copy of the Work, producing a
            derivative work, and to distribute the derivative work" provided the derivative is
            published under the same License, is given a new name distinct from "Unimod", and
            carries a notice of what was changed.

        This generator produces a derivative work (two TSVs extracted and reformatted from the
        source XML, not verbatim copies of it) under those terms: this file documents the
        source, the copyright notice, and the license; the emitted tables are named
        `elements.tsv` / `unimod.tsv`, distinct from "Unimod"; and this docstring records the
        transformation applied (element-list and modification-list extraction into TSV).
        Redistribution and modification are permitted under the Design Science License —
        vendoring is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

URL = "https://www.unimod.org/xml/unimod.xml"
NS = {"umod": "http://www.unimod.org/xmlns/schema/unimod_2"}
OUT = Path(__file__).resolve().parent.parent / "rust" / "core" / "data"

# Field-per-line rather than prose: a reader scanning the top of a data file wants to find one
# fact, and wrapped sentences hide them. Emitted as a '#'-prefixed header on both TSVs so the
# provenance travels with the redistributed artifact, not just this generator's docstring. The
# Rust loader (rust/core/src/unimod.rs) skips '#'-prefixed lines when parsing.
#
# `source-hash` is what makes "extracted from" checkable, and it is deliberately the only
# identity recorded: a fetch date says when somebody ran this, while the digest says exactly what
# they ran it against, and writing a date would make every regeneration differ from the last.
_LICENSE = (
    "# license      Design Science License (https://www.unimod.org/dsl.txt) -- NO WARRANTY.\n"
    "#              Copyright (C) 2002-2006 Unimod. This file is NOT unimod.xml and is not\n"
    "#              affiliated with or endorsed by Unimod; it is a derivative work.\n"
)


def _header(title: str, source_hash: str, extracted: str, columns: str) -> str:
    return (
        f"# {title}\n"
        "#\n"
        f"# source       {URL}\n"
        f"# source-hash  blake2b-256:{source_hash}\n"
        f"# verify       uv run python tools/gen_unimod.py --expect-digest {source_hash}\n"
        "# generator    tools/gen_unimod.py -- regenerate rather than hand-edit\n"
        f"# extracted    {extracted}\n"
        f"# columns      {columns}\n"
        "#\n" + _LICENSE
    )


def _composition(parent: ET.Element) -> str:
    """Render an element list as 'H(20) C(8) 13C(4) N 15N O(2)' (count omitted when 1)."""
    terms = []
    for el in parent.findall("umod:element", NS):
        sym = el.attrib["symbol"]
        n = int(el.attrib["number"])
        terms.append(sym if n == 1 else f"{sym}({n})")
    return " ".join(terms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect-digest",
        metavar="HEX",
        help="fail unless upstream still hashes to this, so the claim in the vendored headers "
        "is a command anyone can re-run rather than a sentence anyone must trust",
    )
    args = parser.parse_args()

    # Read to bytes, then parse from bytes: hashing needs the source to exist as bytes, and
    # parsing a live socket cannot tell a complete document from a truncated one -- a mid-transfer
    # reset that still returns 200 is exactly how a short FASTA got hashed as valid provenance.
    with urllib.request.urlopen(URL, timeout=180) as fh:
        raw = fh.read()
    source_hash = hashlib.blake2b(raw, digest_size=32).hexdigest()
    print(f"fetched {len(raw):,} bytes; blake2b-256 {source_hash}")
    if args.expect_digest and args.expect_digest != source_hash:
        raise SystemExit(
            f"upstream has changed: expected {args.expect_digest}, got {source_hash}. The "
            "vendored tables were extracted from a different unimod.xml; regenerate deliberately "
            "and review the row diff rather than assuming this is a formatting change."
        )
    root = ET.fromstring(raw)

    OUT.mkdir(parents=True, exist_ok=True)

    elements = root.find("umod:elements", NS)
    if elements is None:
        raise RuntimeError("unimod.xml has no <umod:elements> block; format changed")
    rows = []
    for el in elements.findall("umod:elem", NS):
        rows.append((el.attrib["title"], float(el.attrib["mono_mass"])))
    rows.sort()
    (OUT / "elements.tsv").write_text(
        _header(
            "UNIMOD nuclide masses -- derivative work, NOT unimod.xml.",
            source_hash,
            f"{len(rows)} nuclides from <umod:elements>: each <umod:elem>'s `title` and "
            "`mono_mass`, sorted by symbol",
            "symbol, mono_mass",
        )
        + "".join(f"{sym}\t{mass!r}\n" for sym, mass in rows)
    )
    print(f"elements.tsv: {len(rows)} nuclides")

    mods = root.find("umod:modifications", NS)
    if mods is None:
        raise RuntimeError("unimod.xml has no <umod:modifications> block; format changed")
    mrows = []
    for mod in mods.findall("umod:mod", NS):
        delta = mod.find("umod:delta", NS)
        if delta is None:
            continue
        comp = _composition(delta)
        if not comp:
            continue
        mrows.append(
            (
                int(mod.attrib["record_id"]),
                mod.attrib["title"],
                comp,
                float(delta.attrib["mono_mass"]),
            )
        )
    mrows.sort()
    (OUT / "unimod.tsv").write_text(
        _header(
            "UNIMOD modification deltas -- derivative work, NOT unimod.xml.",
            source_hash,
            f"{len(mrows)} modifications from <umod:modifications>: each <umod:mod> that has a "
            "<umod:delta>, keeping `record_id`, `title`, the delta's element list rendered as "
            "'Sym' / 'Sym(count)' (e.g. '13C(4) N 15N'), and the delta's `mono_mass`, sorted by "
            "accession",
            "accession, title, composition, mono_mass -- mono_mass is a TEST FIXTURE: mass is "
            "computed from the composition and asserted against it on every row",
        )
        + "".join(f"{a}\t{t}\t{c}\t{m!r}\n" for a, t, c, m in mrows)
    )
    print(f"unimod.tsv: {len(mrows)} modifications")


if __name__ == "__main__":
    main()
