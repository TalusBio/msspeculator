"""Pick a few real validation-split spectra and vendor them as a fixed diagnostic panel.

The MS2 half of `msspeculator-cli doctor` compares a model against real spectra. Those have to be
committed, because the doctor runs anywhere the binary runs, with no corpus and no network. This
writes that file.

Validation split, not test: validation already drives early stopping and checkpoint selection, so
looking at it every epoch adds nothing to what the run already sees, while test stays untouched.
Not train either, since a panel the model was fit on cannot show a fit going wrong.

Spectra are chosen on quality, not position. A reference the model can be wrong about in many
places is worth more than one it can miss in three, so the ranking is backbone coverage first
(the fraction of fragmentation sites with an observed ion), then Andromeda score. Both are
recorded in the output, and ties break on `spectrum_id`, so the same corpus rewrites the same
file. Run it, read the ranking it prints, commit the diff.

    uv run --extra etl python tools/vendor_reference_spectra.py \
        --prepared-prefix s3://bucket/pepdistill-prepared/v2 \
        --out data/reference_peptides/diagnostic_spectra.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import msspeculator_rs as rs
import polars as pl

from msspeculator.data.prepared import PreparedManifest
from msspeculator.data.prepared_schema import read_prepared_parquet

#: Header of the vendored panel. Peaks are struct-of-arrays within a row, so a sequence is written
#: once rather than once per peak, and the file stays legible in a diff.
COLUMNS = (
    "dataset",
    "proforma",
    "charge",
    "instrument",
    "detector",
    "fragmentation",
    "energy",
    "irt",
    # Why this spectrum and not another one. Recorded so a later run that picks something else is
    # a visible change of evidence rather than an unexplained diff.
    "backbone_coverage",
    "andromeda_score",
    "annotations",
    "fragment_mz",
    "relative_intensity",
)


def _annotation(ion: str, ordinal: int, charge: int) -> str:
    """mzSpecLib peak annotation, matching what the mzSpecLib sink already writes."""
    return f"{ion}{ordinal}^{charge}" if charge > 1 else f"{ion}{ordinal}"


def _peaks(proforma: str, ms2: list[float], min_intensity: float):
    """Kept peaks for one spectrum, as parallel annotation / m/z / intensity lists.

    `ms2` is the flat prepared target grid, exactly `(length - 1, len(ION_TYPES))` with row `i`
    being the site after residue `i + 1`. No `FRAG_OFFSET` here: that offset belongs to the
    model's padded output pool, and applying it to this grid shifts every ordinal by one, which
    reads as a b1 ion nobody's instrument produced.
    """
    peptide = rs.Peptide.from_string(proforma)
    residue_mass = peptide.residue_masses()
    rows, columns = rs.ms2_target_shape(peptide.length)
    if len(ms2) != rows * columns:
        raise SystemExit(f"{proforma}: expected {rows * columns} MS2 values, got {len(ms2)}")
    peak = max(ms2) if ms2 else 0.0
    if peak <= 0:
        return None

    annotations: list[str] = []
    mz_values: list[float] = []
    intensities: list[float] = []
    covered_sites = set()
    for site in range(rows):
        for column, (ion, fragment_charge) in enumerate(rs.ION_TYPES):
            relative = ms2[site * columns + column] / peak
            if relative < min_intensity:
                continue
            ordinal = site + 1 if ion == "b" else peptide.length - 1 - site
            annotations.append(_annotation(ion, ordinal, fragment_charge))
            mz_values.append(rs.fragment_mz(residue_mass, ion, ordinal, fragment_charge))
            intensities.append(relative)
            covered_sites.add(site)
    if not annotations:
        return None
    # Coverage counts sites, not peaks: four charge/ion variants of one backbone cleavage say less
    # about a model than four different cleavages do.
    return {
        "annotations": annotations,
        "fragment_mz": mz_values,
        "relative_intensity": intensities,
        "backbone_coverage": len(covered_sites) / rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prepared-prefix", help="prepared corpus prefix, read via its manifest")
    source.add_argument(
        "--chunks",
        action="append",
        metavar="GLOB",
        help=(
            "prepared Parquet chunks to read instead of a manifest, repeatable. A local "
            "read-through cache holds shards but no manifest, which is what this is for."
        ),
    )
    parser.add_argument("--out", type=Path, required=True, help="TSV to write")
    parser.add_argument("--count", type=int, default=3, help="spectra to vendor")
    parser.add_argument(
        "--detector",
        default="FTMS",
        help=(
            "keep only this detector. FTMS by default: an ion trap discards fragments below "
            "roughly a third of the precursor m/z, so a reference spectrum from one can put its "
            "base peak at the detection edge and make a failing check ambiguous."
        ),
    )
    parser.add_argument(
        "--min-intensity",
        type=float,
        default=0.01,
        help="drop peaks below this fraction of the base peak",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=200,
        help=(
            "per dataset, how many top-scoring spectra to rank on coverage. Andromeda score is a "
            "column and costs nothing to sort on; coverage needs the peaks unpacked, so score "
            "narrows the field first."
        ),
    )
    args = parser.parse_args()

    if args.prepared_prefix:
        chunk_uris = [chunk.uri for chunk in PreparedManifest.load(args.prepared_prefix).chunks]
    else:
        # Sorted, so the same globs pick the same spectra however the filesystem lists them.
        chunk_uris = sorted(
            str(path) for pattern in args.chunks for path in Path().glob(pattern.lstrip("/"))
        )
        if not chunk_uris:
            raise SystemExit(f"no chunks matched {args.chunks}")

    # Score narrows, coverage decides. Each chunk contributes only its best-scoring rows, so the
    # whole corpus never has to be held at once.
    shortlist: dict[str, list[pl.DataFrame]] = {}
    for uri in chunk_uris:
        frame = read_prepared_parquet(uri).filter(
            (pl.col("split") == "val") & (pl.col("detector") == args.detector)
        )
        if frame.is_empty():
            continue
        for key, group in frame.group_by(["dataset"]):
            shortlist.setdefault(str(key[0]), []).append(
                group.top_k(args.candidates, by="andromeda_score")
            )

    ranked: list[dict] = []
    for _dataset, frames in sorted(shortlist.items()):
        best: dict | None = None
        candidates = pl.concat(frames).top_k(args.candidates, by="andromeda_score")
        # `spectrum_id` last, so a tie on both quality measures still resolves the same way twice.
        for row in candidates.sort(
            ["andromeda_score", "spectrum_id"], descending=[True, False]
        ).iter_rows(named=True):
            peaks = _peaks(row["proforma"], list(row["ms2"]), args.min_intensity)
            if peaks is None:
                continue
            key = (peaks["backbone_coverage"], row["andromeda_score"])
            if best is None or key > best["key"]:
                best = {"key": key, "row": row, "peaks": peaks}
        if best is None:
            continue
        row, peaks = best["row"], best["peaks"]
        ranked.append(
            {
                "dataset": row["dataset"],
                "proforma": row["proforma"],
                "charge": row["charge"],
                "instrument": row["instrument"],
                "detector": row["detector"],
                "fragmentation": row["fragmentation"],
                "energy": f"{row['energy']:.4g}",
                "irt": f"{row['irt']:.6g}",
                "backbone_coverage": f"{peaks['backbone_coverage']:.4f}",
                "andromeda_score": f"{row['andromeda_score']:.4g}",
                "annotations": ";".join(peaks["annotations"]),
                "fragment_mz": ";".join(f"{value:.6f}" for value in peaks["fragment_mz"]),
                "relative_intensity": ";".join(
                    f"{value:.6f}" for value in peaks["relative_intensity"]
                ),
            }
        )

    # One per dataset, and the best datasets first: a panel of three spectra from one dataset is
    # three views of one acquisition, and the point is that a model wrong about a whole class of
    # peptide has somewhere to show it.
    ranked.sort(key=lambda item: (-float(item["backbone_coverage"]), item["dataset"]))
    selected = ranked[: args.count]
    if len(selected) < args.count:
        raise SystemExit(
            f"only {len(selected)} usable validation spectra found in {len(chunk_uris)} chunk(s)"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as stream:
        stream.write("\t".join(COLUMNS) + "\n")
        for row in selected:
            stream.write("\t".join(str(row[column]) for column in COLUMNS) + "\n")
    print(f"wrote {len(selected)} spectra -> {args.out}")
    print(f"{'dataset':<26} {'peptidoform':<24} {'z':>2} {'coverage':>9} {'andromeda':>10}")
    for row in ranked:
        mark = "*" if row in selected else " "
        print(
            f"{mark}{row['dataset']:<25} {row['proforma']:<24} {row['charge']:>2} "
            f"{float(row['backbone_coverage']):>9.2f} {float(row['andromeda_score']):>10.1f}"
        )


if __name__ == "__main__":
    main()
