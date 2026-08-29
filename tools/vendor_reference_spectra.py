"""Pick a few real validation-split spectra and vendor them as a fixed diagnostic panel.

The MS2 half of `msspeculator-cli doctor` compares a model against real spectra. Those have to be
committed, because the doctor runs anywhere the binary runs, with no corpus and no network. This
writes that file.

Validation split, not test: validation already drives early stopping and checkpoint selection, so
looking at it every epoch adds nothing to what the run already sees, while test stays untouched.
Not train either, since a panel the model was fit on cannot show a fit going wrong.

Deterministic: spectra are chosen by sorting on `spectrum_id`, so re-running against the same
corpus rewrites the same file. Run it, eyeball the diff, commit it.

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
    for site in range(rows):
        for column, (ion, fragment_charge) in enumerate(rs.ION_TYPES):
            relative = ms2[site * columns + column] / peak
            if relative < min_intensity:
                continue
            ordinal = site + 1 if ion == "b" else peptide.length - 1 - site
            annotations.append(_annotation(ion, ordinal, fragment_charge))
            mz_values.append(rs.fragment_mz(residue_mass, ion, ordinal, fragment_charge))
            intensities.append(relative)
    if not annotations:
        return None
    return annotations, mz_values, intensities


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

    selected: list[dict] = []
    # One spectrum per dataset, taking the lowest `spectrum_id` with usable peaks. A panel of
    # three spectra from one dataset is three views of one acquisition, and the point of the panel
    # is that a model wrong about a whole class of peptide has somewhere to show it.
    seen_datasets: set[str] = set()
    for uri in chunk_uris:
        if len(selected) == args.count:
            break
        frame = read_prepared_parquet(uri)
        frame = frame.filter(
            (pl.col("split") == "val") & (pl.col("detector") == args.detector)
        ).sort("spectrum_id")
        for row in frame.iter_rows(named=True):
            if row["dataset"] in seen_datasets:
                break
            peaks = _peaks(row["proforma"], list(row["ms2"]), args.min_intensity)
            if peaks is None:
                continue
            annotations, mz_values, intensities = peaks
            seen_datasets.add(row["dataset"])
            selected.append(
                {
                    "dataset": row["dataset"],
                    "proforma": row["proforma"],
                    "charge": row["charge"],
                    "instrument": row["instrument"],
                    "detector": row["detector"],
                    "fragmentation": row["fragmentation"],
                    "energy": f"{row['energy']:.4g}",
                    "irt": f"{row['irt']:.6g}",
                    "annotations": ";".join(annotations),
                    "fragment_mz": ";".join(f"{value:.6f}" for value in mz_values),
                    "relative_intensity": ";".join(f"{value:.6f}" for value in intensities),
                }
            )
            break
        if len(selected) == args.count:
            break

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
    for row in selected:
        print(f"  {row['dataset']}  {row['proforma']}  z={row['charge']}")


if __name__ == "__main__":
    main()
