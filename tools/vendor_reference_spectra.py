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

    `ms2` is the flat prepared grid; its shape and fragment ordering come from the extension, so
    this cannot disagree with what the model was trained against.
    """
    peptide = rs.Peptide.from_string(proforma)
    residue_mass = peptide.residue_masses()
    rows, columns = rs.ms2_target_shape(peptide.length)
    peak = max(ms2) if ms2 else 0.0
    if peak <= 0:
        return None

    annotations: list[str] = []
    mz_values: list[float] = []
    intensities: list[float] = []
    for site in range(rows):
        for column, (ion, fragment_charge) in enumerate(rs.ION_TYPES):
            value = ms2[site * columns + column]
            relative = value / peak
            if relative < min_intensity:
                continue
            index = site - rs.FRAG_OFFSET
            if not 0 <= index < peptide.length - 1:
                continue
            ordinal = index + 1 if ion == "b" else peptide.length - 1 - index
            annotations.append(_annotation(ion, ordinal, fragment_charge))
            mz_values.append(rs.fragment_mz(residue_mass, ion, ordinal, fragment_charge))
            intensities.append(relative)
    if not annotations:
        return None
    return annotations, mz_values, intensities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-prefix", required=True, help="prepared corpus prefix")
    parser.add_argument("--out", type=Path, required=True, help="TSV to write")
    parser.add_argument("--count", type=int, default=3, help="spectra to vendor")
    parser.add_argument(
        "--min-intensity",
        type=float,
        default=0.01,
        help="drop peaks below this fraction of the base peak",
    )
    args = parser.parse_args()

    manifest = PreparedManifest.load(args.prepared_prefix)
    selected: list[dict] = []
    # Chunks in manifest order, and rows sorted within each: the first `count` usable validation
    # spectra of the first chunk that has any. Nothing here depends on how the corpus was sharded.
    for chunk in manifest.chunks:
        frame = read_prepared_parquet(chunk.uri)
        frame = frame.filter(pl.col("split") == "val").sort("spectrum_id")
        for row in frame.iter_rows(named=True):
            peaks = _peaks(row["proforma"], list(row["ms2"]), args.min_intensity)
            if peaks is None:
                continue
            annotations, mz_values, intensities = peaks
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
            if len(selected) == args.count:
                break
        if len(selected) == args.count:
            break

    if len(selected) < args.count:
        raise SystemExit(
            f"only {len(selected)} usable validation spectra found in {args.prepared_prefix}"
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
