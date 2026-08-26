"""Score a trained student against a published spectral library, faceted.

Read-only. MS2 only: a library that has never been trained on has no chromatography row, so its
retention time is not predictable and is not scored here.

The acquisition context is whatever the checkpoint already knows. For a timsTOF library that is
usually very little; the instrument and detector rows are still at their zero init unless
something has trained them; so the score is a floor, and the panel says which rows were
actually informed rather than implying the model knew the setup.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from pepdistill.chem import Peptide
from pepdistill.data.encode import FRAG_OFFSET, collate
from pepdistill.data.precursors import Precursor
from pepdistill.diagnostics import sa_histogram
from pepdistill.models.registry import load_checkpoint, load_context

ION_COLUMNS = 4


def read_library(path: str, aliases: list[dict], context: str) -> dict:
    import pepdistill_rs as rs

    return rs.read_speclib(
        path,
        {
            "context": context,
            "instrument": "timsTOF",
            "detector": "TOF",
            "fragmentation": "HCD",
            "aliases": aliases,
            "retention_column": None,
        },
    )


def library_grid(offsets, sites, ions, values, index: int, frag_rows: int) -> np.ndarray:
    """Scatter one precursor's CSR fragments into the model's dense (row, ion) grid."""
    grid = np.zeros((frag_rows, ION_COLUMNS), dtype=np.float32)
    start, end = int(offsets[index]), int(offsets[index + 1])
    for site, ion, value in zip(sites[start:end], ions[start:end], values[start:end]):
        row = FRAG_OFFSET + int(site)
        if row < frag_rows:
            grid[row, int(ion)] = value
    return grid


def spectral_angle(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return float("nan")
    cos = float(np.clip(a.ravel() @ b.ravel() / denominator, -1.0, 1.0))
    return 1.0 - 2.0 * float(np.arccos(cos)) / np.pi


def sum_norm_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Depth audit: distance between the two as distributions.

    Chosen over KL for the same job because it is bounded and carries no smoothing constant that
    would silently set the price of a zero.
    """
    x, y = a.ravel(), b.ravel()
    sx, sy = x.sum(), y.sum()
    if sx <= 0 or sy <= 0:
        return float("nan")
    return float(np.linalg.norm(x / sx - y / sy))


def modification_facet(peptide: Peptide) -> str:
    from pepdistill.chem import unimod_title

    titles = sorted(
        {
            unimod_title(int(str(spec).removeprefix("UNIMOD:")))
            for _, spec in peptide.mods
            if str(spec).startswith("UNIMOD:")
        }
    )
    if not titles:
        return "unmodified"
    return "+".join(titles) if len(titles) <= 2 else f"{len(titles)} modifications"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--context", default="library")
    parser.add_argument(
        "--alias",
        action="append",
        default=[],
        metavar="ACCESSION[:MASS]",
        help="declare a modification present in the library, optionally with its spelled mass",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="score only the first N precursors")
    parser.add_argument("--out", type=Path, help="write the facet report as JSON")
    args = parser.parse_args()

    aliases = []
    for raw in args.alias:
        accession, _, mass = raw.partition(":")
        aliases.append(
            {"accession": int(accession), "observed_mass": float(mass) if mass else None}
        )

    library = read_library(args.library, aliases, args.context)
    stats = library["stats"]
    print(
        f"[library] {stats['rows']:,} rows -> {stats['precursors']:,} precursors; "
        f"fragments dropped {stats['fragments_dropped']:,}; unmapped {stats['unmapped_masses']}"
    )
    if stats["unmapped_masses"]:
        raise SystemExit(
            "refusing to score a library with unresolved mass shifts; declare them with --alias"
        )

    model = load_checkpoint(args.checkpoint)
    bundle = load_context(args.checkpoint)
    encoder = bundle.encoder
    if encoder is None:
        raise SystemExit(f"{args.checkpoint} carries no acquisition encoder")
    model.eval()
    encoder.eval()
    trained = {
        name: float(weight[index].detach().norm()) > 1e-10
        for name, weight, index in (
            ("timsTOF", encoder.inst_emb.weight, encoder.instrument_id("timsTOF")),
            ("TOF", encoder.det_emb.weight, encoder.detector_id("TOF")),
            ("HCD", encoder.frag_emb.weight, encoder.fragmentation_id("HCD")),
        )
    }
    print(f"[context] acquisition rows trained: {trained}; collision energy: absent (masked)")

    proforma = library["proforma"]
    charges = library["charge"]
    offsets, sites, ions, values = (
        library["frag_offset"],
        library["frag_site"],
        library["frag_ion"],
        library["frag_value"],
    )
    total = len(proforma) if args.limit <= 0 else min(args.limit, len(proforma))

    # Batch equal-length peptides together: the collator pads to the longest in the batch, and
    # the library spans 7..40 residues, so mixing lengths would pad most of the grid.
    by_length: dict[int, list[int]] = defaultdict(list)
    peptides = []
    for index in range(total):
        peptide = Peptide.from_string(proforma[index])
        peptides.append(peptide)
        by_length[peptide.length].append(index)

    facets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    scores: list[float] = []
    depth: list[float] = []
    for length, indices in sorted(by_length.items()):
        for start in range(0, len(indices), args.batch_size):
            chunk = indices[start : start + args.batch_size]
            batch = collate([Precursor(peptides[i], int(charges[i]), "library") for i in chunk])
            with torch.inference_mode():
                ms_context = encoder(
                    torch.full((len(chunk),), encoder.instrument_id("timsTOF"), dtype=torch.long),
                    torch.full((len(chunk),), encoder.detector_id("TOF"), dtype=torch.long),
                    torch.full((len(chunk),), encoder.fragmentation_id("HCD"), dtype=torch.long),
                    torch.full((len(chunk),), float("nan")),
                )
                predicted = model.forward_context(batch, ms_context=ms_context)["ms2"].numpy()
            frag_rows = predicted.shape[1]
            for position, index in enumerate(chunk):
                observed = library_grid(offsets, sites, ions, values, index, frag_rows)
                if observed.sum() <= 0:
                    continue
                angle = spectral_angle(predicted[position], observed)
                distance = sum_norm_l2(predicted[position], observed)
                scores.append(angle)
                depth.append(distance)
                peptide = peptides[index]
                mass = peptide.mono_mass()
                for facet, level in (
                    ("modification", modification_facet(peptide)),
                    ("length", f"{(length // 5) * 5}-{(length // 5) * 5 + 4}"),
                    ("charge", f"z{int(charges[index])}"),
                    ("mass", f"{int(mass // 500) * 500}-{int(mass // 500) * 500 + 499}"),
                ):
                    facets[facet][level].append(angle)

    scored = np.asarray(scores)
    print(
        f"\n[overall] {scored.size:,} precursors scored; spectral angle "
        f"median {np.median(scored):.4f}, mean {scored.mean():.4f}; "
        f"sum-norm L2 median {np.median(depth):.4f}"
    )
    report: dict[str, dict] = {
        "library": args.library,
        "checkpoint": args.checkpoint,
        "acquisition_rows_trained": trained,
        "overall": {
            "n": int(scored.size),
            "spectral_angle_median": float(np.median(scored)),
            "spectral_angle_mean": float(scored.mean()),
            "sum_norm_l2_median": float(np.median(depth)),
            "histogram": sa_histogram(scored)["counts"],
        },
        "facets": {},
    }
    for facet, levels in facets.items():
        print(f"\n{facet}")
        report["facets"][facet] = {}
        for level, angles in sorted(levels.items(), key=lambda item: -len(item[1])):
            values_array = np.asarray(angles)
            print(
                f"  {level:<28} n={values_array.size:>7,}  median {np.median(values_array):.4f}"
                f"  mean {values_array.mean():.4f}"
            )
            report["facets"][facet][level] = {
                "n": int(values_array.size),
                "median": float(np.median(values_array)),
                "mean": float(values_array.mean()),
                "histogram": sa_histogram(values_array)["counts"],
            }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
