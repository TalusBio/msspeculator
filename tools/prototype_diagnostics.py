"""Generate PCA and butterfly diagnostic prototypes from a trained checkpoint."""

from __future__ import annotations

import argparse
import os
import tempfile
from itertools import islice
from pathlib import Path

import numpy as np
import torch

from pepdistill.chem import Peptide, fragment_mz_matrix
from pepdistill.data.config import DigestConfig
from pepdistill.data.digest import resolve_fasta
from pepdistill.data.encode import FRAG_OFFSET, collate
from pepdistill.data.precursors import Precursor
from pepdistill.data.sources import enumerate_tryptic_stream
from pepdistill.diagnostics import (
    PcaBasis,
    SpectrumComparison,
    plot_embedding_pca,
    plot_spectrum_butterflies,
)
from pepdistill.models.context import MSContextEncoder
from pepdistill.models.registry import load_checkpoint, load_context
from pepdistill.teacher import get_teacher
from pepdistill.util import resolve_device


def _reference_precursor(sequence: str) -> Precursor:
    mods = tuple((i, "Carbamidomethyl@C") for i, aa in enumerate(sequence) if aa == "C")
    return Precursor(Peptide(sequence, mods), charge=2, split="diagnostic")


def _butterfly_panel(references: list[Precursor], count: int) -> list[Precursor]:
    """Choose readable short/medium examples, preferring a mixture of modified states."""
    ranked = sorted(
        (reference for reference in references if 8 <= reference.peptide.length <= 18),
        key=lambda reference: (
            not bool(reference.peptide.mods),
            abs(reference.peptide.length - 13),
            reference.peptide.sequence,
        ),
    )
    selected: list[Precursor] = []
    for want_modified in (True, False):
        candidate = next(
            (item for item in ranked if bool(item.peptide.mods) == want_modified), None
        )
        if candidate is not None and candidate not in selected:
            selected.append(candidate)
    selected.extend(item for item in ranked if item not in selected)
    return selected[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/diagnostics-prototype"))
    parser.add_argument("--fasta", default="uniprot:UP000000625")
    parser.add_argument("--panel-size", type=int, default=192)
    parser.add_argument("--butterflies", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--teacher", default="alphapeptdeep")
    args = parser.parse_args()

    # Headless/sandboxed machines often have no writable ~/.matplotlib. Keep font discovery
    # out of the user's home and make repeated prototype renders reuse one temporary cache.
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pepdistill-matplotlib")
    )

    device = resolve_device(args.device)
    fasta = resolve_fasta(args.fasta, log=print)
    digest = DigestConfig(missed_cleavages=2, min_length=7, max_length=30)
    sequences = list(
        islice(enumerate_tryptic_stream(fasta, digest, loop=False), args.panel_size)
    )
    references = [_reference_precursor(sequence) for sequence in sequences]
    if len(references) < 2:
        raise SystemExit("reference FASTA produced fewer than two peptides")

    model = load_checkpoint(args.model, map_location=str(device)).to(device).eval()
    context = load_context(args.model, map_location=str(device))
    encoder = context.encoder if context is not None else None
    if encoder is None:
        encoder = MSContextEncoder(context_dim=model.cfg.context_dim)
    encoder = encoder.to(device).eval()
    batch = collate(references).to(device)
    n = len(references)
    with torch.inference_mode():
        pooled = model.pooled_embeddings(batch).cpu().numpy()
        ms_context = encoder(
            torch.full((n,), encoder.instrument_id("Lumos"), dtype=torch.long, device=device),
            torch.full((n,), encoder.detector_id("FTMS"), dtype=torch.long, device=device),
            torch.full((n,), encoder.fragmentation_id("HCD"), dtype=torch.long, device=device),
            torch.full((n,), 30.0, dtype=torch.float32, device=device),
        )
        prediction = model(batch, ms_context=ms_context)["ms2"].cpu().numpy()

    basis = PcaBasis.fit(pooled)
    pca_path = plot_embedding_pca(
        basis.transform(pooled),
        [precursor.peptide.length for precursor in references],
        [bool(precursor.peptide.mods) for precursor in references],
        args.out / "embedding-pca.png",
        title="Fixed E. coli reference panel — pooled student embeddings",
        explained_variance_ratio=basis.explained_variance_ratio,
    )

    selected = _butterfly_panel(references, args.butterflies)
    teacher_kwargs = {} if args.teacher == "fake" else {"device": "cpu", "instrument": "Lumos"}
    teacher = get_teacher(args.teacher, **teacher_kwargs)
    targets = teacher.predict(selected, nces=np.full(len(selected), 30.0))
    butterflies = []
    prediction_by_sequence = {
        precursor.peptide.modified_sequence(): prediction[index]
        for index, precursor in enumerate(references)
    }
    for precursor, target in zip(selected, targets, strict=True):
        if target is None:
            continue
        length = precursor.peptide.length
        student = prediction_by_sequence[precursor.peptide.modified_sequence()][
            FRAG_OFFSET : FRAG_OFFSET + length - 1
        ]
        mz = np.asarray(
            fragment_mz_matrix(precursor.peptide.sequence, precursor.peptide.mods),
            dtype=np.float64,
        )
        butterflies.append(
            SpectrumComparison(
                modified_sequence=precursor.peptide.modified_sequence(),
                charge=precursor.charge,
                fragment_mz=mz,
                student_intensity=student,
                reference_intensity=target.ms2,
                reference_name=("synthetic target" if args.teacher == "fake" else "AlphaPeptDeep"),
            )
        )
    butterfly_path = plot_spectrum_butterflies(
        butterflies,
        args.out / "reference-butterflies.png",
    )
    print(f"PCA -> {pca_path}")
    print(f"butterflies -> {butterfly_path}")


if __name__ == "__main__":
    main()
