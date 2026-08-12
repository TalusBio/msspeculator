"""Generate learned-representation and model-output diagnostics from a checkpoint."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from pepdistill.chem import MOD_DELTA, Peptide, fragment_mz_matrix
from pepdistill.data.encode import FRAG_OFFSET, collate
from pepdistill.data.precursors import Precursor
from pepdistill.diagnostics import (
    EmbeddingConnection,
    IRT_STANDARDS,
    LabeledEmbedding,
    RtObservation,
    SpectrumComparison,
    plot_irt_scatter,
    plot_labeled_embedding_pca,
    plot_spectrum_butterflies,
)
from pepdistill.models.context import MSContextEncoder
from pepdistill.models.registry import load_checkpoint, load_context
from pepdistill.proforma import proforma_sequence
from pepdistill.teacher import get_teacher
from pepdistill.util import resolve_device
from pepdistill_rs import mod_element_comp


def _butterfly_panel(references: list[Precursor], count: int) -> list[Precursor]:
    """Choose evenly spaced members of the fixed iRT panel."""
    if count >= len(references):
        return references
    indices = np.linspace(0, len(references) - 1, num=count, dtype=int)
    return [references[index] for index in indices]


def _amino_acid_embeddings(model) -> list[LabeledEmbedding]:
    families = {
        **{aa: "hydrophobic" for aa in "AILMFWV"},
        **{aa: "polar" for aa in "STNQCYG"},
        **{aa: "positive" for aa in "KRH"},
        **{aa: "negative" for aa in "DE"},
        **{aa: "special" for aa in "P"},
    }
    weights = model.token_emb.weight.detach().cpu().numpy()
    return [
        LabeledEmbedding(aa, families[aa], weights[ord(aa) - ord("A")])
        for aa in "ACDEFGHIKLMNPQRSTVWY"
    ]


def _modification_embeddings(model) -> tuple[list[LabeledEmbedding], list[EmbeddingConnection]]:
    points = []
    connections = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for name, mass in sorted(MOD_DELTA.items()):
            composition = torch.tensor(
                mod_element_comp(name), dtype=torch.float32, device=device
            ).unsqueeze(0)
            mass_tensor = torch.tensor([mass], dtype=torch.float32, device=device)
            atom_label = f"{name}:atoms"
            mass_label = f"{name}:mass"
            points.extend(
                [
                    LabeledEmbedding(
                        atom_label,
                        "composition encoder",
                        model.comp_enc(composition).squeeze(0).cpu().numpy(),
                    ),
                    LabeledEmbedding(
                        mass_label,
                        "mass encoder",
                        model.mass_enc(mass_tensor).squeeze(0).cpu().numpy(),
                    ),
                ]
            )
            connections.append(EmbeddingConnection(atom_label, mass_label))
    return points, connections


def _context_trajectories(encoder) -> tuple[list[LabeledEmbedding], list[EmbeddingConnection]]:
    """Actual combined acquisition vectors while NCE moves from 20 to 40."""
    candidates = (
        ("Lumos", "ITMS", "HCD", "Lumos:ITMS:HCD"),
        ("Lumos", "FTMS", "HCD", "Lumos:Orbitrap/FTMS:HCD"),
        ("QExactive", "FTMS", "HCD", "QExactive:Orbitrap/FTMS:HCD"),
        ("Exploris", "FTMS", "HCD", "Exploris:Orbitrap/FTMS:HCD"),
        ("timsTOF", "TOF", "HCD", "timsTOF:TOF:HCD"),
    )
    combinations = [
        combination
        for combination in candidates
        if combination[0] in encoder.instruments
        and combination[1] in encoder.detectors
        and combination[2] in encoder.fragmentations
    ]
    points: list[LabeledEmbedding] = []
    connections: list[EmbeddingConnection] = []
    device = next(encoder.parameters()).device
    energies = list(range(20, 41, 5))
    with torch.inference_mode():
        for instrument, detector, fragmentation, display_name in combinations:
            n = len(energies)
            vectors = (
                encoder(
                    torch.full(
                        (n,), encoder.instrument_id(instrument), dtype=torch.long, device=device
                    ),
                    torch.full(
                        (n,), encoder.detector_id(detector), dtype=torch.long, device=device
                    ),
                    torch.full(
                        (n,),
                        encoder.fragmentation_id(fragmentation),
                        dtype=torch.long,
                        device=device,
                    ),
                    torch.tensor(energies, dtype=torch.float32, device=device),
                )
                .cpu()
                .numpy()
            )
            labels = [f"{display_name}:NCE{energy}" for energy in energies]
            for index, (label, energy, vector) in enumerate(
                zip(labels, energies, vectors, strict=True)
            ):
                annotation = str(energy) if index in (0, n - 1) else ""
                points.append(LabeledEmbedding(label, display_name, vector, annotation))
            connections.extend(
                EmbeddingConnection(first, second) for first, second in zip(labels, labels[1:])
            )
    return points, connections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/diagnostics-prototype"))
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
    model = load_checkpoint(args.model, map_location=str(device)).to(device).eval()
    context = load_context(args.model, map_location=str(device))
    encoder = context.encoder if context is not None else None
    if encoder is None:
        encoder = MSContextEncoder(context_dim=model.cfg.context_dim)
    encoder = encoder.to(device).eval()
    references = [
        Precursor(Peptide(standard.sequence), standard.charge, "diagnostic")
        for standard in IRT_STANDARDS
    ]
    batch = collate(references).to(device)
    n = len(references)
    with torch.inference_mode():
        ms_context = encoder(
            torch.full((n,), encoder.instrument_id("Lumos"), dtype=torch.long, device=device),
            torch.full((n,), encoder.detector_id("FTMS"), dtype=torch.long, device=device),
            torch.full((n,), encoder.fragmentation_id("HCD"), dtype=torch.long, device=device),
            torch.full((n,), 30.0, dtype=torch.float32, device=device),
        )
        model_output = model(batch, ms_context=ms_context)
        prediction = model_output["ms2"].cpu().numpy()
    aa_path, _ = plot_labeled_embedding_pca(
        _amino_acid_embeddings(model),
        args.out / "amino-acid-embeddings.png",
        title="Learned amino-acid token embeddings",
    )
    mod_points, mod_connections = _modification_embeddings(model)
    mod_path, _ = plot_labeled_embedding_pca(
        mod_points,
        args.out / "modification-encoders.png",
        title="Modification encoders: composition vs mass",
        connections=mod_connections,
    )
    context_points, context_connections = _context_trajectories(encoder)
    context_path, _ = plot_labeled_embedding_pca(
        context_points,
        args.out / "acquisition-contexts.png",
        title="Combined acquisition-context trajectories — NCE 20→40",
        connections=context_connections,
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
                proforma_sequence=proforma_sequence(precursor.peptide),
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
    print(f"amino acids -> {aa_path}")
    print(f"modifications -> {mod_path}")
    print(f"contexts -> {context_path}")
    print(f"butterflies -> {butterfly_path}")
    predicted_irt = model.unstandardize_rt(model_output["rt"]).cpu().numpy()
    irt_path = plot_irt_scatter(
        [
            RtObservation(standard.sequence, standard.irt, float(predicted), "iRT standards")
            for standard, predicted in zip(IRT_STANDARDS, predicted_irt, strict=True)
        ],
        args.out / "irt-scatter.png",
        title="Built-in iRT model doctor",
    )
    print(f"iRT -> {irt_path}")


if __name__ == "__main__":
    main()
