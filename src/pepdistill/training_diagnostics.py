"""Longitudinal, low-frequency diagnostics for a live training run.

The plotting primitives live in :mod:`pepdistill.diagnostics`; this module owns the model-
specific extraction and Lightning lifecycle.  PCA frames and reference spectra are frozen at
construction/first render so changes across steps describe the student, not a moving diagnostic.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

import fsspec
import lightning as L
import numpy as np
import torch
from pepdistill_rs import mod_element_comp

from .chem import MOD_DELTA, Peptide, fragment_mz_matrix
from .data.encode import FRAG_OFFSET, collate
from .data.precursors import Precursor
from .diagnostics import (
    EmbeddingConnection,
    IRT_STANDARDS,
    LabeledEmbedding,
    PcaBasis,
    RtObservation,
    SpectralAngleSeries,
    SpectrumComparison,
    irt_regression_metrics,
    normalized_spectral_angle,
    plot_irt_scatter,
    plot_labeled_embedding_pca,
    plot_spectral_angle_violins,
    plot_spectrum_butterflies,
)
from .proforma import proforma_sequence


T = TypeVar("T")


@dataclass(frozen=True)
class DiagnosticAcquisition:
    """Acquisition point used for the fixed butterfly reference panel."""

    instrument: str = "Lumos"
    detector: str = "FTMS"
    fragmentation: str = "HCD"
    nce: float = 30.0


@dataclass(frozen=True)
class _SpectrumTarget:
    precursor: Precursor
    intensity: np.ndarray


@dataclass(frozen=True)
class DiagnosticRender:
    """Files and scalar summaries produced by one longitudinal snapshot."""

    paths: dict[str, Path]
    metrics: dict[str, float]


def load_reference_distributions(prefix: str) -> dict[str, dict[str, list[int]]]:
    """Read the published spectral-angle reference lines beside a prepared corpus.

    Returns ``{dataset: {series: counts}}`` for whichever of the two reports exist: the teacher
    yardstick (what distillation alone buys) and the corpus replicate ceiling (what any model can
    reach at best). Both are optional -- a corpus prepared before they were published, or a local
    fixture, simply yields no reference series and the panel falls back to the student alone.
    """
    out: dict[str, dict[str, list[int]]] = {}
    sources = {
        "teacher": (f"{prefix.rstrip('/')}/diagnostics/teacher-yardstick.json", "teacher"),
        "ceiling": (f"{prefix.rstrip('/')}/diagnostics/curation-summary.json", "ceiling"),
    }
    for kind, (uri, series) in sources.items():
        try:
            with fsspec.open(uri, "rb") as handle:
                report = json.load(handle)
        except (FileNotFoundError, OSError, ValueError):
            continue
        if kind == "teacher":
            for dataset, entry in (report.get("per_dataset") or {}).items():
                histogram = entry.get("spectral_angle_histogram")
                if histogram:
                    out.setdefault(dataset, {})[series] = list(histogram["counts"])
        else:
            per_source = (report.get("achievable_ceiling") or {}).get("per_source") or {}
            for dataset, subsets in per_source.items():
                if subsets.get("selected"):
                    out.setdefault(dataset, {})[series] = list(subsets["selected"])
    return out


def _evenly_spaced(values: list[T], count: int) -> list[T]:
    if count < 1:
        raise ValueError("butterflies must be positive")
    if count >= len(values):
        return values
    indices = np.linspace(0, len(values) - 1, num=count, dtype=int)
    return [values[index] for index in indices]


def _amino_acid_embeddings(model) -> list[LabeledEmbedding]:
    families = {
        **{aa: "hydrophobic" for aa in "AILMFWV"},
        **{aa: "polar" for aa in "STNQCYG"},
        **{aa: "positive" for aa in "KRH"},
        **{aa: "negative" for aa in "DE"},
        "P": "special",
    }
    weights = model.token_emb.weight.detach().cpu().numpy()
    return [
        LabeledEmbedding(aa, families[aa], weights[ord(aa) - ord("A")])
        for aa in "ACDEFGHIKLMNPQRSTVWY"
    ]


def _modification_embeddings(model) -> tuple[list[LabeledEmbedding], list[EmbeddingConnection]]:
    points: list[LabeledEmbedding] = []
    connections: list[EmbeddingConnection] = []
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
                (
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
                )
            )
            connections.append(EmbeddingConnection(atom_label, mass_label))
    return points, connections


def _context_trajectories(
    encoder, nce_min: float, nce_max: float
) -> tuple[list[LabeledEmbedding], list[EmbeddingConnection]]:
    """Combined acquisition vectors, rather than misleading isolated factor embeddings."""
    candidates = (
        ("Lumos", "ITMS", "HCD", "Lumos:ITMS:HCD"),
        ("Lumos", "FTMS", "HCD", "Lumos:Orbitrap/FTMS:HCD"),
        ("QExactive", "FTMS", "HCD", "QExactive:Orbitrap/FTMS:HCD"),
        ("Exploris", "FTMS", "HCD", "Exploris:Orbitrap/FTMS:HCD"),
        ("timsTOF", "TOF", "HCD", "timsTOF:TOF:HCD"),
    )
    combinations = [
        item
        for item in candidates
        if item[0] in encoder.instruments
        and item[1] in encoder.detectors
        and item[2] in encoder.fragmentations
    ]
    energies = np.linspace(nce_min, nce_max, num=5, dtype=np.float32)
    points: list[LabeledEmbedding] = []
    connections: list[EmbeddingConnection] = []
    device = next(encoder.parameters()).device
    with torch.inference_mode():
        for instrument, detector, fragmentation, display_name in combinations:
            supported = all(
                float(weight[index].detach().norm()) > 1e-10
                for weight, index in (
                    (encoder.inst_emb.weight, encoder.instrument_id(instrument)),
                    (encoder.det_emb.weight, encoder.detector_id(detector)),
                    (encoder.frag_emb.weight, encoder.fragmentation_id(fragmentation)),
                )
            )
            family = display_name if supported else f"{display_name} [untrained factor]"
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
                    torch.as_tensor(energies, device=device),
                )
                .cpu()
                .numpy()
            )
            labels = [f"{display_name}:NCE{energy:g}" for energy in energies]
            for index, (label, energy, vector) in enumerate(
                zip(labels, energies, vectors, strict=True)
            ):
                annotation = f"{energy:g}" if index in (0, n - 1) else ""
                points.append(LabeledEmbedding(label, family, vector, annotation))
            connections.extend(
                EmbeddingConnection(first, second) for first, second in zip(labels, labels[1:])
            )
    return points, connections


class TrainingDiagnosticRenderer:
    """Render a fixed reference panel repeatedly as live model weights change."""

    def __init__(
        self,
        out: str | Path,
        teacher,
        *,
        acquisition: DiagnosticAcquisition = DiagnosticAcquisition(),
        butterflies: int = 3,
        nce_range: tuple[float, float] = (20.0, 40.0),
        reference_prefix: str | None = None,
    ) -> None:
        if nce_range[0] >= nce_range[1]:
            raise ValueError("diagnostic NCE range must be increasing")
        os.environ.setdefault(
            "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pepdistill-matplotlib")
        )
        self.out = Path(out)
        self.acquisition = acquisition
        self.nce_range = nce_range
        # Loaded once: these describe the corpus, not the run, so re-reading them per snapshot
        # would add S3 latency to every render for values that cannot change mid-training.
        self.reference_distributions = (
            load_reference_distributions(reference_prefix) if reference_prefix else {}
        )
        self.reference_name = getattr(teacher, "name", teacher.__class__.__name__)
        references = [
            Precursor(Peptide(standard.sequence), standard.charge, "diagnostic")
            for standard in IRT_STANDARDS
        ]
        selected = _evenly_spaced(references, butterflies)
        labels = teacher.predict(
            selected, nces=np.full(len(selected), acquisition.nce, dtype=np.float32)
        )
        self.targets = tuple(
            _SpectrumTarget(precursor, np.asarray(label.ms2, dtype=np.float32))
            for precursor, label in zip(selected, labels, strict=True)
            if label is not None
        )
        if not self.targets:
            raise ValueError("teacher produced no spectra for the diagnostic reference panel")
        self._bases: dict[str, PcaBasis] = {}

    def _embedding_plot(
        self,
        key: str,
        points: list[LabeledEmbedding],
        path: Path,
        title: str,
        connections: list[EmbeddingConnection] | None = None,
    ) -> Path | None:
        vectors = np.stack([np.asarray(point.vector) for point in points])
        # The acquisition encoder intentionally starts at exactly zero. Fitting PCA to that
        # rank-zero cloud would freeze arbitrary coordinate axes for the rest of the run and
        # could hide most later motion. Defer this one basis until the context actually learns.
        if key not in self._bases and float(np.var(vectors)) <= 1e-16:
            return None
        target, basis = plot_labeled_embedding_pca(
            points,
            path,
            title=title,
            connections=connections or (),
            basis=self._bases.get(key),
        )
        self._bases.setdefault(key, basis)
        return target

    def spectral_angle_panel(
        self, val_sa_histograms: dict[str, Any], path: str | Path
    ) -> Path | None:
        """Draw the student against the teacher and the achievable ceiling, per dataset.

        Only datasets the student actually validated on are drawn, and only when at least one
        reference series exists for them: a lone student violin is already reported as a scalar,
        so the panel earns its place by showing how much of the gap to the ceiling is left.
        """
        groups: list[tuple[str, list[SpectralAngleSeries]]] = []
        for dataset in sorted(val_sa_histograms):
            references = self.reference_distributions.get(dataset, {})
            if not references:
                continue
            series = [SpectralAngleSeries("student", [int(c) for c in val_sa_histograms[dataset]])]
            series += [
                SpectralAngleSeries(name, references[name])
                for name in ("teacher", "ceiling")
                if name in references
            ]
            groups.append((dataset, series))
        if not groups:
            return None
        return plot_spectral_angle_violins(groups, path)

    def render(
        self,
        model,
        encoder,
        label: str,
        *,
        val_sa_histograms: dict[str, Any] | None = None,
    ) -> DiagnosticRender:
        """Render one snapshot while preserving both modules' live training/eval state."""
        model_was_training = model.training
        encoder_was_training = encoder.training
        model.eval()
        encoder.eval()
        target_dir = self.out / label
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            device = next(model.parameters()).device
            references = [
                Precursor(Peptide(standard.sequence), standard.charge, "diagnostic")
                for standard in IRT_STANDARDS
            ]
            batch = collate(references).to(device)
            acquisition = self.acquisition
            with torch.inference_mode():
                ms_context = encoder(
                    torch.full(
                        (len(references),),
                        encoder.instrument_id(acquisition.instrument),
                        dtype=torch.long,
                        device=device,
                    ),
                    torch.full(
                        (len(references),),
                        encoder.detector_id(acquisition.detector),
                        dtype=torch.long,
                        device=device,
                    ),
                    torch.full(
                        (len(references),),
                        encoder.fragmentation_id(acquisition.fragmentation),
                        dtype=torch.long,
                        device=device,
                    ),
                    torch.full(
                        (len(references),), acquisition.nce, dtype=torch.float32, device=device
                    ),
                )
                output = model(batch, ms_context=ms_context)
                predictions = output["ms2"].cpu().numpy()
                predicted_irt = model.unstandardize_rt(output["rt"]).cpu().numpy()

            paths: dict[str, Path] = {}
            aa_path = self._embedding_plot(
                "amino_acids",
                _amino_acid_embeddings(model),
                target_dir / "amino-acid-embeddings.png",
                "Learned amino-acid token embeddings",
            )
            assert aa_path is not None
            paths["amino_acids"] = aa_path
            mod_points, mod_connections = _modification_embeddings(model)
            mod_path = self._embedding_plot(
                "modifications",
                mod_points,
                target_dir / "modification-encoders.png",
                "Modification encoders: composition vs mass",
                mod_connections,
            )
            assert mod_path is not None
            paths["modifications"] = mod_path
            context_points, context_connections = _context_trajectories(encoder, *self.nce_range)
            context_path = self._embedding_plot(
                "acquisition_contexts",
                context_points,
                target_dir / "acquisition-contexts.png",
                f"Combined acquisition-context trajectories — NCE "
                f"{self.nce_range[0]:g}→{self.nce_range[1]:g}",
                context_connections,
            )
            if context_path is not None:
                paths["acquisition_contexts"] = context_path

            prediction_by_sequence = {
                precursor.peptide.sequence: predictions[index]
                for index, precursor in enumerate(references)
            }
            comparisons = []
            agreements = []
            for target in self.targets:
                precursor = target.precursor
                length = precursor.peptide.length
                student = prediction_by_sequence[precursor.peptide.sequence][
                    FRAG_OFFSET : FRAG_OFFSET + length - 1
                ]
                comparisons.append(
                    SpectrumComparison(
                        proforma_sequence=proforma_sequence(precursor.peptide),
                        charge=precursor.charge,
                        fragment_mz=np.asarray(
                            fragment_mz_matrix(precursor.peptide.sequence, precursor.peptide.mods),
                            dtype=np.float64,
                        ),
                        student_intensity=student,
                        reference_intensity=target.intensity,
                        reference_name=self.reference_name,
                    )
                )
                agreements.append(normalized_spectral_angle(student, target.intensity))
            paths["butterflies"] = plot_spectrum_butterflies(
                comparisons, target_dir / "reference-butterflies.png"
            )

            rt_observations = [
                RtObservation(
                    standard.sequence,
                    standard.irt,
                    float(predicted),
                    "iRT standards",
                )
                for standard, predicted in zip(IRT_STANDARDS, predicted_irt, strict=True)
            ]
            paths["irt"] = plot_irt_scatter(
                rt_observations,
                target_dir / "irt-scatter.png",
                title="Built-in iRT model doctor",
            )
            if val_sa_histograms:
                panel = self.spectral_angle_panel(
                    val_sa_histograms, target_dir / "spectral-angle-violins.png"
                )
                if panel is not None:
                    paths["spectral_angle_violins"] = panel
            rt_metrics = irt_regression_metrics(rt_observations)
            metrics = {
                "teacher_spectral_angle": float(np.mean(agreements)),
                "irt_slope": rt_metrics.slope,
                "irt_intercept": rt_metrics.intercept,
                "irt_r_squared": rt_metrics.r_squared,
                "irt_mae": rt_metrics.mae,
            }
            return DiagnosticRender(paths, metrics)
        finally:
            model.train(model_was_training)
            encoder.train(encoder_was_training)


class TrainingDiagnosticCallback(L.Callback):
    """Render at stage boundaries, epoch boundaries, and an optional wall-clock interval."""

    def __init__(
        self,
        renderer: TrainingDiagnosticRenderer,
        stage: str,
        *,
        every_n_epochs: int = 1,
        interval_minutes: float = 60.0,
        render_initial: bool = True,
        artifact_mirror: Callable[[Path], str] | None = None,
        wandb_logger=None,
    ) -> None:
        super().__init__()
        if every_n_epochs < 0:
            raise ValueError("diagnostic every_n_epochs must be non-negative")
        if interval_minutes < 0:
            raise ValueError("diagnostic interval_minutes must be non-negative")
        self.renderer = renderer
        self.stage = stage
        self.every_n_epochs = every_n_epochs
        self.interval_seconds = interval_minutes * 60.0
        self.render_initial = render_initial
        self.artifact_mirror = artifact_mirror
        self.wandb_logger = wandb_logger
        self._last_render_at = 0.0
        self._last_step: int | None = None

    @staticmethod
    def _live_modules(pl_module):
        model = pl_module.model
        encoder = getattr(pl_module, "encoder", None)
        if encoder is None:
            encoder = getattr(pl_module, "context_encoder", None)
        if encoder is None:
            raise RuntimeError("training diagnostics require an MSContextEncoder")
        return model, encoder

    def _render(self, trainer: L.Trainer, pl_module: L.LightningModule, reason: str) -> None:
        step = int(trainer.global_step)
        if self._last_step == step:
            return
        epoch = int(trainer.current_epoch) + 1
        label = f"{self.stage}-step-{step:08d}-epoch-{epoch:04d}"
        model, encoder = self._live_modules(pl_module)
        # Present only on the real-data regime, and only after a validation check has run.
        result = self.renderer.render(
            model,
            encoder,
            label,
            val_sa_histograms=getattr(pl_module, "val_sa_histograms", None),
        )
        self._last_step = step
        self._last_render_at = time.monotonic()
        if self.artifact_mirror is not None:
            for path in result.paths.values():
                self.artifact_mirror(path)
        if self.wandb_logger is not None:
            import wandb

            payload = {
                f"diagnostics/{self.stage}/{name}": value for name, value in result.metrics.items()
            }
            payload.update(
                {
                    f"diagnostics/{self.stage}/{name}": wandb.Image(str(path))
                    for name, path in result.paths.items()
                }
            )
            payload[f"diagnostics/{self.stage}/epoch"] = epoch
            # Keep image and scalar diagnostics on the same ordered logger as training
            # telemetry. Direct Run.log can advance W&B past an older throttled metric.
            self.wandb_logger.log_metrics(payload, step=step)
        trainer.print(
            f"[diagnostics] {self.stage} {reason} at step {step:,}: "
            f"teacher agreement={result.metrics['teacher_spectral_angle']:.4f}, "
            f"iRT R2={result.metrics['irt_r_squared']:.4f} -> "
            f"{next(iter(result.paths.values())).parent}"
        )

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.render_initial:
            self._render(trainer, pl_module, "initial")

    def on_train_batch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch, batch_idx
    ) -> None:
        if (
            self.interval_seconds
            and time.monotonic() - self._last_render_at >= self.interval_seconds
        ):
            self._render(trainer, pl_module, "interval")

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        epoch = int(trainer.current_epoch) + 1
        if self.every_n_epochs and epoch % self.every_n_epochs == 0:
            self._render(trainer, pl_module, "epoch")

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._render(trainer, pl_module, "final")


__all__ = [
    "DiagnosticAcquisition",
    "DiagnosticRender",
    "TrainingDiagnosticCallback",
    "TrainingDiagnosticRenderer",
]
