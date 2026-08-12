"""Low-frequency visual diagnostics for representation and spectral-shape drift."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import fsspec
import numpy as np


@dataclass(frozen=True)
class PcaBasis:
    """A fixed PCA frame fitted once so coordinates remain comparable across epochs."""

    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray

    @classmethod
    def fit(cls, embeddings: np.ndarray, n_components: int = 2) -> "PcaBasis":
        values = np.asarray(embeddings, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2:
            raise ValueError("PCA requires a 2D array with at least two observations")
        if not 1 <= n_components <= min(values.shape):
            raise ValueError("n_components exceeds the embedding matrix rank")
        mean = values.mean(axis=0)
        _, singular, vt = np.linalg.svd(values - mean, full_matrices=False)
        variance = singular**2
        ratio = (
            variance[:n_components] / variance.sum() if variance.sum() else np.zeros(n_components)
        )
        return cls(mean, vt[:n_components], ratio)

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        values = np.asarray(embeddings, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.mean.shape[0]:
            raise ValueError("embedding width does not match the PCA basis")
        return (values - self.mean) @ self.components.T


@dataclass(frozen=True)
class SpectrumComparison:
    """One reference precursor's aligned fragment axis and two intensity predictions."""

    proforma_sequence: str
    charge: int
    fragment_mz: np.ndarray
    student_intensity: np.ndarray
    reference_intensity: np.ndarray
    reference_name: str = "teacher/target"


@dataclass(frozen=True)
class LabeledEmbedding:
    """One named vector in a shared learned space."""

    label: str
    family: str
    vector: np.ndarray
    annotation: str | None = None


@dataclass(frozen=True)
class EmbeddingConnection:
    """A labeled relationship to draw between two projected embeddings."""

    first_label: str
    second_label: str


@dataclass(frozen=True)
class RtObservation:
    """Observed and context-free predicted iRT for one peptide."""

    sequence: str
    observed_irt: float
    predicted_irt: float
    dataset: str = ""


@dataclass(frozen=True)
class RtRegressionMetrics:
    """Scalar summary shown alongside an observed-vs-predicted iRT panel."""

    slope: float
    intercept: float
    r_squared: float
    mae: float


@dataclass(frozen=True)
class IrtStandard:
    """One immutable peptide in the canonical iRT calibration panel."""

    sequence: str
    irt: float
    charge: int = 2


IRT_STANDARDS = (
    IrtStandard("LGGNEQVTR", -24.916114),
    IrtStandard("GAGSSEPVTGLDAK", 0.0009403333333324326),
    IrtStandard("VEATFGVDESNAK", 12.389374888888767),
    IrtStandard("YILAGVENSK", 19.78791066666666),
    IrtStandard("TPVISGGPYEYR", 28.714581222222122),
    IrtStandard("TPVITGAPYEYR", 33.381242999999984),
    IrtStandard("DGLDAASYYAPVR", 42.26388844444456),
    IrtStandard("ADVTPADFSEWSK", 54.621042),
    IrtStandard("GTFIIDPGGVIR", 70.51874133333332),
    IrtStandard("GTFIIDPAAVIR", 87.23322233333332),
    IrtStandard("LFLQFGAQGSPFLK", 100.00282166666665),
)


@dataclass(frozen=True)
class ReferenceSpectrum:
    """Immutable teacher/experimental diagnostic values for one prepared precursor."""

    dataset: str
    sequence: str
    serialized_mods: str
    charge: int
    fragment_mz: np.ndarray
    experimental_intensity: np.ndarray
    teacher_intensity: np.ndarray


@dataclass(frozen=True)
class DiagnosticReferencePanel:
    """Cached fixed panel used for teacher yardsticks and longitudinal butterflies."""

    spectra: tuple[ReferenceSpectrum, ...]

    def save(self, uri: str | Path) -> None:
        metadata = []
        mz_parts = []
        experimental_parts = []
        teacher_parts = []
        offset = 0
        for spectrum in self.spectra:
            shape = tuple(np.asarray(spectrum.experimental_intensity).shape)
            if np.asarray(spectrum.fragment_mz).shape != shape:
                raise ValueError(f"fragment m/z shape mismatch for {spectrum.sequence}")
            if np.asarray(spectrum.teacher_intensity).shape != shape:
                raise ValueError(f"teacher intensity shape mismatch for {spectrum.sequence}")
            size = int(np.prod(shape))
            metadata.append(
                {
                    "dataset": spectrum.dataset,
                    "sequence": spectrum.sequence,
                    "serialized_mods": spectrum.serialized_mods,
                    "charge": spectrum.charge,
                    "shape": shape,
                    "offset": offset,
                    "size": size,
                }
            )
            mz_parts.append(np.asarray(spectrum.fragment_mz, dtype=np.float32).ravel())
            experimental_parts.append(
                np.asarray(spectrum.experimental_intensity, dtype=np.float32).ravel()
            )
            teacher_parts.append(np.asarray(spectrum.teacher_intensity, dtype=np.float32).ravel())
            offset += size
        payload = io.BytesIO()
        np.savez_compressed(
            payload,
            metadata=np.asarray(json.dumps(metadata)),
            fragment_mz=np.concatenate(mz_parts) if mz_parts else np.empty(0, np.float32),
            experimental=(
                np.concatenate(experimental_parts)
                if experimental_parts
                else np.empty(0, np.float32)
            ),
            teacher=np.concatenate(teacher_parts) if teacher_parts else np.empty(0, np.float32),
        )
        with fsspec.open(str(uri), "wb") as stream:
            stream.write(payload.getvalue())

    @classmethod
    def load(cls, uri: str | Path) -> "DiagnosticReferencePanel":
        with fsspec.open(str(uri), "rb") as stream:
            payload = io.BytesIO(stream.read())
        with np.load(payload, allow_pickle=False) as arrays:
            metadata = json.loads(str(arrays["metadata"]))
            result = []
            for row in metadata:
                start = int(row["offset"])
                stop = start + int(row["size"])
                shape = tuple(int(value) for value in row["shape"])
                result.append(
                    ReferenceSpectrum(
                        dataset=str(row["dataset"]),
                        sequence=str(row["sequence"]),
                        serialized_mods=str(row["serialized_mods"]),
                        charge=int(row["charge"]),
                        fragment_mz=arrays["fragment_mz"][start:stop].reshape(shape).copy(),
                        experimental_intensity=(
                            arrays["experimental"][start:stop].reshape(shape).copy()
                        ),
                        teacher_intensity=arrays["teacher"][start:stop].reshape(shape).copy(),
                    )
                )
        return cls(tuple(result))

    def teacher_yardstick(self) -> dict[str, float]:
        """Mean teacher-vs-experimental spectral agreement, separately per dataset."""
        grouped: dict[str, list[float]] = {}
        for spectrum in self.spectra:
            grouped.setdefault(spectrum.dataset, []).append(
                normalized_spectral_angle(
                    spectrum.teacher_intensity, spectrum.experimental_intensity
                )
            )
        return {dataset: float(np.mean(values)) for dataset, values in sorted(grouped.items())}


def normalized_spectral_angle(first: np.ndarray, second: np.ndarray) -> float:
    """Normalized spectral contrast angle in ``[0, 1]`` (one is identical)."""
    a = np.asarray(first, dtype=np.float64).ravel()
    b = np.asarray(second, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("spectra must have the same shape")
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    cosine = float(np.dot(a, b) / denominator) if denominator else 0.0
    return float(1.0 - 2.0 * np.arccos(np.clip(cosine, -1.0, 1.0)) / np.pi)


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("diagnostic plots require the 'tracking' extra") from exc
    return plt


def plot_labeled_embedding_pca(
    embeddings: Sequence[LabeledEmbedding],
    path: str | Path,
    *,
    title: str,
    connections: Sequence[EmbeddingConnection] = (),
    basis: PcaBasis | None = None,
) -> tuple[Path, PcaBasis]:
    """PCA plot for small named vocabularies such as residues, mods, or context factors."""
    if len(embeddings) < 2:
        raise ValueError("at least two labeled embeddings are required")
    vectors = np.stack([np.asarray(item.vector, dtype=np.float64) for item in embeddings])
    basis = basis or PcaBasis.fit(vectors)
    xy = basis.transform(vectors)
    by_label = {item.label: xy[index] for index, item in enumerate(embeddings)}

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    families = list(dict.fromkeys(item.family for item in embeddings))
    palette = plt.get_cmap("tab10")
    family_colors = {
        family: palette(family_index % 10) for family_index, family in enumerate(families)
    }
    family_by_label = {item.label: item.family for item in embeddings}
    markers = ("o", "s", "^", "D", "P", "X", "v", "<", ">")
    line_styles = ("-", "--", "-.", ":")
    for family_index, family in enumerate(families):
        mask = np.asarray([item.family == family for item in embeddings])
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=58 + 8 * (len(families) - family_index - 1),
            marker=markers[family_index % len(markers)],
            alpha=0.78,
            color=family_colors[family],
            label=family,
        )
    for connection in connections:
        first = by_label[connection.first_label]
        second = by_label[connection.second_label]
        ax.plot(
            [first[0], second[0]],
            [first[1], second[1]],
            color=family_colors[family_by_label[connection.first_label]],
            alpha=0.65,
            linewidth=1.3,
            linestyle=line_styles[
                families.index(family_by_label[connection.first_label]) % len(line_styles)
            ],
            zorder=0,
        )
    annotation_offsets = ((5, 5), (5, -12), (5, 18), (5, -25), (5, 31), (5, -38))
    x_tolerance = max(float(np.ptp(xy[:, 0])) * 0.025, 1e-9)
    y_tolerance = max(float(np.ptp(xy[:, 1])) * 0.025, 1e-9)
    previous_points: list[np.ndarray] = []
    for item, point in zip(embeddings, xy, strict=True):
        neighbor_count = sum(
            abs(point[0] - previous[0]) <= x_tolerance
            and abs(point[1] - previous[1]) <= y_tolerance
            for previous in previous_points
        )
        offset = annotation_offsets[min(neighbor_count, len(annotation_offsets) - 1)]
        annotation = item.label if item.annotation is None else item.annotation
        if annotation:
            ax.annotate(
                annotation,
                point,
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
                annotation_clip=False,
            )
        previous_points.append(point)
    ratio = basis.explained_variance_ratio
    ax.set(
        title=title,
        xlabel=f"PC1 ({ratio[0]:.1%})",
        ylabel=f"PC2 ({ratio[1]:.1%})",
    )
    ax.axhline(0, color="0.88", linewidth=0.8, zorder=0)
    ax.axvline(0, color="0.88", linewidth=0.8, zorder=0)
    ax.margins(x=0.08, y=0.08)
    ax.legend(frameon=False)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target, basis


def plot_irt_scatter(
    observations: Sequence[RtObservation],
    path: str | Path,
    *,
    title: str = "Context-free predicted iRT vs observed iRT",
) -> Path:
    """Plot observed/predicted iRT with identity and least-squares fit diagnostics."""
    if len(observations) < 2:
        raise ValueError("at least two iRT observations are required")
    observed = np.asarray([item.observed_irt for item in observations], dtype=np.float64)
    predicted = np.asarray([item.predicted_irt for item in observations], dtype=np.float64)
    metrics = irt_regression_metrics(observations)
    low = float(min(observed.min(), predicted.min()))
    high = float(max(observed.max(), predicted.max()))
    margin = max((high - low) * 0.04, 1.0)
    axis = np.asarray([low - margin, high + margin])

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.5, 7), constrained_layout=True)
    datasets = list(dict.fromkeys(item.dataset or "reference" for item in observations))
    palette = plt.get_cmap("tab10")
    for index, dataset in enumerate(datasets):
        mask = np.asarray([(item.dataset or "reference") == dataset for item in observations])
        ax.scatter(
            observed[mask],
            predicted[mask],
            s=18,
            alpha=0.45,
            color=palette(index % 10),
            label=dataset,
        )
    ax.plot(axis, axis, linestyle="--", color="0.25", linewidth=1.2, label="identity")
    ax.plot(
        axis,
        metrics.slope * axis + metrics.intercept,
        color="#D1495B",
        linewidth=1.4,
        label="fit",
    )
    ax.set(
        xlim=axis,
        ylim=axis,
        aspect="equal",
        xlabel="observed iRT",
        ylabel="predicted context-free iRT",
        title=title,
    )
    ax.text(
        0.03,
        0.97,
        f"slope={metrics.slope:.3f}\nintercept={metrics.intercept:.3f}\n"
        f"R²={metrics.r_squared:.3f}\nMAE={metrics.mae:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    if len(datasets) <= 8:
        ax.legend(frameon=False)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def irt_regression_metrics(observations: Sequence[RtObservation]) -> RtRegressionMetrics:
    """Compute the regression summary used by both plots and experiment tracking."""
    if len(observations) < 2:
        raise ValueError("at least two iRT observations are required")
    observed = np.asarray([item.observed_irt for item in observations], dtype=np.float64)
    predicted = np.asarray([item.predicted_irt for item in observations], dtype=np.float64)
    slope, intercept = np.polyfit(observed, predicted, 1)
    correlation = np.corrcoef(observed, predicted)[0, 1]
    return RtRegressionMetrics(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(correlation**2) if np.isfinite(correlation) else 0.0,
        mae=float(np.mean(np.abs(predicted - observed))),
    )


def plot_spectrum_butterflies(
    references: Sequence[SpectrumComparison],
    path: str | Path,
    *,
    title: str = "Reference spectra: student vs teacher",
) -> Path:
    """Plot mirrored spectra: student above zero and teacher/target below zero.

    Intensities are independently base-peak normalized because this diagnostic compares spectral
    shape rather than absolute response.
    """
    if not references:
        raise ValueError("at least one reference spectrum is required")
    plt = _pyplot()
    fig, axes = plt.subplots(
        len(references),
        1,
        figsize=(11, 3.2 * len(references)),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(title)
    for ax, comparison in zip(axes[:, 0], references, strict=True):
        mz = np.asarray(comparison.fragment_mz, dtype=float).ravel()
        student = np.asarray(comparison.student_intensity, dtype=float).ravel()
        target = np.asarray(comparison.reference_intensity, dtype=float).ravel()
        if not (mz.shape == student.shape == target.shape):
            raise ValueError(f"spectrum arrays do not align for {comparison.proforma_sequence}")
        student = student / max(float(student.max()), 1e-12)
        target = target / max(float(target.max()), 1e-12)
        agreement = normalized_spectral_angle(student, target)
        ax.vlines(mz, 0, student, color="#2878B5", linewidth=1.0, label="student")
        ax.vlines(
            mz,
            0,
            -target,
            color="#E07B39",
            linewidth=1.0,
            label=comparison.reference_name,
        )
        ax.axhline(0, color="0.2", linewidth=0.8)
        ax.set(
            title=(
                f"{comparison.proforma_sequence}  z={comparison.charge}  "
                f"spectral agreement={agreement:.3f}"
            ),
            ylabel="normalized intensity",
            ylim=(-1.08, 1.08),
        )
        ax.legend(frameon=False, ncol=2, loc="upper right")
    axes[-1, 0].set_xlabel("fragment m/z")
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target_path, dpi=160)
    plt.close(fig)
    return target_path


__all__ = [
    "IRT_STANDARDS",
    "PcaBasis",
    "DiagnosticReferencePanel",
    "EmbeddingConnection",
    "LabeledEmbedding",
    "IrtStandard",
    "ReferenceSpectrum",
    "RtObservation",
    "RtRegressionMetrics",
    "SpectrumComparison",
    "normalized_spectral_angle",
    "irt_regression_metrics",
    "plot_irt_scatter",
    "plot_labeled_embedding_pca",
    "plot_spectrum_butterflies",
]
