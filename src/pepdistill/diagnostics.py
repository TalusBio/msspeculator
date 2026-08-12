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
        ratio = variance[:n_components] / variance.sum() if variance.sum() else np.zeros(n_components)
        return cls(mean, vt[:n_components], ratio)

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        values = np.asarray(embeddings, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.mean.shape[0]:
            raise ValueError("embedding width does not match the PCA basis")
        return (values - self.mean) @ self.components.T


@dataclass(frozen=True)
class SpectrumComparison:
    """One reference precursor's aligned fragment axis and two intensity predictions."""

    modified_sequence: str
    charge: int
    fragment_mz: np.ndarray
    student_intensity: np.ndarray
    reference_intensity: np.ndarray
    reference_name: str = "teacher/target"


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
            teacher_parts.append(
                np.asarray(spectrum.teacher_intensity, dtype=np.float32).ravel()
            )
            offset += size
        payload = io.BytesIO()
        np.savez_compressed(
            payload,
            metadata=np.asarray(json.dumps(metadata)),
            fragment_mz=np.concatenate(mz_parts) if mz_parts else np.empty(0, np.float32),
            experimental=(
                np.concatenate(experimental_parts) if experimental_parts else np.empty(0, np.float32)
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
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("diagnostic plots require the 'tracking' extra") from exc
    return plt


def plot_embedding_pca(
    coordinates: np.ndarray,
    lengths: Sequence[int],
    modified: Sequence[bool],
    path: str | Path,
    *,
    title: str,
    explained_variance_ratio: Sequence[float] | None = None,
) -> Path:
    """Plot a fixed reference panel in a two-dimensional latent PCA frame."""
    xy = np.asarray(coordinates)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] == 0:
        raise ValueError("PCA coordinates must have shape (n, 2)")
    lengths_array = np.asarray(lengths)
    modified_array = np.asarray(modified, dtype=bool)
    if len(xy) != len(lengths_array) or len(xy) != len(modified_array):
        raise ValueError("coordinate metadata lengths do not match")

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    for mask, marker, label in (
        (~modified_array, "o", "unmodified"),
        (modified_array, "^", "modified"),
    ):
        if mask.any():
            points = ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                c=lengths_array[mask],
                cmap="viridis",
                marker=marker,
                s=34,
                alpha=0.78,
                linewidths=0.25,
                edgecolors="black",
                label=label,
            )
    ratio = [] if explained_variance_ratio is None else list(explained_variance_ratio)
    x_label = f"PC1 ({ratio[0]:.1%})" if len(ratio) > 0 else "PC1"
    y_label = f"PC2 ({ratio[1]:.1%})" if len(ratio) > 1 else "PC2"
    ax.set(xlabel=x_label, ylabel=y_label, title=title)
    ax.axhline(0, color="0.88", linewidth=0.8, zorder=0)
    ax.axvline(0, color="0.88", linewidth=0.8, zorder=0)
    ax.legend(frameon=False)
    fig.colorbar(points, ax=ax, label="peptide length")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


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
        len(references), 1, figsize=(11, 3.2 * len(references)), squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(title)
    for ax, comparison in zip(axes[:, 0], references, strict=True):
        mz = np.asarray(comparison.fragment_mz, dtype=float).ravel()
        student = np.asarray(comparison.student_intensity, dtype=float).ravel()
        target = np.asarray(comparison.reference_intensity, dtype=float).ravel()
        if not (mz.shape == student.shape == target.shape):
            raise ValueError(
                f"spectrum arrays do not align for {comparison.modified_sequence}"
            )
        student = student / max(float(student.max()), 1e-12)
        target = target / max(float(target.max()), 1e-12)
        agreement = normalized_spectral_angle(student, target)
        ax.vlines(mz, 0, student, color="#2878B5", linewidth=1.0, label="student")
        ax.vlines(
            mz, 0, -target, color="#E07B39", linewidth=1.0,
            label=comparison.reference_name,
        )
        ax.axhline(0, color="0.2", linewidth=0.8)
        ax.set(
            title=(
                f"{comparison.modified_sequence}  z={comparison.charge}  "
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
    "PcaBasis",
    "DiagnosticReferencePanel",
    "ReferenceSpectrum",
    "SpectrumComparison",
    "normalized_spectral_angle",
    "plot_embedding_pca",
    "plot_spectrum_butterflies",
]
