"""Run the student and assemble a long-format spectral library (parquet)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from ..chem import ION_TYPES, fragment_mz_matrix
from ..data.encode import collate, frag_offset
from ..data.precursors import Precursor
from ..models.student import StudentModel

LIBRARY_COLUMNS = [
    "modified_sequence",
    "stripped_sequence",
    "charge",
    "precursor_mz",
    "rt_pred",
    "ccs_pred",
    "ion_type",
    "fragment_charge",
    "fragment_ordinal",
    "fragment_mz",
    "relative_intensity",
]


@torch.no_grad()
def predict_library(
    model: StudentModel,
    precursors: list[Precursor],
    batch_size: int = 512,
    min_intensity: float = 0.01,
    device: str = "cpu",
) -> pd.DataFrame:
    """Predict fragment intensities/RT/CCS and emit one row per retained fragment."""
    model.eval().to(device)
    rows: list[dict] = []

    for start in range(0, len(precursors), batch_size):
        chunk = precursors[start : start + batch_size]
        batch = collate(chunk).to(device)
        out = model.denormalize(model(batch))
        ms2 = out["ms2"].cpu()
        rt = out["rt"].cpu()
        ccs = out["ccs"].cpu()

        for k, prec in enumerate(chunk):
            pep = prec.peptide
            n = pep.length
            mz_matrix = fragment_mz_matrix(pep.sequence, list(pep.mods))  # (n-1, n_ion)
            # Fragment sites live at adjacent-pool indices [off, off+n-1).
            off = frag_offset()
            intensities = ms2[k, off : off + n - 1]  # (n-1, n_ion)
            peak = float(intensities.max()) if intensities.numel() else 0.0
            if peak <= 0:
                continue
            base = {
                "modified_sequence": pep.modified_sequence(),
                "stripped_sequence": pep.sequence,
                "charge": prec.charge,
                "precursor_mz": pep.precursor_mz(prec.charge),
                "rt_pred": float(rt[k]),
                "ccs_pred": float(ccs[k]),
            }
            for i in range(n - 1):
                for j, (ion, z) in enumerate(ION_TYPES):
                    rel = float(intensities[i, j]) / peak
                    if rel < min_intensity:
                        continue
                    ordinal = (i + 1) if ion == "b" else (n - 1 - i)
                    rows.append(
                        {
                            **base,
                            "ion_type": ion,
                            "fragment_charge": z,
                            "fragment_ordinal": ordinal,
                            "fragment_mz": mz_matrix[i][j],
                            "relative_intensity": rel,
                        }
                    )

    return pd.DataFrame(rows, columns=LIBRARY_COLUMNS)


def write_library(df: pd.DataFrame, path: str | Path) -> None:
    df.to_parquet(path, index=False)
