"""AlphaPeptDeep teacher. Requires the optional ``teacher`` extra (peptdeep)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..chem import ION_TYPES
from ..data.precursors import Precursor
from .base import PrecursorLabels, Teacher

# peptdeep fragment-intensity columns matching our ION_TYPES, in order.
_PEPTDEEP_ION_COLS = tuple(f"{ion}_z{z}" for ion, z in ION_TYPES)

_IMPORT_HINT = (
    "peptdeep is required for the AlphaPeptDeep teacher. Install the extra:\n"
    "    uv pip install 'pepdistill[teacher]'"
)


class PeptDeepTeacher(Teacher):
    """Wraps :class:`peptdeep.pretrained_models.ModelManager`."""

    name = "alphapeptdeep"

    def __init__(
        self,
        device: str = "cpu",
        nce: float = 30.0,
        instrument: str = "Lumos",
        batch_size: int = 1024,
    ) -> None:
        try:
            from peptdeep.pretrained_models import ModelManager
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(_IMPORT_HINT) from exc

        self.nce = nce
        self.instrument = instrument
        self.batch_size = batch_size
        self._mgr = ModelManager(mask_modloss=True, device=device)
        self._mgr.load_installed_models()

    def _frame(self, precursors: list[Precursor]) -> pd.DataFrame:
        rows = []
        for i, p in enumerate(precursors):
            pep = p.peptide
            rows.append(
                {
                    "sequence": pep.sequence,
                    "mods": ";".join(name for _, name in pep.mods),
                    # peptdeep/alphabase mod_sites are 1-based.
                    "mod_sites": ";".join(str(site + 1) for site, _ in pep.mods),
                    "charge": p.charge,
                    "nce": self.nce,
                    "instrument": self.instrument,
                    # predict_all sorts by length and resets the index, so we carry an
                    # explicit column to restore the caller's order afterwards.
                    "orig_idx": i,
                }
            )
        return pd.DataFrame(rows)

    def predict(self, precursors: list[Precursor]) -> list[PrecursorLabels]:  # pragma: no cover
        df = self._frame(precursors)
        res = self._mgr.predict_all(
            df, predict_items=["rt", "mobility", "ms2"], multiprocessing=False
        )
        prec_df: pd.DataFrame = res["precursor_df"]
        frag_df: pd.DataFrame = res["fragment_intensity_df"]
        if "orig_idx" not in prec_df.columns:
            raise RuntimeError("peptdeep dropped the orig_idx column; cannot restore order")

        # Column presence varies with the loaded model; fill missing ion types with 0.
        cols = [c if c in frag_df.columns else None for c in _PEPTDEEP_ION_COLS]

        rt_col = "rt_pred" if "rt_pred" in prec_df.columns else "rt_norm_pred"
        ccs_col = "ccs_pred" if "ccs_pred" in prec_df.columns else "mobility_pred"

        # Pre-size and place each label at its original position (predict_all reorders).
        out: list[PrecursorLabels | None] = [None] * len(precursors)
        for row in prec_df.itertuples():
            start, stop = int(row.frag_start_idx), int(row.frag_stop_idx)
            block = frag_df.iloc[start:stop]
            ms2 = np.zeros((len(block), len(ION_TYPES)), dtype=np.float32)
            for j, col in enumerate(cols):
                if col is not None:
                    ms2[:, j] = block[col].to_numpy(dtype=np.float32)
            peak = ms2.max() if ms2.size else 0.0
            if peak > 0:
                ms2 /= peak
            out[int(row.orig_idx)] = PrecursorLabels(
                ms2=ms2, rt=float(getattr(row, rt_col)), ccs=float(getattr(row, ccs_col))
            )
        if any(o is None for o in out):
            raise RuntimeError("peptdeep returned fewer precursors than provided")
        return out  # type: ignore[return-value]
