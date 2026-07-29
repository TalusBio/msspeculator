"""Vectorized, length-bucketed spectral-library prediction.

The naive predictor loops in Python over every fragment (millions of rows); that, not
the network, was the bottleneck. Here we group precursors by length so each bucket is a
dense tensor with no padding, run the model once per bucket, and compute fragment m/z +
assemble the output with pure numpy — no per-fragment Python. Only O(precursors) Python
remains (mod placement + modified-sequence strings), never O(fragments).

Supports a torch model or an ONNX session (see :mod:`pepdistill.predict.onnx`).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from ..chem import ION_TYPES
from ..data.precursors import Precursor
from .library import LIBRARY_COLUMNS

# Per-column ion-type metadata in ION_TYPES order (used for the output ordinal/ion grid).
_ION_IS_B = np.array([ion == "b" for ion, _ in ION_TYPES], dtype=bool)


class ModelRunner:
    """Adapts either a torch StudentModel or an ONNX session to one call signature."""

    def run(
        self,
        tokens: np.ndarray,
        mod_comp: np.ndarray,
        mod_mass: np.ndarray,
        mod_present: np.ndarray,
        mod_named: np.ndarray,
        charge: np.ndarray,
    ):
        """Return (ms2 (B,L-1,n_ion) in [0,1], rt (B,) native, ccs (B,) native)."""
        raise NotImplementedError


class TorchRunner(ModelRunner):
    def __init__(self, model, device: str = "cpu", ms_context: np.ndarray | None = None) -> None:
        import torch

        self._torch = torch
        self.model = model.to(device).eval()
        self.device = device
        # Optional MS context (MS2 only), e.g. encoder(instrument, detector, fragmentation,
        # energy). One vector broadcast to every precursor in a bucket. RT/CCS stay
        # context-free (iRT is run-independent; CCS is peptide+charge only). Kept as the
        # plain ndarray the caller passed in; converted to a tensor per-bucket in run().
        self.ms_context = ms_context

    def run(self, tokens, mod_comp, mod_mass, mod_present, mod_named, charge):
        torch = self._torch
        ctx = None
        if self.ms_context is not None:
            ctx_t = torch.as_tensor(self.ms_context, dtype=torch.float32, device=self.device)
            ctx = ctx_t.reshape(1, -1).expand(len(tokens), -1)
        with torch.no_grad():
            ms2, rt, ccs = self.model.forward_dense(
                torch.from_numpy(tokens).to(self.device),
                torch.from_numpy(mod_comp).to(self.device),
                torch.from_numpy(mod_mass).to(self.device),
                torch.from_numpy(mod_present).to(self.device),
                torch.from_numpy(mod_named).to(self.device),
                torch.from_numpy(charge).to(self.device),
                ms_context=ctx,
            )
        return ms2.cpu().numpy(), rt.cpu().numpy(), ccs.cpu().numpy()


def _bucket_arrays(precs: list[Precursor], length: int):
    """Dense token/mod/charge/residue-mass arrays for a same-length bucket (via the ext).

    Tokens are wrapped with N/C-term ids -> shape (B, length+2). ``residue_mass`` stays
    (B, length): termini carry no mass and never enter m/z.
    """
    import pepdistill_rs as _rs

    peptides = [p.peptide for p in precs]
    charges = [int(p.charge) for p in precs]
    a = _rs.bucket_arrays(peptides, charges, length)
    return (
        a["tokens"], a["mod_comp"], a["mod_mass"], a["mod_present"], a["mod_named"],
        a["charge"], a["residue_mass"],
    )


def _fragment_mz(residue_mass: np.ndarray, charge: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized fragment m/z and precursor m/z for a same-length bucket (via the ext).

    Returns (mz (B, L-1, n_ion), precursor_mz (B,)).
    """
    import pepdistill_rs as _rs

    return _rs.bucket_fragment_mz(residue_mass, charge)


def predict_library_fast(
    runner: ModelRunner,
    precursors: list[Precursor],
    batch_size: int = 4096,
    min_intensity: float = 0.01,
) -> pd.DataFrame:
    by_len: dict[int, list[Precursor]] = defaultdict(list)
    for p in precursors:
        by_len[p.peptide.length].append(p)

    col_arrays: dict[str, list[np.ndarray]] = {c: [] for c in LIBRARY_COLUMNS}
    mod_seq_all: list[str] = []
    strip_seq_all: list[str] = []

    n_ion = len(ION_TYPES)
    ion_names = np.array([ion for ion, _ in ION_TYPES])
    ion_charges = np.array([z for _, z in ION_TYPES], dtype=np.int64)

    for length, precs in by_len.items():
        if length < 2:
            continue
        frag_pos = length - 1
        for start in range(0, len(precs), batch_size):
            chunk = precs[start : start + batch_size]
            tokens, mod_comp, mod_mass, mod_present, mod_named, charge, residue_mass = (
                _bucket_arrays(chunk, length)
            )
            ms2, rt, ccs = runner.run(tokens, mod_comp, mod_mass, mod_present, mod_named, charge)
            # Fragment sites are at adjacent-pool indices [off, off+frag_pos). Termini are
            # mandatory, so the mandatory N-term token occupies index 0 and off is always 1.
            off = 1
            ms2 = ms2[:, off : off + frag_pos, :]
            mz, precursor_mz = _fragment_mz(residue_mass, charge)

            bsz = len(chunk)
            # Normalize each precursor's spectrum to its base peak.
            peak = ms2.reshape(bsz, -1).max(axis=1)
            peak[peak <= 0] = 1.0
            rel = ms2 / peak[:, None, None]  # (B, F, n_ion)

            # ordinal grid, same for all precursors in this bucket.
            pos = np.arange(frag_pos)
            ordinal = np.where(_ION_IS_B[None, :], pos[:, None] + 1, frag_pos - pos[:, None])
            ordinal_grid = np.broadcast_to(ordinal, (bsz, frag_pos, n_ion))
            ion_grid = np.broadcast_to(ion_names[None, None, :], (bsz, frag_pos, n_ion))
            zcol_grid = np.broadcast_to(ion_charges[None, None, :], (bsz, frag_pos, n_ion))
            prec_idx = np.broadcast_to(np.arange(bsz)[:, None, None], (bsz, frag_pos, n_ion))

            keep = rel >= min_intensity  # (B, F, n_ion)
            flat = keep.reshape(-1)
            sel = np.nonzero(flat)[0]
            if sel.size == 0:
                continue

            pidx = prec_idx.reshape(-1)[sel]
            col_arrays["charge"].append(charge[pidx].astype(np.int64))
            col_arrays["precursor_mz"].append(precursor_mz[pidx])
            col_arrays["rt_pred"].append(rt[pidx])
            col_arrays["ccs_pred"].append(ccs[pidx])
            col_arrays["ion_type"].append(ion_grid.reshape(-1)[sel])
            col_arrays["fragment_charge"].append(zcol_grid.reshape(-1)[sel].astype(np.int64))
            col_arrays["fragment_ordinal"].append(ordinal_grid.reshape(-1)[sel].astype(np.int64))
            col_arrays["fragment_mz"].append(mz.reshape(-1)[sel])
            col_arrays["relative_intensity"].append(rel.reshape(-1)[sel])

            # String columns: build once per precursor (O(precursors)), expand by pidx.
            mods_present = any(p.peptide.mods for p in chunk)
            strip = np.array([p.peptide.sequence for p in chunk])
            if mods_present:
                modseq = np.array([p.peptide.modified_sequence() for p in chunk])
            else:
                modseq = strip
            strip_seq_all.append(strip[pidx])
            mod_seq_all.append(modseq[pidx])

    if not col_arrays["charge"]:
        return pd.DataFrame(columns=LIBRARY_COLUMNS)

    data = {
        "modified_sequence": np.concatenate(mod_seq_all),
        "stripped_sequence": np.concatenate(strip_seq_all),
    }
    for c in LIBRARY_COLUMNS:
        if c in ("modified_sequence", "stripped_sequence"):
            continue
        data[c] = np.concatenate(col_arrays[c])
    return pd.DataFrame(data, columns=LIBRARY_COLUMNS)
