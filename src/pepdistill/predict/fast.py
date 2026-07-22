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

from ..chem import H2O, MOD_DELTA, PROTON, RESIDUE_MASS, ION_TYPES
from ..data.encode import (
    AA_OFFSET,
    CTERM_IDX,
    MOD_SCALE,
    NTERM_IDX,
    PAD_IDX,
    frag_offset,
    use_termini,
)
from ..data.precursors import Precursor
from .library import LIBRARY_COLUMNS

# Token id is just ord(aa) - AA_OFFSET (no lookup). Residue mass still needs a table,
# indexed by ord(aa) so it vectorizes over the raw sequence bytes of a bucket.
_AA_MASS = np.zeros(256, dtype=np.float64)
for _aa, _m in RESIDUE_MASS.items():
    _AA_MASS[ord(_aa)] = _m

# Per-column (ion, charge) metadata in ION_TYPES order.
_ION_IS_B = np.array([ion == "b" for ion, _ in ION_TYPES], dtype=bool)
_ION_Z = np.array([z for _, z in ION_TYPES], dtype=np.float64)


class ModelRunner:
    """Adapts either a torch StudentModel or an ONNX session to one call signature."""

    def run(self, tokens: np.ndarray, mod_delta: np.ndarray, charge: np.ndarray):
        """Return (ms2 (B,L-1,n_ion) in [0,1], rt (B,) native, ccs (B,) native)."""
        raise NotImplementedError


class TorchRunner(ModelRunner):
    def __init__(self, model, device: str = "cpu") -> None:
        import torch

        self._torch = torch
        self.model = model.to(device).eval()
        self.device = device

    def run(self, tokens, mod_delta, charge):
        torch = self._torch
        with torch.no_grad():
            ms2, rt, ccs = self.model.forward_dense(
                torch.from_numpy(tokens).to(self.device),
                torch.from_numpy(mod_delta).to(self.device),
                torch.from_numpy(charge).to(self.device),
            )
        return ms2.cpu().numpy(), rt.cpu().numpy(), ccs.cpu().numpy()


def _bucket_arrays(precs: list[Precursor], length: int):
    """Dense token/mod/charge arrays for a same-length bucket (vectorized).

    Tokens are wrapped with N/C-term ids -> shape (B, length+2). ``residue_mass`` stays
    (B, length): termini carry no mass and never enter m/z.
    """
    b = len(precs)
    off = frag_offset()
    extra = 2 if use_termini() else 0
    # Sequences share a length -> one contiguous byte matrix, then table lookup.
    seq_bytes = np.frombuffer("".join(p.peptide.sequence for p in precs).encode(), dtype=np.uint8)
    codes = seq_bytes.reshape(b, length)
    residue_mass = _AA_MASS[codes].copy()  # (B, L) float64, residues only

    tok = np.full((b, length + extra), PAD_IDX, dtype=np.int64)
    if use_termini():
        tok[:, 0] = NTERM_IDX
        tok[:, 1 + length] = CTERM_IDX
    tok[:, off : off + length] = codes.astype(np.int64) - AA_OFFSET  # ord(aa) - ord('A')

    mod_delta = np.zeros((b, length + extra), dtype=np.float32)
    for i, p in enumerate(precs):  # O(precursors), not O(fragments)
        for site, name in p.peptide.mods:
            d = MOD_DELTA[name]
            residue_mass[i, site] += d
            mod_delta[i, off + site] += d / MOD_SCALE
    charge = np.array([p.charge for p in precs], dtype=np.int64)
    return tok, mod_delta, charge, residue_mass


def _fragment_mz(residue_mass: np.ndarray, charge: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized fragment m/z and precursor m/z for a same-length bucket.

    Returns (mz (B, L-1, n_ion), precursor_mz (B,)).
    """
    prefix = np.cumsum(residue_mass, axis=1)  # (B, L); prefix[:,k] = sum residues 0..k
    total = prefix[:, -1]  # (B,)
    # position i (0..L-2): b ordinal i+1 -> prefix[:, i]; y -> total - prefix[:, i] + H2O
    b_neutral = prefix[:, :-1]  # (B, L-1)
    y_neutral = total[:, None] - prefix[:, :-1] + H2O  # (B, L-1)

    # (B, L-1, n_ion): pick b or y neutral per column, then charge arithmetic.
    neutral = np.where(_ION_IS_B[None, None, :], b_neutral[..., None], y_neutral[..., None])
    z = _ION_Z[None, None, :]
    mz = (neutral + z * PROTON) / z

    precursor_mz = (total + H2O + charge * PROTON) / charge
    return mz.astype(np.float64), precursor_mz


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
            tokens, mod_delta, charge, residue_mass = _bucket_arrays(chunk, length)
            ms2, rt, ccs = runner.run(tokens, mod_delta, charge)
            # Fragment sites are at adjacent-pool indices [off, off+frag_pos).
            off = frag_offset()
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
