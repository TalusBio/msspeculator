"""Optional Rust-accelerated collate (see ``rust/`` crate, module ``pepdistill_rs``).

Pure-Python :func:`pepdistill.data.encode.collate` is the reference oracle; this is a
drop-in accelerator that packs the numpy arrays in Rust and wraps them (zero-copy) as the
same :class:`Batch`. Chemistry constants stay in Python: we compute each mod's scaled delta
here and hand Rust only ``(site, delta)`` pairs to place. Import is best-effort — if the
crate isn't built (``maturin develop`` in ``rust/``), :data:`HAVE_RS` is False and callers
fall back to the pure path.
"""

from __future__ import annotations

import torch

from ..chem import MOD_DELTA
from .encode import MOD_SCALE, Batch, use_termini
from .precursors import Precursor

try:
    import pepdistill_rs as _rs

    HAVE_RS = True
except ImportError:  # crate not built
    _rs = None
    HAVE_RS = False


def collate_rs(precursors: list[Precursor]) -> Batch:
    """Rust-packed equivalent of :func:`pepdistill.data.encode.collate`."""
    if _rs is None:
        raise RuntimeError("pepdistill_rs not built; run `maturin develop` in rust/")

    seqs = [p.peptide.sequence for p in precursors]
    charges = [int(p.charge) for p in precursors]
    mod_sites: list[list[int]] = []
    mod_deltas: list[list[float]] = []
    for p in precursors:
        sites, deltas = [], []
        for site, name in p.peptide.mods:
            sites.append(int(site))
            deltas.append(MOD_DELTA[name] / MOD_SCALE)
        mod_sites.append(sites)
        mod_deltas.append(deltas)

    a = _rs.collate(seqs, charges, mod_sites, mod_deltas, use_termini())
    return Batch(
        torch.from_numpy(a["tokens"]),
        torch.from_numpy(a["mod_delta"]),
        torch.from_numpy(a["charge"]),
        torch.from_numpy(a["lengths"]),
        torch.from_numpy(a["pad_mask"]),
        torch.from_numpy(a["frag_mask"]),
    )
