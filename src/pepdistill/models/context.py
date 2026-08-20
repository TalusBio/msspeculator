"""Acquisition/chromatography context generators for :class:`StudentModel` conditioning.

Two modules, one per side of the backbone's context split:

- :class:`MSContextEncoder` composes ``ms_context`` (MS2 side) from acquisition factors —
  instrument, detector, fragmentation (categorical embeddings) plus collision energy
  (continuous, via an MLP) — so the vector is a learned function of metadata shared across
  every source, rather than a per-source id gradient-descended from scratch.
- :class:`ChromRunbook` generates ``chrom_context`` (RT side) from a per-dataset id, row 0
  reserved as the neutral/iRT row.

Both are zero-init: an all-unknown/energy-less input or the neutral runbook row reproduces
the base (context-free) model exactly (:class:`StudentModel`'s context projections have zero
bias, so ctx=0 -> 0 bias), while their nonzero weights still pass gradient back so the
vectors can learn. Parameter-efficient fine-tuning = freeze the StudentModel and optimize
only these modules.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import nn

DEFAULT_INSTRUMENTS = ("unknown", "Lumos", "QExactive", "Exploris", "timsTOF")
DEFAULT_DETECTORS = ("unknown", "FTMS", "ITMS", "TOF")
DEFAULT_FRAGMENTATIONS = ("unknown", "HCD", "CID", "ETD", "EThcD")


def _widened(table: nn.Embedding, rows: int) -> nn.Embedding:
    """A zero-init table of `rows` rows holding everything `table` has already trained.

    Rebuilt rather than resized in place: the old parameter may already be registered with an
    optimizer, and copying forward is what keeps a trained row attached to its own name.
    """
    wider = nn.Embedding(rows, table.embedding_dim, padding_idx=0)
    nn.init.zeros_(wider.weight)
    with torch.no_grad():
        wider.weight[: table.num_embeddings].copy_(table.weight)
    return wider


def _assign_rows(existing: Mapping[str, int], names: Iterable[str]) -> dict[str, int]:
    """Rows for the names `existing` does not already hold, numbered above its highest.

    Above rather than into gaps, so a sparse index left by an earlier curriculum stays sparse
    instead of handing a new source a row some other source's weights were trained in.

    >>> _assign_rows({"alpha": 1, "beta": 4}, ["beta", "gamma", "gamma", "delta"])
    {'gamma': 5, 'delta': 6}
    """
    fresh = [name for name in dict.fromkeys(names) if name not in existing]
    start = max(existing.values(), default=0) + 1
    return {name: start + offset for offset, name in enumerate(fresh)}


class MSContextEncoder(nn.Module):
    """Compose ``ms_context`` (MS2 side) from acquisition factors: instrument, detector,
    fragmentation (categorical embeddings, index 0 = unknown/blank -> zero row) plus collision
    energy (continuous, via a plain MLP whose first ``Linear`` IS the learned affine — no
    BatchNorm1d: a single-NCE batch would collapse a batch-statistic normalization to a
    degenerate center/scale, so the affine is a trained parameter instead). Every term is
    zero-init, so an all-unknown / energy-less input returns the zero vector — the
    context-free base. Collision energy is never fabricated: a spectrum with no recorded energy
    carries NaN, and the NaN entries of the ``energy`` tensor are masked out per example, so
    that term contributes zero for exactly those rows. (``energy=None`` omits the term for a
    whole call; the real-data path always passes a tensor and relies on the mask, so it is
    reachable only from callers that have no energy axis at all, e.g. tests.)

    Alongside the factors it carries a table of **named acquisition setups**: rows addressed by
    name rather than composed from metadata, for a source that records no factors to compose
    from. A published spectral library is the case — it reports no instrument and no collision
    energy, and a timsTOF ramps energy with ion mobility anyway, so there is nothing for the
    factor terms to consume and its offset from the base model has to be fitted as a row.
    Additive alongside the factor terms and zero-init like them, so a setup nobody named
    changes nothing, and a source with both factors and a name gets both.
    """

    def __init__(
        self,
        context_dim: int = 16,
        instruments: tuple[str, ...] = DEFAULT_INSTRUMENTS,
        detectors: tuple[str, ...] = DEFAULT_DETECTORS,
        fragmentations: tuple[str, ...] = DEFAULT_FRAGMENTATIONS,
        setups: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.instruments = tuple(instruments)
        self.detectors = tuple(detectors)
        self.fragmentations = tuple(fragmentations)
        self._inst_ix = {n: i for i, n in enumerate(self.instruments)}
        self._det_ix = {n: i for i, n in enumerate(self.detectors)}
        self._frag_ix = {n: i for i, n in enumerate(self.fragmentations)}
        self.inst_emb = nn.Embedding(len(self.instruments), context_dim, padding_idx=0)
        self.det_emb = nn.Embedding(len(self.detectors), context_dim, padding_idx=0)
        self.frag_emb = nn.Embedding(len(self.fragmentations), context_dim, padding_idx=0)
        self.energy_mlp = nn.Sequential(
            nn.Linear(1, context_dim), nn.GELU(), nn.Linear(context_dim, context_dim)
        )
        for emb in (self.inst_emb, self.det_emb, self.frag_emb):
            nn.init.zeros_(emb.weight)
        nn.init.zeros_(self.energy_mlp[-1].weight)  # energy term starts neutral (0)
        nn.init.zeros_(self.energy_mlp[-1].bias)
        # Row 0 is the unnamed setup, kept neutral by `padding_idx` exactly as the factor
        # tables' unknown row is: "this source has no name" must cost nothing.
        self._setups: dict[str, int] = dict(setups or {})
        self.setup_emb = nn.Embedding(
            max(self._setups.values(), default=0) + 1, context_dim, padding_idx=0
        )
        nn.init.zeros_(self.setup_emb.weight)

    @property
    def context_dim(self) -> int:
        return self.inst_emb.embedding_dim

    @property
    def setups(self) -> dict[str, int]:
        """Setup name -> row, travelling with the weights those rows index."""
        return dict(self._setups)

    def setup_row(self, name: str) -> int:
        """Row for a setup this encoder has a vector for.

        Raises for an unknown name: substituting row 0 would answer with the base model while
        reporting that the requested setup had been applied.
        """
        try:
            return self._setups[name]
        except KeyError:
            known = ", ".join(sorted(self._setups)) or "none"
            raise KeyError(f"no acquisition setup named {name!r}; this encoder knows: {known}")

    def ensure_setups(self, names: Iterable[str]) -> None:
        """Give every name a row, growing the table if needed, leaving trained rows put.

        >>> enc = MSContextEncoder(context_dim=4)
        >>> enc.ensure_setups(["Evosep60SPD_heron", "Evosep60SPD_heron", "diaPASEF_cyspat"])
        >>> enc.setups
        {'Evosep60SPD_heron': 1, 'diaPASEF_cyspat': 2}
        >>> enc.ensure_setups(["Evosep60SPD_heron"])  # already named: a no-op
        >>> enc.setups
        {'Evosep60SPD_heron': 1, 'diaPASEF_cyspat': 2}
        """
        assigned = _assign_rows(self._setups, names)
        if not assigned:
            return
        needed = max(assigned.values()) + 1  # +1: row 0 is the unnamed setup
        if needed > self.setup_emb.num_embeddings:
            self.setup_emb = _widened(self.setup_emb, needed)
        self._setups.update(assigned)

    def instrument_id(self, name: str) -> int:
        return self._inst_ix.get(name, 0)

    def detector_id(self, name: str) -> int:
        return self._det_ix.get(name, 0)

    def fragmentation_id(self, name: str) -> int:
        return self._frag_ix.get(name, 0)

    def forward(
        self,
        instrument_id: torch.Tensor,
        detector_id: torch.Tensor,
        fragmentation_id: torch.Tensor,
        energy: torch.Tensor | None,
        setup_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # `padding_idx=0` keeps the unknown rows gradient-free. Multiplying by the masks also
        # protects neutrality when loading an older checkpoint whose row 0 learned before that
        # invariant was enforced (`padding_idx` does not rewrite loaded weights).
        inst = self.inst_emb(instrument_id) * instrument_id.ne(0).unsqueeze(-1)
        det = self.det_emb(detector_id) * detector_id.ne(0).unsqueeze(-1)
        frag = self.frag_emb(fragmentation_id) * fragmentation_id.ne(0).unsqueeze(-1)
        out = inst + det + frag
        if setup_id is not None:
            out = out + self.setup_emb(setup_id) * setup_id.ne(0).unsqueeze(-1)
        if energy is not None:
            # Per-example masking, not imputation. A NaN energy means the run recorded none,
            # and it must contribute exactly zero rather than a value we invented.
            #
            # The mask is applied AFTER the MLP on purpose: energy_mlp's first Linear carries a
            # bias, so mlp(0) != 0 — filling missing energy with zero beforehand would inject a
            # learned constant, which is exactly the fabrication this avoids. nan_to_num only
            # stops NaN propagating through the lane that is about to be zeroed.
            present = torch.isfinite(energy)
            term = self.energy_mlp(torch.nan_to_num(energy, nan=0.0).unsqueeze(-1))
            out = out + present.unsqueeze(-1).to(term.dtype) * term
        return out


class ChromRunbook(nn.Module):
    """Per-dataset chromatography context for the RT head. One embedding keyed by dataset with
    row 0 reserved as the iRT / neutral row (context-free). Zero-init, so an untrained book and
    the neutral row both reproduce the base (iRT) RT; other rows learn each dataset's LC offset.

    The book owns the name -> row map rather than being handed one. A row means nothing without
    the name it was trained for, and while the two lived apart a corpus that gained a source
    renumbered the names (the prepared manifest numbers them by sorted position) while the rows
    stayed put, silently reattaching every learned row to a different dataset.
    """

    def __init__(
        self,
        n_datasets: int,
        context_dim: int = 16,
        names: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(
            n_datasets + 1, context_dim, padding_idx=0
        )  # +1: index 0 = neutral (iRT)
        nn.init.zeros_(self.emb.weight)
        self._names: dict[str, int] = dict(names or {})
        for name, row in self._names.items():
            if not 1 <= row <= n_datasets:
                raise ValueError(
                    f"dataset {name!r} claims row {row}, outside 1..{n_datasets}; row 0 is the "
                    "neutral iRT row and is never a dataset"
                )
        # Per-dataset output affine on the RT head. `emb` above is an ADDITIVE bias in feature
        # space, which can bend the mapping but cannot express a rescale — yet a dataset's raw
        # RT differs from the iRT frame by SCALE as much as offset (gradient length, minutes vs
        # indexed units, dead volume). Both zero-init, so scale=exp(0)=1 and shift=0: the
        # neutral row and an untrained book are exactly identity.
        self.log_scale = nn.Embedding(n_datasets + 1, 1, padding_idx=0)
        self.shift = nn.Embedding(n_datasets + 1, 1, padding_idx=0)
        nn.init.zeros_(self.log_scale.weight)
        nn.init.zeros_(self.shift.weight)

    @property
    def context_dim(self) -> int:
        return self.emb.embedding_dim

    @property
    def n_datasets(self) -> int:
        return self.emb.num_embeddings - 1

    @property
    def names(self) -> dict[str, int]:
        """Dataset name -> row, travelling with the weights those rows index."""
        return dict(self._names)

    def row(self, name: str) -> int:
        """Row for a dataset this book has learned.

        Raises for an unknown name: guessing a row would silently predict one dataset's
        chromatography for another, which reads as a mediocre model rather than a mistake.
        """
        try:
            return self._names[name]
        except KeyError:
            known = ", ".join(sorted(self._names)) or "none"
            raise KeyError(
                f"no chromatography row for {name!r}; this book knows: {known}"
            ) from None

    def ensure(self, names: Iterable[str]) -> None:
        """Give every name a row, growing the embeddings if needed, leaving trained rows put.

        The only way a row is assigned. A dataset that already has one keeps it, so a corpus that
        gains a source cannot renumber what is already trained:

        >>> book = ChromRunbook(n_datasets=2, context_dim=4, names={"alpha": 1, "beta": 2})
        >>> book.ensure(["aardvark", "alpha", "beta"])
        >>> book.names
        {'alpha': 1, 'beta': 2, 'aardvark': 3}
        >>> book.n_datasets
        3

        Names already present are a no-op, so repeated calls across a curriculum are safe:

        >>> book.ensure(["alpha"])
        >>> book.names
        {'alpha': 1, 'beta': 2, 'aardvark': 3}

        New rows go above the highest in use, so a book whose index is already sparse stays that
        way rather than reusing a gap some earlier curriculum left behind:

        >>> sparse = ChromRunbook(n_datasets=7, context_dim=4, names={"alpha": 7})
        >>> sparse.ensure(["gamma"])
        >>> sparse.names
        {'alpha': 7, 'gamma': 8}
        """
        assigned = _assign_rows(self._names, names)
        if not assigned:
            return
        needed = max(assigned.values())
        if needed > self.n_datasets:
            self._grow(needed)
        self._names.update(assigned)

    def adopt_names(self, names: Mapping[str, int]) -> None:
        """Take an externally-held index, for a book that was built before it had one.

        The transition path only: every new call site assigns rows through :meth:`ensure`. Refuses
        to overwrite an index the book already has, since the two disagreeing is the failure this
        whole arrangement exists to prevent.
        """
        if self._names and dict(names) != self._names:
            raise ValueError(
                f"runbook already names {sorted(self._names)}; refusing to replace that with "
                f"{sorted(names)}"
            )
        for name, row in names.items():
            if not 1 <= row <= self.n_datasets:
                raise ValueError(f"dataset {name!r} claims row {row}, outside 1..{self.n_datasets}")
        self._names = dict(names)

    def _grow(self, n_datasets: int) -> None:
        """Widen every table to `n_datasets` rows, copying what is already trained."""
        for name in ("emb", "log_scale", "shift"):
            setattr(self, name, _widened(getattr(self, name), n_datasets + 1))

    def forward(self, dataset_id: torch.Tensor) -> torch.Tensor:
        return self.emb(dataset_id) * dataset_id.ne(0).unsqueeze(-1)

    def affine(self, dataset_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(scale, shift)``, each ``(B,)``, for the RT head's output.

        Scale is ``exp(log_scale)``: strictly positive by construction, exactly 1.0 at init,
        and well-behaved multiplicatively. Log space also means weight decay pulls the scale
        toward 1 rather than toward 0, which is the right prior for a rescale.

        Deliberately unclamped — a scale that runs away means the data disagrees with the
        model, and clamping would bury that signal under a value that merely looks poorly fit.
        """
        present = dataset_id.ne(0)
        log_scale = self.log_scale(dataset_id).squeeze(-1) * present
        shift = self.shift(dataset_id).squeeze(-1) * present
        return torch.exp(log_scale), shift

    def neutral(self, n: int, device: torch.device | str) -> torch.Tensor:
        return self.forward(torch.zeros(n, dtype=torch.long, device=device))
