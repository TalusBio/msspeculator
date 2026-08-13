"""AlphaPeptDeep teacher. Requires the optional ``teacher`` extra (peptdeep)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..chem import ION_TYPES
from ..data.precursors import Precursor
from .base import PrecursorLabels, Teacher


def _mod_name(pep, spec) -> str:
    """peptdeep identifies modifications by NAME, so a bare mass delta cannot be expressed.

    Refuse rather than invent a name: a fabricated identifier would be silently mis-looked-up
    against peptdeep's own modification table, producing a confident wrong spectrum.
    """
    if not isinstance(spec, str):
        raise ValueError(
            f"peptide {pep.modified_sequence()!r} carries a mass-only modification "
            f"({spec:+}), which the peptdeep teacher cannot represent — it identifies "
            "modifications by name. Supply a named modification, or label this peptide with a "
            "teacher that accepts raw deltas."
        )
    return spec


# alphabase site suffixes, and the mod_site each one implies. Verified against alphabase:
# `get_candidate_sites` documents "0: N-term, -1: C-term, 1-n: others", and
# `smiles/peptide.py` keys residue-anchored terminal mods ("Q^Any_N-term") off mod_site "0".
_NTERM_SUFFIXES = ("Any_N-term", "Protein_N-term")
_CTERM_SUFFIXES = ("Any_C-term", "Protein_C-term")


def _site_for_alphabase_name(name: str, residue_site: int) -> int:
    """The mod_site that alphabase expects for an already-resolved modification name.

    Derived from the name itself so the name and the site cannot disagree: a residue-anchored
    terminal mod such as ``Glu->pyro-Glu@E^Any_N-term`` is filed at site 0 even though it also
    names a residue, while a plain side-chain mod on the first residue is filed at site 1.
    """
    suffix = name.split("@", 1)[1] if "@" in name else ""
    terminus = suffix.split("^")[-1] if "^" in suffix else suffix
    if terminus in _NTERM_SUFFIXES:
        return 0
    if terminus in _CTERM_SUFFIXES:
        return -1
    return residue_site + 1


def _alphabase_mod(pep, site, spec) -> tuple[str, int]:
    """Map one of our ``(site, spec)`` modifications onto ``(alphabase name, mod_site)``.

    Our vocabulary is deliberately mixed: ``unimod::ALIASES`` freezes four historical names, two
    of which are already alphabase-style ("Carbamidomethyl@C"), while everything else is a bare
    UNIMOD title ("Phospho", "TMT6plex"). alphabase keys every modification as ``Name@Site``, so
    bare names must gain the right suffix and qualified names must be left alone.

    Name and site are resolved together because they are coupled — see
    :func:`_site_for_alphabase_name`. Every candidate is checked against alphabase's own table,
    so an unresolvable modification still raises instead of reaching the model mis-specified.
    """
    from alphabase.constants.modification import MOD_MASS

    name = _mod_name(pep, spec)
    # Two of our frozen aliases already carry a residue suffix ("Oxidation@M"), so the base name
    # has to be recovered before a terminal or different-residue form can be built from it.
    base = name.split("@", 1)[0]

    # The terminal branch must come before any "already an alphabase name" shortcut: our site is
    # the authority on where the modification sits, and a residue-suffixed alias found on a
    # terminus would otherwise be filed on residue 1 -- a plausible, confident, wrong spectrum.
    if site == "n":
        candidates = [f"{base}@{suffix}" for suffix in _NTERM_SUFFIXES]
    elif site == "c":
        candidates = [f"{base}@{suffix}" for suffix in _CTERM_SUFFIXES]
    else:
        residue = pep.sequence[site] if 0 <= site < len(pep.sequence) else ""
        declared = name.split("@", 1)[1] if "@" in name else ""
        if len(declared) == 1 and declared != residue:
            raise ValueError(
                f"peptide {pep.modified_sequence()!r} carries {spec!r} at site {site!r}, whose "
                f"residue is {residue!r} rather than the {declared!r} the name declares; refusing "
                "to relocate a modification onto a residue it does not name."
            )
        # Prefer the plain side-chain form; only fall back to a terminal form when the residue
        # actually sits at that terminus, which is how mods like pyro-Glu are registered.
        candidates = [f"{base}@{residue}"]
        if site == 0:
            candidates += [f"{base}@{residue}^{s}" for s in _NTERM_SUFFIXES]
            candidates += [f"{base}@{s}" for s in _NTERM_SUFFIXES]
        if site == len(pep.sequence) - 1:
            candidates += [f"{base}@{residue}^{s}" for s in _CTERM_SUFFIXES]
            candidates += [f"{base}@{s}" for s in _CTERM_SUFFIXES]

    resolved = next((option for option in candidates if option in MOD_MASS), None)
    if resolved is None:
        raise ValueError(
            f"peptide {pep.modified_sequence()!r} carries modification {spec!r} at site "
            f"{site!r}, which does not resolve to any alphabase modification name (tried "
            f"{candidates}). Add it with alphabase's add_new_modifications before using this "
            "teacher, rather than substituting a different modification."
        )
    return resolved, _site_for_alphabase_name(resolved, site if isinstance(site, int) else 0)


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
        # We only consume ordinary b/y columns below, so mod-loss outputs have no effect here.
        # ``None`` is intentional: peptdeep warns for either True or False because
        # ``mask_modloss`` is deprecated; None leaves the model's charged-fragment configuration
        # untouched while keeping the ordinary b/y columns we consume.
        self._mgr = ModelManager(mask_modloss=None, device=device)
        self._mgr.load_installed_models()

    def _frame(self, precursors: list[Precursor], nces=None) -> pd.DataFrame:
        rows = []
        for i, p in enumerate(precursors):
            pep = p.peptide
            resolved = [_alphabase_mod(pep, site, spec) for site, spec in pep.mods]
            rows.append(
                {
                    "sequence": pep.sequence,
                    "mods": ";".join(name for name, _ in resolved),
                    # Residue sites are 1-based; 0 is the peptide N-term and -1 the C-term.
                    "mod_sites": ";".join(str(mod_site) for _, mod_site in resolved),
                    "charge": p.charge,
                    "nce": self.nce if nces is None else float(nces[i]),
                    "instrument": self.instrument,
                    # predict_all sorts by length and resets the index, so we carry an
                    # explicit column to restore the caller's order afterwards.
                    "orig_idx": i,
                }
            )
        return pd.DataFrame(rows)

    def predict(
        self, precursors: list[Precursor], nces=None
    ) -> list[PrecursorLabels]:  # pragma: no cover
        df = self._frame(precursors, nces)
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
