"""AlphaPeptDeep teacher. Requires the optional ``teacher`` extra (peptdeep)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..chem import ION_TYPES, unimod_title
from ..data.precursors import Precursor
from .base import PrecursorLabels, Teacher


def _modification_title(pep, spec) -> str:
    """The bare UNIMOD title for one modification spec, for peptdeep to key its table on.

    ``pep`` is only used to name the offending peptide in an error. ``spec`` is one of its mod
    specs: a ``"UNIMOD:<accession>"`` identity, or a ``float`` bare mass delta.

    The *title*, deliberately, not our alias. Aliases are a read-only compatibility table for
    input, and two of the four carry a residue suffix, so an alias lookup returns
    ``Carbamidomethyl@C`` for accession 4 but a bare ``Phospho`` for 21. The caller appends the
    site itself, so it needs the bare title every time:

    >>> from pepdistill.chem import Peptide
    >>> peptide = Peptide.from_string("AC[UNIMOD:4]DES[UNIMOD:21]K")
    >>> [_modification_title(peptide, spec) for _, spec in peptide.mods]
    ['Carbamidomethyl', 'Phospho']

    A bare mass delta has no name to translate to, and inventing one would be looked up against
    peptdeep's table and yield a confident wrong spectrum:

    >>> _modification_title(Peptide.from_string("PEP[+15.5]TIDEK"), 15.5)
    Traceback (most recent call last):
    ValueError: peptide 'PEP[+15.5]TIDEK' carries a mass-only modification (+15.5), ...
    """
    if not isinstance(spec, str):
        raise ValueError(
            f"peptide {pep.modified_sequence()!r} carries a mass-only modification "
            f"({spec:+}), which the peptdeep teacher cannot represent — it identifies "
            "modifications by name. Supply a named modification, or label this peptide with a "
            "teacher that accepts raw deltas."
        )
    if not spec.startswith("UNIMOD:"):
        raise ValueError(
            f"peptide {pep.modified_sequence()!r} carries {spec!r}, which is not a UNIMOD "
            "accession. peptdeep keys its table on UNIMOD titles, so an elemental formula has "
            "no name to translate to even though it is a perfectly good modification for us."
        )
    title = unimod_title(int(spec.removeprefix("UNIMOD:")))
    if title is None:
        raise ValueError(
            f"peptide {pep.modified_sequence()!r} carries {spec!r}, which is not in the vendored "
            "UNIMOD table, so it cannot be named for the peptdeep teacher."
        )
    return title


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

    This is the boundary at which our canonical identity becomes peptdeep's spelling, and the only
    place a foreign one is emitted. alphabase keys every modification as ``Name@Site``, so the bare
    UNIMOD title gains the suffix that our site implies.

    Name and site are resolved together because they are coupled — see
    :func:`_site_for_alphabase_name`. Every candidate is checked against alphabase's own table,
    so an unresolvable modification still raises instead of reaching the model mis-specified.
    """
    from alphabase.constants.modification import MOD_MASS

    base = _modification_title(pep, spec)

    # The terminal branch must come before any "already an alphabase name" shortcut: our site is
    # the authority on where the modification sits, and a residue-suffixed alias found on a
    # terminus would otherwise be filed on residue 1 -- a plausible, confident, wrong spectrum.
    if site == "n":
        candidates = [f"{base}@{suffix}" for suffix in _NTERM_SUFFIXES]
    elif site == "c":
        candidates = [f"{base}@{suffix}" for suffix in _CTERM_SUFFIXES]
    else:
        residue = pep.sequence[site] if 0 <= site < len(pep.sequence) else ""
        # No "does the name's residue match this site" check any more: a bare title declares no
        # residue, so there is nothing to contradict. A modification on a residue it does not occur
        # on is caught below instead, by not resolving to any name alphabase knows.
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
