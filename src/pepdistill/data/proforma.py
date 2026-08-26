"""Polars expressions over a canonical ProForma column.

Prepared shards store one canonical ProForma string per spectrum, so anything that used to read a
separate stripped ``sequence`` column now derives it. That derivation is defined once, here.

These are string operations standing in for the real thing. Counting ``[A-Z]`` is wrong; the
literal ``UNIMOD`` inside every modification token is uppercase; so the tokens have to be removed
before the residues can be counted, and doing that with an inline regex at each call site is how
the same subtly-wrong expression ends up copied around. The authority on this grammar is our own
parser in ``rust/core/src/proforma.rs``; when it is exposed as a Polars plugin these become calls
to it, and this module is the only thing that changes.
"""

from __future__ import annotations

import polars as pl

#: A modification token, or a terminal separator. What remains is the bare residues.
#: Kept identical to what the Rust grammar accepts: `[UNIMOD:<digits>]` and `-`.
MOD_TOKEN_PATTERN = r"\[UNIMOD:\d+\]|-"


def stripped_sequence(column: str = "proforma") -> pl.Expr:
    """The bare residues of a canonical ProForma column."""
    return pl.col(column).str.replace_all(MOD_TOKEN_PATTERN, "")


def stripped_length(column: str = "proforma") -> pl.Expr:
    """Residue count of a canonical ProForma column, ignoring modification tokens."""
    return stripped_sequence(column).str.len_chars()


__all__ = ["MOD_TOKEN_PATTERN", "stripped_sequence", "stripped_length"]
