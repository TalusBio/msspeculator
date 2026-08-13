"""Explicit object-store credentials for Polars.

Polars reads ``s3://`` through its own object store, which resolves credentials from the
environment and from instance metadata but **not** from the AWS SSO cache. The same
``scan_parquet`` call therefore works on a Batch worker with an instance role and fails on a
developer's laptop with a ``169.254.169.254`` timeout, even though ``aws s3 ls`` works there.

Rather than depend on that difference, every Polars read of a remote path passes credentials
resolved by botocore, which understands the SSO cache, profiles, environment variables and
instance roles alike. `ast-grep` enforces this (see `.ast-grep/rules/`), because the failure is
environmental: a bare call reviews cleanly, passes CI, and then only breaks for whoever runs it
somewhere else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Polars' object store understands these keys; botocore hands us exactly this triple.
_REGION = "us-west-2"


def is_remote(target: Any) -> bool:
    """Whether a Polars read target needs object-store credentials."""
    if isinstance(target, (list, tuple)):
        return any(is_remote(item) for item in target)
    if isinstance(target, Path):
        return False
    return isinstance(target, str) and "://" in target


def parquet_storage_options(target: Any) -> dict[str, str]:
    """Credentials for a Polars read of ``target``, or ``{}`` for a local path.

    Resolved per call rather than cached: a frozen snapshot would not pick up a refreshed
    instance-role or renewed SSO session, and these reads span whole preparation jobs.
    """
    if not is_remote(target):
        return {}
    import botocore.session

    credentials = botocore.session.get_session().get_credentials()
    if credentials is None:
        # No credentials anywhere. Returning {} leaves Polars to its own resolution, so a public
        # bucket still reads and a private one fails with the object store's own message rather
        # than an error invented here.
        return {}
    frozen = credentials.get_frozen_credentials()
    options = {
        "aws_access_key_id": frozen.access_key,
        "aws_secret_access_key": frozen.secret_key,
        "aws_region": _REGION,
    }
    if frozen.token:
        options["aws_session_token"] = frozen.token
    return options


__all__ = ["is_remote", "parquet_storage_options"]
