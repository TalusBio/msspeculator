"""Static, checked-in catalog of PROSPECT files (name -> size/checksum/url per record).

Shipped as package data so listing a record never hits the Zenodo API — only the actual
file bytes are fetched (and those go through the cache). Regenerate when Zenodo changes:

    python -c "from pepdistill.data.prospect_catalog import build_catalog; build_catalog()"
"""

from __future__ import annotations

import json
from importlib.resources import files as _pkg_files

_CATALOG: dict | None = None
_CATALOG_PATH = "src/pepdistill/data/prospect_catalog.json"

_SHARDS: dict | None = None
_SHARDS_PATH = "src/pepdistill/data/prospect_shards.json"


def load_catalog() -> dict:
    """Parsed catalog: ``{"records": {name: {record_id, doi, files: {name: {...}}}}}``."""
    global _CATALOG
    if _CATALOG is None:
        raw = _pkg_files("pepdistill.data").joinpath("prospect_catalog.json").read_text()
        _CATALOG = json.loads(raw)
    return _CATALOG


def load_shard_index() -> dict:
    """Per-shard sizes inside every annotation zip: ``{record: {zip: [[name, packed, raw]]}}``.

    Lets a pool and a shard subset be chosen entirely offline. That choice is bounded by RAM,
    not by download size, because decoding materializes every shard and the merge holds a
    second copy — and the shards are far from uniform (third pool's run 90 MB to 388 MB), so
    "take N shards" says very little without these numbers.
    """
    global _SHARDS
    if _SHARDS is None:
        raw = _pkg_files("pepdistill.data").joinpath("prospect_shards.json").read_text()
        _SHARDS = json.loads(raw)
    return _SHARDS


def build_catalog(
    out_path: str | None = _CATALOG_PATH, records: dict[str, str] | None = None
) -> dict:
    """Query Zenodo and (re)write the catalog JSON. Network; run manually to refresh."""
    import urllib.request

    from .prospect import RECORDS

    recs = records or RECORDS
    cat: dict = {
        "_note": "PROSPECT file catalog (Zenodo). Regenerate: prospect_catalog.build_catalog().",
        "records": {},
    }
    for name, rid in recs.items():
        with urllib.request.urlopen(f"https://zenodo.org/api/records/{rid}", timeout=60) as r:
            meta = json.load(r)
        fdict = {}
        for f in meta.get("files", []):
            links = f.get("links", {})
            fdict[f["key"]] = {
                "size": f.get("size", 0),
                "checksum": f.get("checksum", ""),
                "url": links.get("content") or links.get("self", ""),
            }
        cat["records"][name] = {"record_id": rid, "doi": meta.get("doi", ""), "files": fdict}
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(cat, fh, indent=1, sort_keys=True)
    return cat


def build_shard_index(
    out_path: str | None = _SHARDS_PATH,
    *,
    delay_s: float = 2.0,
    max_attempts: int = 5,
    resume: bool = True,
) -> dict:
    """Enumerate every annotation zip's shards by RANGE-READING its central directory.

    Network, but tiny: a zip's central directory sits at the end of the file, so each archive
    costs a couple of range requests regardless of size. The whole PROSPECT collection is 79
    zips / ~243 GB and indexes without downloading any spectra.

    **Zenodo rate-limits this, and the failure is disguised.** A first run indexed 28 zips, got
    one explicit ``429 TOO MANY REQUESTS``, and then saw the remaining 50 surface as
    ``FileNotFoundError`` — fsspec reports a throttled response as a missing file, which reads
    as "this zip no longer exists on Zenodo" when it is very much still there. Hence
    ``delay_s`` between probes and retry-with-backoff on BOTH error shapes; a `FileNotFoundError`
    is only recorded as final after ``max_attempts``.

    ``resume=True`` keeps entries already indexed and re-probes only what is missing or errored,
    so an interrupted or throttled run can be continued instead of restarted.

    Run manually and commit the result, same as :func:`build_catalog`:

        python -c "from pepdistill.data.prospect_catalog import build_shard_index; build_shard_index()"
    """
    import time

    from .prospect import RECORDS, ProspectSource

    prior: dict = {}
    if resume:
        try:
            prior = load_shard_index().get("records", {})
        except FileNotFoundError:
            prior = {}

    cat = load_catalog()["records"]
    index: dict = {
        "_note": (
            "Per-shard sizes inside PROSPECT annotation zips, read from each zip's central "
            "directory (no downloads). Entries are [name, packed_bytes, raw_bytes]. "
            "Regenerate: prospect_catalog.build_shard_index(). Zenodo rate-limits; a throttled "
            "response surfaces as FileNotFoundError, so failures here may mean 'try again', "
            "not 'gone'."
        ),
        "records": {},
    }
    for record in RECORDS:
        src = ProspectSource(record)
        zips = sorted(k for k in cat[record]["files"] if k.endswith(".zip"))
        per_zip: dict = {}
        for name in zips:
            done = prior.get(record, {}).get(name)
            if isinstance(done, list):  # already indexed; an {"error": ...} entry is retried
                per_zip[name] = done
                continue
            for attempt in range(1, max_attempts + 1):
                try:
                    infos = src.annotation_shard_info(name)
                except Exception as exc:  # noqa: BLE001 - retry, then record; never hide the zip
                    if attempt == max_attempts:
                        per_zip[name] = {"error": f"{type(exc).__name__}: {exc}"}
                        print(f"  {record}/{name}: FAILED after {attempt} — {type(exc).__name__}")
                        break
                    backoff = delay_s * 2**attempt
                    print(f"  {record}/{name}: {type(exc).__name__}, retry in {backoff:.0f}s")
                    time.sleep(backoff)
                    continue
                per_zip[name] = [[i.name, i.packed_bytes, i.raw_bytes] for i in infos]
                print(
                    f"  {record}/{name}: {len(infos)} shards, "
                    f"{sum(i.raw_bytes for i in infos) / 1e9:.2f} GB raw"
                )
                break
            time.sleep(delay_s)
        index["records"][record] = per_zip
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(index, fh, indent=1, sort_keys=True)
    n_err = sum(1 for z in index["records"].values() for v in z.values() if isinstance(v, dict))
    if n_err:
        print(f"{n_err} zip(s) still unindexed — re-run to retry just those (resume=True).")
    return index
