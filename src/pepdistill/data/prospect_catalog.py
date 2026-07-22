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


def load_catalog() -> dict:
    """Parsed catalog: ``{"records": {name: {record_id, doi, files: {name: {...}}}}}``."""
    global _CATALOG
    if _CATALOG is None:
        raw = _pkg_files("pepdistill.data").joinpath("prospect_catalog.json").read_text()
        _CATALOG = json.loads(raw)
    return _CATALOG


def build_catalog(out_path: str | None = _CATALOG_PATH, records: dict[str, str] | None = None) -> dict:
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
