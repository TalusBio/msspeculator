"""PROSPECT source, offline: catalog listing + read + schema validation + acquisition factors.

Listing comes from the checked-in catalog (no network). A synthetic parquet is seeded into
the local cache under a REAL catalog filename, so read() hits the cache and never downloads.
"""

import io
import math
import os
import zipfile

import pandas as pd
import pytest

from pepdistill.data.cache import FileCache
from pepdistill.data.prospect import (
    RECORDS,
    ProspectSchema,
    ProspectSource,
    parse_modseq,
)
from pepdistill.data.prospect_catalog import load_catalog

REAL_FILE = "TUM_third_pool_meta_data.parquet"  # exists in the 'prospect' record catalog


def _seed(tmp_path):
    rid = RECORDS["prospect"]
    root = tmp_path / "local" / "zenodo" / rid
    os.makedirs(root, exist_ok=True)
    df = pd.DataFrame(
        {
            "modified_sequence": ["ABC", "DEFK"],
            "precursor_charge": [2, 3],
            "aligned_collision_energy": [0.28, 0.30],
            "mass_analyzer": ["ITMS", "ITMS"],
            "fragmentation": ["CID", "CID"],
            "retention_time": [12.1, 20.4],
        }
    )
    df.to_parquet(root / REAL_FILE)
    return FileCache([str(tmp_path / "local")], write_through=False)


def test_catalog_loads():
    cat = load_catalog()["records"]
    assert set(cat) == set(RECORDS)
    entry = cat["prospect"]["files"][REAL_FILE]
    assert entry["size"] > 0 and entry["url"].endswith("/content")


def test_files_listed_from_catalog_offline(tmp_path):
    # No cache seeding, no network: listing is pure catalog.
    src = ProspectSource("prospect", cache=FileCache([str(tmp_path)]))
    files = src.files()
    assert REAL_FILE in files and all(f.endswith(".parquet") for f in files)


def test_read_and_acquisition_factors(tmp_path):
    src = ProspectSource("prospect", cache=_seed(tmp_path))
    df = src.read(REAL_FILE)
    assert len(df) == 2
    acq = src.acquisition_key(df)
    assert list(acq["mass_analyzer"]) == ["ITMS", "ITMS"]
    assert "collision_energy" in acq.columns and "fragmentation" in acq.columns


def test_schema_mismatch_raises(tmp_path):
    src = ProspectSource("prospect", cache=_seed(tmp_path), schema=ProspectSchema(charge="nope"))
    with pytest.raises(ValueError, match="missing required columns"):
        src.read(REAL_FILE)


def test_unknown_record():
    with pytest.raises(ValueError, match="unknown PROSPECT record"):
        ProspectSource("not_a_record")


def test_parse_modseq():
    assert parse_modseq("PEPTIDE") == ("PEPTIDE", ())
    assert parse_modseq("[UNIMOD:737]ET[UNIMOD:21]TLHLVLR") == (
        "ETTLHLVLR",
        (("n", "TMT6plex"), (1, "Phospho")),
    )


def test_parse_modseq_resolves_accessions_without_a_hand_map():
    from pepdistill.data.prospect import parse_modseq

    seq, mods = parse_modseq("[UNIMOD:737]ET[UNIMOD:21]TLHLVLR")
    assert seq == "ETTLHLVLR"
    assert mods == (("n", "TMT6plex"), (1, "Phospho"))


def test_parse_modseq_resolves_a_mod_we_never_hand_mapped():
    """Acetyl (UNIMOD:1) was never in the old _UNIMOD_TO_NAME table."""
    from pepdistill.data.prospect import parse_modseq

    seq, mods = parse_modseq("[UNIMOD:1]PEPTIDE")
    assert seq == "PEPTIDE"
    assert mods == (("n", "Acetyl"),)


def test_parse_modseq_raises_on_unresolvable_accession():
    with pytest.raises(ValueError, match="99999999"):
        parse_modseq("PEP[UNIMOD:99999999]TIDE")


def test_unimod_sentinel_map_is_gone():
    import pepdistill.data.prospect as p

    assert not hasattr(p, "_UNIMOD_TO_NAME")


def _meta_ann():
    meta = pd.DataFrame(
        {
            "raw_file": ["rfA", "rfB", "rfC"],
            "scan_number": [1, 2, 3],
            "modified_sequence": ["PEPTIDEK", "[UNIMOD:21]SPEPK", "A[UNIMOD:1]CDEK"],
            "precursor_charge": [2, 2, 3],
            "indexed_retention_time": [50.0, 30.0, 40.0],
            "aligned_collision_energy": [0.30, 0.28, 0.31],
            "mass_analyzer": ["FTMS", "FTMS", "ITMS"],
            "fragmentation": ["HCD", "HCD", "CID"],
        }
    )
    rows = [
        # rfA / PEPTIDEK (n=8): kept fragments
        ("rfA", 1, "b", 1, 1, 0.5, ""),  # site 0, col (b,1)=0
        ("rfA", 1, "y", 1, 1, 0.9, ""),  # site n-1-1=6, col (y,1)=1
        ("rfA", 1, "b", 2, 2, 0.3, ""),  # site 1, col (b,2)=2
        # filtered: precursor ion, frag charge 3, neutral loss
        ("rfA", 1, "precursor", 0, 1, 1.0, ""),
        ("rfA", 1, "b", 3, 3, 0.7, ""),
        ("rfA", 1, "y", 2, 1, 0.7, "H2O"),
        # rfB / SPEPK (n=5), phospho on S: one fragment so it's non-empty
        ("rfB", 2, "b", 2, 1, 0.4, ""),  # site 1, col 0
        # rfC / ACDEK (n=5), Acetyl on residue 0: never in the old hand-map, but resolves
        # via the vendored table now -> kept, not dropped.
        ("rfC", 3, "b", 1, 1, 0.6, ""),  # site 0, col 0
    ]
    ann = pd.DataFrame(
        rows,
        columns=[
            "raw_file",
            "scan_number",
            "ion_type",
            "no",
            "charge",
            "intensity",
            "neutral_loss",
        ],
    )
    return meta, ann


def test_to_labels_decoding(tmp_path):
    src = ProspectSource("prospect", cache=FileCache([str(tmp_path)]))
    meta, ann = _meta_ann()
    out = src.to_labels(meta, ann)

    by_seq = {p.peptide.sequence: (p, lab) for p, lab in zip(out.precursors, out.labels)}
    # rfC's Acetyl (UNIMOD:1) was never in the old hand-map but resolves via the vendored
    # table now, so it's kept rather than silently dropped.
    assert set(by_seq) == {"PEPTIDEK", "SPEPK", "ACDEK"}

    _p, lab = by_seq["PEPTIDEK"]
    assert lab.ms2.shape == (7, 4)
    assert lab.ms2[0, 0] == pytest.approx(0.5)  # b1 z1
    assert lab.ms2[6, 1] == pytest.approx(0.9)  # y1 z1 -> site n-1-1
    assert lab.ms2[1, 2] == pytest.approx(0.3)  # b2 z2
    assert lab.ms2.sum() == pytest.approx(0.5 + 0.9 + 0.3)  # filtered rows excluded
    assert lab.rt == pytest.approx(50.0)
    assert math.isnan(lab.ccs)  # PROSPECT has no ion mobility

    ps, _ = by_seq["SPEPK"]
    # Leading token (before any residue) routes to the N-term site, not residue 0.
    assert ps.peptide.mods == [("n", "Phospho")]

    pc, labc = by_seq["ACDEK"]
    assert pc.peptide.mods == [(0, "Acetyl")]
    assert labc.ms2.shape == (4, 4)
    assert labc.ms2[0, 0] == pytest.approx(0.6)  # b1 z1

    # raw_file is the context stratification key; per-run acquisition captured.
    assert set(out.source_ids) == {"rfA", "rfB", "rfC"}
    assert out.acquisition["rfA"]["mass_analyzer"] == "FTMS"
    assert out.acquisition["rfA"]["collision_energy"] == pytest.approx(0.30)


def test_read_annotation_from_zip(tmp_path):
    # Seed a synthetic annotation zip at the real catalog key for a test_ptm zip.
    rid = RECORDS["test_ptm"]
    root = tmp_path / "local" / "zenodo" / rid
    os.makedirs(root, exist_ok=True)
    _, ann = _meta_ann()
    buf = io.BytesIO()
    ann.to_parquet(buf)
    with zipfile.ZipFile(root / "TMT_TUM_perm_pT.zip", "w") as z:
        z.writestr("TMT_TUM_perm_pT/shard_annotation.parquet", buf.getvalue())

    src = ProspectSource(
        "test_ptm", cache=FileCache([str(tmp_path / "local")], write_through=False)
    )
    df = src.read_annotation("TMT_TUM_perm_pT.zip")
    assert len(df) == len(ann) and "ion_type" in df.columns


def test_streaming_reads_cached_zip_by_member(tmp_path):
    # A cached zip is opened locally (no network); streaming member-selection still applies.
    rid = RECORDS["test_ptm"]
    root = tmp_path / "local" / "zenodo" / rid
    os.makedirs(root, exist_ok=True)
    _, ann = _meta_ann()
    buf = io.BytesIO()
    ann.to_parquet(buf)
    with zipfile.ZipFile(root / "TMT_TUM_perm_pT.zip", "w") as z:
        z.writestr("pool/a_annotation.parquet", buf.getvalue())
        z.writestr("pool/b_annotation.parquet", buf.getvalue())

    src = ProspectSource(
        "test_ptm", cache=FileCache([str(tmp_path / "local")], write_through=False)
    )
    assert src.annotation_shards("TMT_TUM_perm_pT.zip") == [
        "pool/a_annotation.parquet",
        "pool/b_annotation.parquet",
    ]
    one = src.read_annotation_streaming("TMT_TUM_perm_pT.zip", max_members=1)
    assert len(one) == len(ann)  # only the first shard read
    both = src.read_annotation_streaming(
        "TMT_TUM_perm_pT.zip", members=["pool/b_annotation.parquet"]
    )
    assert len(both) == len(ann)
