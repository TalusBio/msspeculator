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

from pepdistill.chem import Peptide
from pepdistill.data.cache import FileCache
from pepdistill.data.meta_index import MetaIndex, SpectrumMeta
from pepdistill.data.prospect import (
    RECORDS,
    ProspectSchema,
    ProspectSource,
    decode_fragments,
    fragment_filter_mask,
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


def test_parse_modseq_raises_on_a_resolvable_but_unencodable_accession():
    """Iodo (UNIMOD:129) resolves to a name and then fails to project onto the six-element
    basis. Catching that at parse time keeps it out of collate, where it would abort a
    multi-hour training run mid-epoch instead of rejecting the offending row at ingest."""
    import pepdistill_rs as _rs

    assert _rs.unimod_name(129) == "Iodo", "fixture assumes 129 resolves"
    with pytest.raises(ValueError) as e:
        parse_modseq("PEPC[UNIMOD:129]TIDER")
    msg = str(e.value)
    assert "129" in msg and "Iodo" in msg, msg
    assert '"I"' in msg or "'I'" in msg, f"must name the offending element: {msg}"


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


def test_to_labels_tolerates_a_present_but_null_andromeda_score(tmp_path):
    """A None in an object-dtype andromeda_score column must fall back to NaN, not crash.

    build_meta_index already guards this (`is not None`); to_labels must match it. The column
    being wholly ABSENT was already handled by getattr's default -- this covers PRESENT-but-null,
    which float(getattr(...)) alone does not."""
    src = ProspectSource("prospect", cache=FileCache([str(tmp_path)]))
    meta, ann = _meta_ann()
    meta["andromeda_score"] = pd.Series([None, 50.0, 60.0], dtype=object)
    out = src.to_labels(meta, ann)
    assert set(p.peptide.sequence for p in out.precursors) == {"PEPTIDEK", "SPEPK", "ACDEK"}


def test_to_labels_raises_on_unresolvable_accession(tmp_path):
    """The raise must propagate all the way out of to_labels, not just out of parse_modseq in
    isolation - there is no silent-drop-on-unknown-mod fallback left anywhere in ingest.
    9999 is confirmed absent from the vendored table (max real accession is in the low
    thousands)."""
    meta = pd.DataFrame(
        {
            "raw_file": ["rfZ"],
            "scan_number": [1],
            "modified_sequence": ["PEP[UNIMOD:9999]TIDEK"],
            "precursor_charge": [2],
            "indexed_retention_time": [40.0],
            "aligned_collision_energy": [0.30],
            "mass_analyzer": ["FTMS"],
            "fragmentation": ["HCD"],
        }
    )
    ann = pd.DataFrame(
        [("rfZ", 1, "b", 1, 1, 0.5, "")],
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
    src = ProspectSource("prospect", cache=FileCache([str(tmp_path)]))
    with pytest.raises(ValueError, match="9999"):
        src.to_labels(meta, ann)


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


def test_every_prospect_accession_is_encodable():
    """Accessions observed in the PROSPECT catalog (see tools/sweep_prospect_mods.py).

    Frozen from a FULL sweep (``--records all``, all five ``RECORDS`` keys: prospect, tmt,
    multi_ptm, tmt_ptm, test_ptm) run on 2026-07-28. Every accession resolved and had a valid
    6-element composition -- nothing in the real data trips the frozen [C,H,N,O,S,P] basis.

    This is offline and does not re-run the sweep: it asserts against the frozen list only, so
    it runs in CI without network. A partial (default-args) sweep run later may legitimately
    observe FEWER accessions than this list -- that is expected under-reporting, not evidence
    this list is stale. Only a full ``--records all`` sweep is authoritative for changing it.

    Note: accession 1848 resolves to "Gluratylation" -- that is UNIMOD's own spelling of what
    is normally written "Glutarylation" in the vendored table's title, not a typo introduced
    here. Do not "fix" the spelling; it would break the name-based lookup.
    """
    import pepdistill_rs as _rs

    observed = [
        737,  # TMT6plex
        35,  # Oxidation@M
        4,  # Carbamidomethyl@C
        21,  # Phospho
        121,  # GG
        1,  # Acetyl
        34,  # Methyl
        28,  # Gln->pyro-Glu
        43,  # HexNAc
        7,  # Deamidated
        27,  # Glu->pyro-Glu
        36,  # Dimethyl
        58,  # Propionyl
        1289,  # Butyryl
        747,  # Malonyl
        1363,  # Crotonyl
        1848,  # Gluratylation (UNIMOD's own spelling; not our typo)
        64,  # Succinyl
        122,  # Formyl
        1849,  # hydroxyisobutyryl
        354,  # Nitro
        37,  # Trimethyl
    ]
    for acc in observed:
        name = _rs.unimod_name(acc)
        assert name is not None, f"accession {acc} no longer resolves"
        _rs.mod_element_comp(name)  # raises if outside the 6-element basis


def test_annotation_shard_info_reports_names_and_sizes(tmp_path, monkeypatch):
    """Shard sizes come from the zip central directory, so a pool can be budgeted first.

    Shard count alone is misleading: third pool's six shards run 90 MB to 388 MB, so picking
    "all of them" is a 16x jump over the smallest, not 6x. Every selected shard is extracted to
    local parquet and re-read each epoch, so this is the number that bounds a run.
    """
    import zipfile

    from pepdistill.data.prospect import ProspectSource

    zpath = tmp_path / "pool.zip"
    payload_a, payload_b = b"a" * 5000, b"b" * 900
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        z.writestr("pool/one_annotation.parquet", payload_a)
        z.writestr("pool/two_annotation.parquet", payload_b)
        z.writestr("pool/readme.txt", b"not a shard")

    src = ProspectSource("prospect")
    monkeypatch.setattr(src, "_open_remote_zip", lambda _name: zipfile.ZipFile(zpath))

    infos = src.annotation_shard_info("pool.zip")
    assert [i.short_name for i in infos] == ["one_annotation.parquet", "two_annotation.parquet"]
    assert [i.raw_bytes for i in infos] == [len(payload_a), len(payload_b)]
    assert all(i.packed_bytes > 0 for i in infos)

    # The name-only view must stay consistent with it.
    assert src.annotation_shards("pool.zip") == [i.name for i in infos]


def _index_for(peptide, charge=2, irt=50.0, raw_rt=5.0, split="train"):
    idx = MetaIndex()
    idx.by_key[("RUN_A", 7)] = SpectrumMeta(
        peptide, charge, irt, raw_rt, split, "FTMS", "HCD", 28.0, 100.0
    )
    return idx


def _mask_frame(neutral_loss, dtype=None):
    n = len(neutral_loss)
    col = pd.Series(neutral_loss, dtype=dtype) if dtype else pd.Series(neutral_loss)
    return pd.DataFrame(
        {
            "ion_type": ["b"] * n,
            "charge": [1] * n,
            "neutral_loss": col,
        }
    )


def test_fragment_filter_mask_treats_object_nan_and_empty_string_as_no_loss():
    ann = _mask_frame([None, "", "H2O"])
    assert list(fragment_filter_mask(ann, ProspectSchema())) == [True, True, False]


def test_fragment_filter_mask_handles_a_category_column_with_no_empty_string_category():
    """The streaming reader hands this ``category`` dtype (dictionary-encoded on disk). A
    shard whose fragments ALL carry a neutral loss has no ``""`` category at all, and the old
    ``fillna("") == ""`` raised ``TypeError`` in that case -- this pins that it no longer does,
    and that the answer matches the object-dtype case for the same values.
    """
    ann = _mask_frame(
        pd.Categorical([None, "H2O", "NH3"], categories=["H2O", "NH3"])
    )
    assert list(fragment_filter_mask(ann, ProspectSchema())) == [True, False, False]


def test_fragment_filter_mask_agrees_between_object_and_category_dtype():
    values = [None, "", "H2O", "NH3", ""]
    obj = fragment_filter_mask(_mask_frame(values), ProspectSchema())
    cat = fragment_filter_mask(
        _mask_frame(pd.Categorical(values, categories=["H2O", "NH3", ""])), ProspectSchema()
    )
    assert list(obj) == list(cat) == [True, True, False, False, True]


def test_decode_fragments_scatters_b_and_y_into_the_right_cells():
    idx = _index_for(Peptide("PEPK", ()))
    frag = pd.DataFrame(
        {
            "raw_file": ["RUN_A"] * 3,
            "scan_number": [7, 7, 7],
            "ion_type": ["b", "y", "b"],
            "no": [1, 1, 2],
            "charge": [1, 1, 2],
            "intensity": [0.5, 0.25, 0.75],
        }
    )
    real, _ = decode_fragments(idx, frag, ProspectSchema())
    assert len(real.precursors) == 1
    ms2 = real.labels[0].ms2
    assert ms2.shape == (3, 4)          # residues - 1, len(ION_TYPES)
    assert ms2[0, 0] == 0.5             # b1 -> site 0, col 0
    assert ms2[2, 1] == 0.25            # y1 -> site n-1-ord = 2, col 1
    assert ms2[1, 2] == 0.75            # b2 z=2 -> site 1, col 0 + 2*(2-1)
    assert real.raw_rt == [5.0] and real.source_ids == ["RUN_A"]
    assert real.labels[0].rt == 50.0 and math.isnan(real.labels[0].ccs)


def test_decode_fragments_drops_spectra_with_no_surviving_cells():
    idx = _index_for(Peptide("PEPK", ()))
    # ordinal 9 on a 4-mer: site is out of range, so nothing lands and the spectrum is dropped.
    frag = pd.DataFrame(
        {
            "raw_file": ["RUN_A"], "scan_number": [7], "ion_type": ["b"],
            "no": [9], "charge": [1], "intensity": [1.0],
        }
    )
    real, _ = decode_fragments(idx, frag, ProspectSchema())
    assert real.precursors == []


def test_decode_fragments_max_collapses_duplicate_cells():
    idx = _index_for(Peptide("PEPK", ()))
    frag = pd.DataFrame(
        {
            "raw_file": ["RUN_A"] * 3, "scan_number": [7, 7, 7], "ion_type": ["b", "b", "b"],
            "no": [1, 1, 1], "charge": [1, 1, 1], "intensity": [0.4, 0.9, 0.2],
        }
    )
    real, _ = decode_fragments(idx, frag, ProspectSchema())
    # Winner (0.9) is interior and differs from first (0.4), last (0.2) and sum (1.5), so
    # max is the only aggregation that passes.
    assert real.labels[0].ms2[0, 0] == pytest.approx(0.9)


def test_decode_fragments_ignores_scans_absent_from_the_index():
    idx = _index_for(Peptide("PEPK", ()))
    frag = pd.DataFrame(
        {
            "raw_file": ["RUN_A"], "scan_number": [999], "ion_type": ["b"],
            "no": [1], "charge": [1], "intensity": [1.0],
        }
    )
    assert decode_fragments(idx, frag, ProspectSchema())[0].precursors == []


def test_decode_fragments_rejects_a_non_by_ion_type():
    idx = _index_for(Peptide("PEPK", ()))
    frag = pd.DataFrame(
        {
            "raw_file": ["RUN_A"], "scan_number": [7], "ion_type": ["a"],
            "no": [1], "charge": [1], "intensity": [1.0],
        }
    )
    with pytest.raises(ValueError, match="ion_type"):
        decode_fragments(idx, frag, ProspectSchema())


def test_decode_fragments_rejects_an_out_of_range_fragment_charge():
    idx = _index_for(Peptide("PEPK", ()))
    frag = pd.DataFrame(
        {
            "raw_file": ["RUN_A"], "scan_number": [7], "ion_type": ["b"],
            "no": [1], "charge": [3], "intensity": [1.0],
        }
    )
    with pytest.raises(ValueError, match="charge"):
        decode_fragments(idx, frag, ProspectSchema())


def test_decode_fragments_reports_a_nan_fragment_charge_through_the_named_error():
    """A NaN in the fragment-charge column must produce decode_fragments' OWN message.

    The per-row `int(x)` this validation used to do raised a bare "cannot convert float NaN to
    integer" from the Python builtin, which names neither the column, the caller's mistake, nor
    the fix. That is exactly the silent-ish failure the named errors exist to replace."""
    idx = _index_for(Peptide("PEPK", ()))
    frag = pd.DataFrame(
        {
            "raw_file": ["RUN_A"], "scan_number": [7], "ion_type": ["b"],
            "no": [1], "charge": [float("nan")], "intensity": [1.0],
        }
    )
    with pytest.raises(ValueError, match=r"requires fragment charge in \{1, 2\} only") as exc:
        decode_fragments(idx, frag, ProspectSchema())
    assert "fragment_filter_mask" in str(exc.value)
    assert "cannot convert float NaN" not in str(exc.value)


def test_decode_fragments_rejects_a_non_integral_scan_number():
    idx = _index_for(Peptide("PEPK", ()))
    frag = pd.DataFrame(
        {
            "raw_file": ["RUN_A"], "scan_number": [7.5], "ion_type": ["b"],
            "no": [1], "charge": [1], "intensity": [1.0],
        }
    )
    with pytest.raises(ValueError, match="scan_number"):
        decode_fragments(idx, frag, ProspectSchema())


def test_decode_fragments_rejects_a_nan_scan_number():
    idx = _index_for(Peptide("PEPK", ()))
    frag = pd.DataFrame(
        {
            "raw_file": ["RUN_A"], "scan_number": [float("nan")], "ion_type": ["b"],
            "no": [1], "charge": [1], "intensity": [1.0],
        }
    )
    with pytest.raises(ValueError, match="scan_number"):
        decode_fragments(idx, frag, ProspectSchema())


def test_decode_fragments_out_keys_track_emission_order_across_raw_files():
    """Two raw files x two scans, discovered interleaved (not sorted, not reversed) -- the
    order a naive reimplementation (sorted by key, or built from index.by_key) would get wrong.
    Downstream code zips out_keys against precursors/labels/raw_rt to attach that spectrum's
    own acquisition factors, so a mis-order there would silently mis-assign metadata."""
    idx = MetaIndex()
    idx.by_key[("RUN_A", 7)] = SpectrumMeta(
        Peptide("PEPK", ()), 2, 50.0, 5.0, "train", "FTMS", "HCD", 28.0, 100.0
    )
    idx.by_key[("RUN_A", 9)] = SpectrumMeta(
        Peptide("PEPK", ()), 2, 55.0, 5.5, "train", "FTMS", "HCD", 28.0, 100.0
    )
    idx.by_key[("RUN_B", 3)] = SpectrumMeta(
        Peptide("PEPK", ()), 2, 60.0, 6.0, "train", "FTMS", "HCD", 28.0, 100.0
    )
    idx.by_key[("RUN_B", 11)] = SpectrumMeta(
        Peptide("PEPK", ()), 2, 65.0, 6.5, "train", "FTMS", "HCD", 28.0, 100.0
    )
    # Discovery order: RUN_B/3, RUN_A/7, RUN_B/11, RUN_A/9 -- neither sorted nor reversed.
    frag = pd.DataFrame(
        {
            "raw_file": ["RUN_B", "RUN_A", "RUN_B", "RUN_A"],
            "scan_number": [3, 7, 11, 9],
            "ion_type": ["b", "b", "b", "b"],
            "no": [1, 1, 1, 1],
            "charge": [1, 1, 1, 1],
            "intensity": [0.5, 0.6, 0.7, 0.8],
        }
    )
    real, out_keys = decode_fragments(idx, frag, ProspectSchema())
    assert out_keys == [("RUN_B", 3), ("RUN_A", 7), ("RUN_B", 11), ("RUN_A", 9)]
    assert [k[0] for k in out_keys] == real.source_ids
