import io

import pytest

from pepdistill.data.config import DigestConfig, SplitConfig
from pepdistill.data.digest import cleave_protein, digest_records, resolve_fasta
from pepdistill.data.precursors import (
    enumerate_precursors,
    frame_to_precursors,
    precursors_to_frame,
)
from pepdistill.data.split import assign_split


def test_trypsin_cleaves_after_kr_not_before_p():
    cfg = DigestConfig(missed_cleavages=0, min_length=1, max_length=100)
    peps = list(cleave_protein("SAMPKAER", cfg))
    # cut after K (not before P? next is A, so it cuts) and after R.
    assert "SAMPK" in peps
    assert "AER" in peps


def test_uniprot_fasta_is_downloaded_once_to_cache(tmp_path, monkeypatch):
    calls: list[str] = []

    def open_url(url, timeout):
        calls.append(url)
        assert timeout == 120
        return io.BytesIO(b">protein\nSAMPLERK\n")

    monkeypatch.setattr("pepdistill.data.digest.urllib.request.urlopen", open_url)
    logs: list[str] = []
    first = resolve_fasta("uniprot:UP000000625", cache_dir=tmp_path, log=logs.append)
    second = resolve_fasta("uniprot:UP000000625", cache_dir=tmp_path, log=logs.append)

    assert first == second == tmp_path / "fasta" / "UP000000625.fasta"
    assert first.read_bytes() == b">protein\nSAMPLERK\n"
    assert len(calls) == 1
    assert "downloading UP000000625" in logs[0]
    assert "using cached UP000000625" in logs[-1]


def test_uniprot_fasta_reference_is_validated(tmp_path):
    with pytest.raises(ValueError, match="invalid UniProt proteome reference"):
        resolve_fasta("uniprot:not-an-accession", cache_dir=tmp_path)


def test_trypsin_skips_kp_bond():
    cfg = DigestConfig(missed_cleavages=0, min_length=1, max_length=100)
    peps = list(cleave_protein("AAKPAAR", cfg))
    # K followed by P -> no cut, so whole thing is one peptide.
    assert peps == ["AAKPAAR"]


def test_missed_cleavages():
    cfg = DigestConfig(missed_cleavages=1, min_length=1, max_length=100)
    peps = set(cleave_protein("AAKBBKCCR".replace("B", "A"), cfg))
    assert any(p.count("K") + p.count("R") == 2 for p in peps)


def test_length_filter():
    cfg = DigestConfig(missed_cleavages=0, min_length=7, max_length=30)
    peps = list(cleave_protein("AAKSAMPLERKMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMR", cfg))
    assert all(7 <= len(p) <= 30 for p in peps)
    assert "AAK" not in peps  # too short


def test_split_is_deterministic_and_partitions():
    cfg = SplitConfig()
    assert assign_split("SAMPLER", cfg) == assign_split("SAMPLER", cfg)
    seqs = [f"PEPTIDE{i}K" for i in range(2000)]
    buckets = [assign_split(s, cfg) for s in seqs]
    assert set(buckets) == {"train", "val", "test"}
    # Roughly the requested 80/10/10.
    frac_train = buckets.count("train") / len(buckets)
    assert 0.72 < frac_train < 0.88


def test_split_ignores_mods_and_charge():
    # Split keys on the bare sequence, so it does not depend on charge/mods.
    cfg = SplitConfig()
    assert assign_split("ACDEMK", cfg) == assign_split("ACDEMK", cfg)


def test_bad_split_fractions():
    with pytest.raises(ValueError):
        SplitConfig(train=0.5, val=0.4, test=0.4)


def test_enumerate_applies_fixed_and_variable_mods():
    dcfg = DigestConfig(min_charge=2, max_charge=2, max_variable_mods=1)
    scfg = SplitConfig()
    precs = enumerate_precursors(["ACDEMK"], dcfg, scfg)
    # One Cys (fixed CAM always), one Met (0 or 1 oxidation) -> 2 mod-forms, 1 charge.
    assert len(precs) == 2
    for p in precs:
        assert any(name == "Carbamidomethyl@C" for _, name in p.peptide.mods)
    ox_counts = {sum(1 for _, n in p.peptide.mods if n == "Oxidation@M") for p in precs}
    assert ox_counts == {0, 1}


def test_precursor_frame_roundtrip():
    dcfg = DigestConfig(min_charge=2, max_charge=3)
    scfg = SplitConfig()
    precs = enumerate_precursors(["ACDEMKSAMPLER"], dcfg, scfg)
    frame = precursors_to_frame(precs)
    back = frame_to_precursors(frame)
    assert [p.peptide for p in back] == [p.peptide for p in precs]
    assert [p.charge for p in back] == [p.charge for p in precs]


def test_digest_records_dedupes_and_sorts():
    recs = [("p1", "SAMPLERKAAAAAAAK"), ("p2", "SAMPLERKAAAAAAAK")]
    cfg = DigestConfig(missed_cleavages=0)
    peps = digest_records(recs, cfg)
    assert peps == sorted(set(peps))


def test_precursor_frame_roundtrips_terminal_and_mass_only_mods():
    from pepdistill.chem import Peptide
    from pepdistill.data.precursors import (
        Precursor,
        frame_to_precursors,
        precursors_to_frame,
    )

    precs = [
        Precursor(Peptide("ETTLHLVLR", (("n", "TMT6plex"), (1, "Phospho"))), 2, "train"),
        Precursor(Peptide("PEPTIDE", ((2, 42.010565),)), 3, "train"),
        Precursor(Peptide("PEK", (("c", "Phospho"),)), 2, "train"),
    ]
    back = frame_to_precursors(precursors_to_frame(precs))
    assert back == precs


def test_nterm_and_side_chain_mods_are_distinct_sites():
    from pepdistill.chem import Peptide

    p = Peptide("KPEPTIDE", (("n", "TMT6plex"), (0, "TMT6plex")))
    assert len(p.mods) == 2
    bare = Peptide("KPEPTIDE").mono_mass()
    assert abs(p.mono_mass() - bare - 2 * 229.1629321) < 1e-4


def test_unspecific_digest_refuses_an_unreachable_length_window():
    """An empty digest is a config error, not a result.

    `unspecific` cuts after every residue, so a peptide spans at most missed_cleavages + 1.
    Against the default min_length=7 that silently yielded 0 peptides from a whole proteome.
    """
    import pytest

    from pepdistill.data.config import DigestConfig
    from pepdistill.data.digest import digest_records

    records = [("p1", "MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFK")]
    with pytest.raises(ValueError, match="at most"):
        digest_records(records, DigestConfig(enzyme="unspecific"))  # missed=2, min_length=7

    # Raising missed_cleavages so the span can reach min_length makes it work.
    peps = digest_records(
        records, DigestConfig(enzyme="unspecific", missed_cleavages=10, min_length=8, max_length=11)
    )
    assert peps and all(8 <= len(p) <= 11 for p in peps)


def test_digest_refuses_an_empty_result_from_real_proteins():
    """A specific enzyme with an impossible window must also fail loudly, not return []."""
    import pytest

    from pepdistill.data.config import DigestConfig
    from pepdistill.data.digest import digest_records

    records = [("p1", "MKWVTFISLLFLFSSAYSRGVFRR")]
    with pytest.raises(ValueError, match="0 peptides"):
        digest_records(records, DigestConfig(min_length=200, max_length=300))
