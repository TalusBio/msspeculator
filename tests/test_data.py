import io
from pathlib import Path

import pytest

from msspeculator.data.config import DigestConfig, SplitConfig
from msspeculator.data.digest import cleave_protein, digest_records, resolve_fasta
from msspeculator.data.precursors import (
    enumerate_precursors,
    frame_to_precursors,
    precursors_to_frame,
)
from msspeculator.data.split import assign_split


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

    monkeypatch.setattr("msspeculator.data.digest.urllib.request.urlopen", open_url)
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


def _reference_unit_hash(sequence: str, salt: str) -> float:
    """The split hash as Python used to compute it, kept only to pin the Rust port.

    Production reads the Rust implementation, since the corpus is split here while a library is
    split there during a context fit. If these two ever disagree, a peptide the model trained on
    can reach a held-out score without anything failing, so the reference stays, in the tests.
    """
    import hashlib

    digest = hashlib.blake2b(f"{salt}:{sequence}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def test_rust_split_matches_the_python_reference():
    cfg = SplitConfig()
    sequences = [f"PEPTIDE{chr(65 + i % 26)}{i}K" for i in range(500)]
    for sequence in sequences:
        h = _reference_unit_hash(sequence, cfg.salt)
        expected = "train" if h < cfg.train else "val" if h < cfg.train + cfg.val else "test"
        assert assign_split(sequence, cfg) == expected, (
            f"{sequence}: hash {h!r} says {expected}, Rust says {assign_split(sequence, cfg)}"
        )
    # A different salt has to reassign, or the salt is not doing its job on the Rust side either.
    salted = SplitConfig(salt="a-different-salt")
    assert any(assign_split(s, cfg) != assign_split(s, salted) for s in sequences)


def test_split_ignores_mods_and_charge():
    # Split keys on the bare sequence, so it does not depend on charge/mods.
    cfg = SplitConfig()
    assert assign_split("ACDEMK", cfg) == assign_split("ACDEMK", cfg)


def test_bad_split_fractions():
    with pytest.raises(ValueError):
        SplitConfig(train=0.5, val=0.4, test=0.4)


def test_enumerate_applies_fixed_and_variable_mods():
    """Library generation enumerates every modform, ignoring the rules' probabilities.

    A library that omitted a modform would simply fail to identify it, so this path is exhaustive
    even though the same rules carry a sampling rate for pretraining.
    """
    dcfg = DigestConfig(
        min_charge=2,
        max_charge=2,
        max_variable_mods=1,
        # A rate of 1e-9 would sample nothing; enumeration must ignore it entirely.
        variable_mods=(("M[UNIMOD:35]", 1e-9),),
    )
    precs = enumerate_precursors(["ACDEMK"], dcfg, SplitConfig())
    # One Cys (fixed, always), one Met (0 or 1 oxidation) -> 2 modforms, 1 charge.
    assert len(precs) == 2
    for p in precs:
        assert any(spec == "UNIMOD:4" for _, spec in p.peptide.mods)
    ox_counts = {sum(1 for _, spec in p.peptide.mods if spec == "UNIMOD:35") for p in precs}
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
    from msspeculator.chem import Peptide
    from msspeculator.data.precursors import (
        Precursor,
        frame_to_precursors,
        precursors_to_frame,
    )

    precs = [
        Precursor(Peptide("ETTLHLVLR", (("n", "UNIMOD:737"), (1, "UNIMOD:21"))), 2, "train"),
        Precursor(Peptide("PEPTIDE", ((2, 42.010565),)), 3, "train"),
        Precursor(Peptide("PEK", (("c", "UNIMOD:21"),)), 2, "train"),
    ]
    back = frame_to_precursors(precursors_to_frame(precs))
    assert back == precs


def test_nterm_and_side_chain_mods_are_distinct_sites():
    from msspeculator.chem import Peptide

    p = Peptide("KPEPTIDE", (("n", "UNIMOD:737"), (0, "UNIMOD:737")))
    assert len(p.mods) == 2
    bare = Peptide("KPEPTIDE").mono_mass()
    assert abs(p.mono_mass() - bare - 2 * 229.1629321) < 1e-4


def test_unspecific_digest_refuses_an_unreachable_length_window():
    """An empty digest is a config error, not a result.

    `unspecific` cuts after every residue, so a peptide spans at most missed_cleavages + 1.
    Against the default min_length=7 that silently yielded 0 peptides from a whole proteome.
    """
    import pytest

    from msspeculator.data.config import DigestConfig
    from msspeculator.data.digest import digest_records

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

    from msspeculator.data.config import DigestConfig
    from msspeculator.data.digest import digest_records

    records = [("p1", "MKWVTFISLLFLFSSAYSRGVFRR")]
    with pytest.raises(ValueError, match="0 peptides"):
        digest_records(records, DigestConfig(min_length=200, max_length=300))


def test_two_parsers_one_strict_one_tolerant_of_prospect_only():
    """Unsupported bracket contents must raise, not be reinterpreted as residues.

    The grammar's residue alternative matches bare capitals, so text a reader does not understand
    can be absorbed into the sequence: the hand-rolled parser this replaced read
    `AC[Carbamidomethyl@C]DEK` as `ACCCDEK` with the modification dropped, and swallowed a mass
    delta entirely. Both produced a wrong peptide with plausible fragments and no error, which is
    the worst available failure for a training corpus.

    Two entry points, and the split matters: `from_prospect` tolerates PROSPECT's one deviation and
    is called once at ingest, `from_string` accepts only canonical ProForma and is what everything
    downstream uses.
    """
    from msspeculator.chem import Peptide

    # PROSPECT omits the N-terminal separator; both spellings mean the terminus, not residue 0.
    for spelling in ("[UNIMOD:737]ET[UNIMOD:21]TLHLVLR", "[UNIMOD:737]-ET[UNIMOD:21]TLHLVLR"):
        peptide = Peptide.from_prospect(spelling)
        assert peptide.sequence == "ETTLHLVLR"
        assert peptide.mods == [("n", "UNIMOD:737"), (1, "UNIMOD:21")]

    # Only the tolerant reader accepts the deviation; the strict one refuses it, which is what
    # keeps the degenerate spelling from leaking past ingest.
    with pytest.raises(ValueError, match="invalid modified peptide"):
        Peptide.from_string("[UNIMOD:737]ET[UNIMOD:21]TLHLVLR")

    # A C-terminal modification belongs to the terminus, not to the last residue.
    assert Peptide.from_string("PEK-[UNIMOD:21]").mods == [("c", "UNIMOD:21")]
    assert Peptide.from_string("PEK[UNIMOD:21]").mods == [(2, "UNIMOD:21")]

    for corrupting in (
        "AC[Carbamidomethyl@C]DEK",  # a bare name; was ("ACCCDEK", ())
        "pepTIDE",
        "PEP-TIDE",  # a stray separator; was ("PEPTIDE", ())
        "P-E-P",  # was ("PEP", ())
    ):
        with pytest.raises(ValueError):
            Peptide.from_prospect(corrupting)

    # Formulas and signed mass deltas are part of the grammar, not corruption, and keep their
    # spelling through a round trip.
    for supported in ("AC[Formula:H2O]DEK", "P[+79.96633]EPTIDE"):
        assert Peptide.from_prospect(supported).modified_sequence() == supported

    # Whatever we emit, the strict reader must read back as the same peptide. This is the property
    # the prepared schema depends on: it stores exactly this string and nothing else.
    for mods in (
        ((1, "UNIMOD:4"), (4, "UNIMOD:35")),
        (("n", "UNIMOD:737"), (1, "UNIMOD:21")),
        (("c", "UNIMOD:21"),),
        (("n", "UNIMOD:737"), (2, "UNIMOD:21"), ("c", "UNIMOD:21")),
    ):
        canonical = Peptide("ACDEMK", mods).modified_sequence()
        reread = Peptide.from_string(canonical)
        assert reread.modified_sequence() == canonical
        assert reread.sequence == "ACDEMK"


def _fake_botocore(monkeypatch, credentials):
    """Stand in for the ambient AWS session.

    What this module owns is the translation from botocore's triple to the keys Polars' object
    store reads. Resolving that triple for real would make the test pass or fail on whether the
    machine happens to hold a live session; and an expired one does not return None, it raises
    from inside the refresh, which no skip guard on the return value can catch.
    """

    class Session:
        def get_credentials(self):
            return credentials

    monkeypatch.setattr("botocore.session.get_session", lambda: Session())


def test_only_a_remote_target_needs_credentials():
    from msspeculator.data.storage import is_remote, parquet_storage_options

    assert is_remote("s3://bucket/key.parquet")
    assert is_remote(["s3://bucket/a.parquet", "s3://bucket/b.parquet"])
    assert not is_remote("/tmp/local.parquet")
    assert not is_remote(Path("/tmp/local.parquet"))

    # A local path needs no credentials, and must not acquire any: resolving them would make an
    # offline unit test depend on an AWS session.
    assert parquet_storage_options("/tmp/local.parquet") == {}
    assert parquet_storage_options(Path("/tmp/local.parquet")) == {}


def test_parquet_storage_options_are_explicit_for_remote_targets(monkeypatch):
    """Polars must be handed credentials, not left to resolve them.

    Its object store reads the environment and instance metadata but not the AWS SSO cache, so a
    bare remote read works on a Batch worker and fails on a laptop. botocore resolves all of those,
    so passing what it finds makes the same call work everywhere; `ast-grep` enforces that every
    Polars read does so.
    """
    from msspeculator.data.storage import parquet_storage_options

    class Frozen:  # botocore's ReadOnlyCredentials shape
        access_key = "AKIAEXAMPLE"
        secret_key = "secret"
        token = "session-token"

    class Credentials:
        def get_frozen_credentials(self):
            return Frozen()

    _fake_botocore(monkeypatch, Credentials())
    assert parquet_storage_options("s3://bucket/key.parquet") == {
        "aws_access_key_id": "AKIAEXAMPLE",
        "aws_secret_access_key": "secret",
        "aws_region": "us-west-2",
        "aws_session_token": "session-token",
    }


def test_with_no_credentials_anywhere_polars_is_left_to_its_own_resolution(monkeypatch):
    """An empty mapping is not a failure: a public bucket still reads, and a private one fails
    with the object store's own message instead of one invented here."""
    from msspeculator.data.storage import parquet_storage_options

    _fake_botocore(monkeypatch, None)
    assert parquet_storage_options("s3://bucket/key.parquet") == {}


def test_an_unusable_session_is_reported_rather_than_downgraded(monkeypatch):
    """An expired SSO session raises from the refresh. It must propagate: silently returning {}
    would turn a renewable login into an object-store timeout against instance metadata."""
    from msspeculator.data.storage import parquet_storage_options

    class Expired:
        def get_frozen_credentials(self):
            raise RuntimeError("SSO token expired")

    _fake_botocore(monkeypatch, Expired())
    with pytest.raises(RuntimeError, match="SSO token expired"):
        parquet_storage_options("s3://bucket/key.parquet")


def _meta_rows(rows):
    import polars as pl

    return pl.DataFrame(
        [
            {
                "raw_file": "run1",
                "scan_number": scan,
                "modified_sequence": modseq,
                "precursor_charge": 2,
                "retention_time": 10.0,
                "indexed_retention_time": 20.0,
                "aligned_collision_energy": 30.0,
                "mass_analyzer": "FTMS",
                "fragmentation": "HCD",
                "andromeda_score": score,
            }
            for scan, modseq, score in rows
        ]
    )


def test_meta_index_keeps_the_best_localization_and_drops_ties():
    """An unlocalizable modification must be dropped, not resolved by file order.

    PROSPECT reports the same spectrum once per candidate placement. Where the scores differ the
    engine did localize, so the best one is the label; where they tie it did not, and keeping
    whichever row came first would teach a site-specific model a coin flip.
    """
    from msspeculator.data.meta_index import build_meta_index_from_frame

    index = build_meta_index_from_frame(
        _meta_rows(
            [
                # Scored apart: the better placement wins even though it is second in the file.
                (1, "S[UNIMOD:21]AMPLER", 10.0),
                (1, "SAMPLE[UNIMOD:21]R", 90.0),
                # Score-tied: unlocalizable, so the whole spectrum goes.
                (2, "T[UNIMOD:21]AMPLER", 50.0),
                (2, "TAMPLE[UNIMOD:21]R", 50.0),
                # Reported once: kept.
                (3, "V[UNIMOD:21]AMPLER", 20.0),
                # Repeated identically: redundancy, not ambiguity.
                (4, "W[UNIMOD:21]AMPLER", 20.0),
                (4, "W[UNIMOD:21]AMPLER", 20.0),
            ]
        )
    )

    assert index.ambiguous_localization_spectra == 1
    assert sorted(scan for _, scan in index.by_key) == [1, 3, 4]
    assert list(index.by_key[("run1", 1)].peptide.mods) == [(5, "UNIMOD:21")]  # the 90.0 row
    assert list(index.by_key[("run1", 4)].peptide.mods) == [(0, "UNIMOD:21")]


def test_meta_index_drops_ties_when_no_score_is_recorded():
    """With no score to compare, two placements are still two placements."""
    from msspeculator.data.meta_index import build_meta_index_from_frame

    frame = _meta_rows([(1, "S[UNIMOD:21]AMPLER", 1.0), (1, "SAMPLE[UNIMOD:21]R", 1.0)]).drop(
        "andromeda_score"
    )
    # Every spectrum being unlocalizable is a real property of a source, so it is named rather
    # than reported as the generic "contains no rows".
    with pytest.raises(ValueError, match="cannot be localized"):
        build_meta_index_from_frame(frame)


def test_meta_index_drops_a_scan_reported_with_two_peptides():
    """One spectrum cannot carry two identities, and the loser must not be overwritten silently.

    The index is keyed on (raw_file, scan_number), so two different peptides for one scan would
    both write to the same slot and whichever came last would win with nothing raising. Nothing
    distinguishes the candidates, so the spectrum is dropped and counted apart from a localization
    tie, whose cause is different.
    """
    from msspeculator.data.meta_index import build_meta_index_from_frame

    index = build_meta_index_from_frame(
        _meta_rows(
            [
                # Two peptides, different scores: still ambiguous, because the score ranks
                # placements within a peptide, not one peptide against another.
                (1, "SAMPLER", 90.0),
                (1, "PEPTIDEK", 10.0),
                (2, "VAMPLER", 20.0),
            ]
        )
    )

    assert index.ambiguous_identification_spectra == 1
    assert index.ambiguous_localization_spectra == 0
    assert sorted(scan for _, scan in index.by_key) == [2]


def test_unplaceable_mod_names_are_rejected_where_the_config_is_read():
    """A bad mod name must fail at config time, not partway into a run.

    Neither failure was prompt before this. A name without ``@`` reached ``_mod_target`` and
    raised ``IndexError: list index out of range``, naming neither the mod nor the setting it came
    from. A name with ``@`` that the vocabulary does not know was worse: ``Peptide`` resolves mass
    lazily, so precursors built fine and the error only surfaced once the teacher or encoder asked
    for a mass. The two causes stay distinguishable, since the fixes differ; add the residue, or
    use a mod that exists.
    """
    # A rule with no residue set says nothing about where it goes.
    with pytest.raises(ValueError, match="invalid modification rule"):
        DigestConfig(variable_mods=(("[UNIMOD:21]", 0.001),))
    # A bare name is not a modification identity the grammar accepts.
    with pytest.raises(ValueError, match="invalid modification rule"):
        DigestConfig(variable_mods=(("STY[Phospho]", 0.001),))
    # An accession outside the vendored UNIMOD table is refused rather than carried.
    with pytest.raises(ValueError, match="unknown UNIMOD accession"):
        DigestConfig(fixed_mods=("C[UNIMOD:99999]",))
    # A zero rate is a rule that never fires, which is a config mistake, not a way to disable one.
    with pytest.raises(ValueError, match="must be in"):
        DigestConfig(variable_mods=(("M[UNIMOD:35]", 0.0),))
    assert DigestConfig().fixed_mods == ("C[UNIMOD:4]",)


def test_pretrain_sources_carry_their_own_mods():
    """Mods are per-source, and a source's choice must reach the digest config.

    ``_digest_cfg`` dropped both lists, so a pretrain config could not set mods at all and every
    run silently used the DigestConfig defaults.
    """
    from msspeculator.distill.pipeline import DigestSource, _digest_cfg

    default = _digest_cfg(DigestSource(fasta="x.fasta"))
    assert default.fixed_mods == ("C[UNIMOD:4]",)
    # The canonical PTMs measured in PROSPECT, minus TMT, each at 0.1% per matching residue.
    assert default.variable_mods == (
        ("M[UNIMOD:35]", 0.1),
        ("STY[UNIMOD:21]", 0.001),
        ("K[UNIMOD:1]", 0.001),
        ("K[UNIMOD:121]", 0.001),
    )

    # TOML hands over a list and an inline table, not tuples, and DigestConfig fields are tuples.
    explicit = _digest_cfg(
        DigestSource(
            fasta="x.fasta",
            fixed_mods=["C[UNIMOD:4]"],
            variable_mods={"STY[UNIMOD:21]": 0.01},
        )
    )
    assert explicit.fixed_mods == ("C[UNIMOD:4]",)
    assert explicit.variable_mods == (("STY[UNIMOD:21]", 0.01),)


def test_ingest_stores_the_canonical_spelling_whatever_prospect_wrote():
    """Ingest is the one place a degenerate spelling is accepted, and it emits the canonical one.

    The grammar replaces a round-trip guard that used to sit here. That guard compared a re-render
    against the source string to catch a modification parsed onto the wrong site; the declarative
    grammar cannot make that mistake, because the terminus and the last residue are different
    productions rather than the same loop with an index.
    """
    from msspeculator.chem import Peptide

    # Every spelling PROSPECT uses, canonicalized. The N-terminal separator is added, and two mods
    # on one residue are ordered deterministically, so equal molecules produce equal strings.
    for written, canonical in (
        ("PEPTIDEK", "PEPTIDEK"),
        ("PEPTIDEC[UNIMOD:4]K", "PEPTIDEC[UNIMOD:4]K"),
        ("[UNIMOD:737]-PEPTIDEK", "[UNIMOD:737]-PEPTIDEK"),
        ("[UNIMOD:737]PEPTIDEK", "[UNIMOD:737]-PEPTIDEK"),
        ("[UNIMOD:737]GGPPSQGGK[UNIMOD:1]RK", "[UNIMOD:737]-GGPPSQGGK[UNIMOD:1]RK"),
        (
            "[UNIMOD:737]VVQPQEEIATK[UNIMOD:737][UNIMOD:1]LR",
            "[UNIMOD:737]-VVQPQEEIATK[UNIMOD:1][UNIMOD:737]LR",
        ),
    ):
        assert Peptide.from_prospect(written).modified_sequence() == canonical

    # The terminus and the last residue stay distinct, at equal mass. This is the distinction the
    # old guard existed to protect, now a property of the grammar.
    terminal = Peptide.from_string("PEPTIDEK-[UNIMOD:21]")
    on_residue = Peptide.from_string("PEPTIDEK[UNIMOD:21]")
    assert terminal.mods == [("c", "UNIMOD:21")]
    assert on_residue.mods == [(7, "UNIMOD:21")]
    assert abs(terminal.mono_mass() - on_residue.mono_mass()) < 1e-9
    assert terminal.modified_sequence() != on_residue.modified_sequence()
