//! The FASTA and the peptides it digests to, stored as one blob and a table of locations.
//!
//! A tryptic digest of a proteome produces far more peptides than the proteome has residues, and
//! almost every one of them is a substring of something already in memory. Storing each as its
//! own `String`, mapped to a set of owned accession strings, spends hundreds of megabytes
//! restating bytes the FASTA already holds. Here a peptide is two `u32` offsets into one shared
//! blob, and its proteins are `u32` indices into one shared table.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};

use crate::progress::{Phase, Reporter};

const VALID_AA: &str = "GASPVTCLINDQKEMHFRYW";

/// Prefixed onto every protein of a decoy spectrum, which is the convention every search engine
/// reads. Never stored: it is applied when a [`FastaId`] is written.
pub(crate) const DECOY_PREFIX: &str = "DECOY_";

/// One protein as the FASTA identified it: the first whitespace-delimited token of a description
/// line, unparsed.
///
/// Deliberately not called an accession, a name, or a symbol. For a UniProt FASTA this holds
/// `sp|P00001|A_HUMAN`, which is a database, an accession and an entry name run together; calling
/// that any one of them would be wrong, and the four candidate meanings collide badly enough
/// already. Whatever the header said is what a consumer gets back.
///
/// Borrows rather than owns, and applies the decoy prefix while writing rather than while
/// storing, so a peptide in ten proteins costs no allocation at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FastaId<'a> {
    prefix: &'static str,
    id: &'a str,
}

impl std::fmt::Display for FastaId<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.prefix)?;
        f.write_str(self.id)
    }
}

/// The stripped residues of one spectrum's peptide.
///
/// A view over the digest rather than a sequence of its own, because a decoy's residues are a
/// permutation of its target's: pinning the termini and reading the interior backwards produces
/// them on demand. A decoy therefore costs no allocation and no second copy of residues the FASTA
/// already holds.
///
/// Residues are ASCII by construction, so the iterator yields bytes.
#[derive(Clone, Copy)]
pub struct Residues<'a> {
    source: &'a str,
    reversed: bool,
}

impl<'a> Residues<'a> {
    /// The peptide as digested.
    ///
    /// Public, like [`Self::pseudo_reversed`], because a caller assembling a [`SpectrumRow`] for
    /// its own tests needs both; the pipeline calls exactly these two, so there is no third
    /// arrangement that only a test executes.
    ///
    /// [`SpectrumRow`]: crate::SpectrumRow
    pub fn target(source: &'a str) -> Self {
        Self {
            source,
            reversed: false,
        }
    }

    /// The pseudo-reversed reading of the same peptide: first and last residue pinned so the
    /// decoy keeps its tryptic context, everything between them backwards.
    pub fn pseudo_reversed(source: &'a str) -> Self {
        Self {
            source,
            reversed: true,
        }
    }

    pub fn len(&self) -> usize {
        self.source.len()
    }

    pub fn is_empty(&self) -> bool {
        self.source.is_empty()
    }

    pub fn iter(self) -> impl ExactSizeIterator<Item = u8> + 'a {
        let bytes = self.source.as_bytes();
        let length = bytes.len();
        let reversed = self.reversed;
        (0..length).map(move |i| {
            // A sequence of two residues or fewer is all termini, so there is nothing to turn
            // around and the reading is the same either way.
            let source = if reversed && length > 2 && i > 0 && i < length - 1 {
                length - 1 - i
            } else {
                i
            };
            bytes[source]
        })
    }
}

impl std::fmt::Display for Residues<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use std::fmt::Write;
        if !self.reversed {
            return f.write_str(self.source);
        }
        for residue in self.iter() {
            // ASCII by construction, so the byte and the char are the same value.
            f.write_char(residue as char)?;
        }
        Ok(())
    }
}

/// The proteins whose digest produced one peptide.
///
/// A list rather than a joined string. DIA-NN's `;` separator is a property of that one format,
/// so only its writer should have to know about it; a group that arrives pre-joined has to be
/// split apart again by everything else, on a delimiter a FASTA identifier is free to contain.
///
/// One representation: a table of identifiers plus this peptide's indices into it, which is what
/// the digest holds. A caller assembling a row by hand builds the same shape rather than a
/// parallel one, so no arrangement is executed only by tests.
#[derive(Clone, Copy)]
pub struct ProteinGroup<'a> {
    identifiers: &'a [String],
    members: &'a [u32],
    prefix: &'static str,
}

impl<'a> ProteinGroup<'a> {
    /// Build a group, for a caller assembling a [`SpectrumRow`] itself.
    ///
    /// `members` indexes `identifiers`, so the same identifier table can back many peptides;
    /// a group standing alone passes `&[0, 1, ..]`.
    ///
    /// [`SpectrumRow`]: crate::SpectrumRow
    pub fn new(identifiers: &'a [String], members: &'a [u32], decoy: bool) -> Self {
        Self {
            identifiers,
            members,
            prefix: if decoy { DECOY_PREFIX } else { "" },
        }
    }

    /// The proteins, ordered by identifier.
    ///
    /// Takes `self` rather than `&self`, which `Copy` makes free, so the iterator outlives the
    /// group it came from and a caller can hand it straight to a `for` or a `join`.
    pub fn iter(self) -> impl Iterator<Item = FastaId<'a>> {
        self.members.iter().map(move |&member| FastaId {
            prefix: self.prefix,
            id: &self.identifiers[member as usize],
        })
    }
}

/// A half-open range of residues, as offsets into [`Proteome::residues`].
///
/// `u32` rather than `usize`, which halves the table and caps a run at four billion residues:
/// about four hundred times the human proteome, and checked rather than wrapped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Span {
    start: u32,
    end: u32,
}

/// Every protein sequence in one allocation.
struct Proteome {
    /// All residues concatenated, uppercased, with no separators between proteins.
    residues: String,
    /// One entry per distinct identifier, in the order the FASTA first listed it.
    accessions: Vec<String>,
}

impl Proteome {
    fn slice(&self, span: Span) -> &str {
        &self.residues[span.start as usize..span.end as usize]
    }
}

/// One distinct peptide sequence and the proteins it came from.
struct DigestedPeptide {
    /// Where the residues are, in [`Proteome::residues`].
    residues: Span,
    /// Where this peptide's protein indices are, in [`Digest::memberships`].
    proteins: Span,
}

/// What the digest is allowed to cut.
pub(crate) struct DigestRules {
    pub(crate) missed_cleavages: usize,
    pub(crate) min_length: usize,
    pub(crate) max_length: usize,
}

/// A FASTA and its tryptic digest.
///
/// Held behind an `Arc` so a [`PeptideRef`] handed to the enumeration loop is twelve bytes and an
/// atomic increment rather than a copy of the peptide and its protein names.
pub(crate) struct Digest {
    proteome: Proteome,
    /// FASTA records read, which a repeated identifier makes larger than the identifier table.
    records: usize,
    /// One entry per distinct peptide sequence, ordered by residues. The order is what makes a
    /// rerun assign the same decoy pair ids to the same peptides.
    peptides: Vec<DigestedPeptide>,
    /// Every peptide's protein indices, concatenated. One allocation rather than one `Vec` per
    /// peptide, which at eight million peptides is eight million allocations holding four bytes.
    memberships: Vec<u32>,
}

impl Digest {
    /// Read a FASTA and digest it, reporting bytes consumed as it goes.
    ///
    /// Digesting during the read rather than after it means the file is never held twice: a
    /// record's residues go straight into the blob, and the peptides it yields are recorded as
    /// spans before the next record is read. Grouping the spans by sequence happens once at the
    /// end, on a table that holds no strings at all.
    pub(crate) fn read(path: &Path, rules: &DigestRules, reporter: &Reporter<'_>) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening FASTA {}", path.display()))?;
        // The progress denominator, read once. A file whose length will not come back still
        // digests; it reports against zero, which `Progress::fraction` answers as 0.0.
        let total = file.metadata().map(|meta| meta.len()).unwrap_or(0);
        let mut reader = BufReader::new(file);

        let mut builder = Builder::new(rules);
        let mut id: Option<String> = None;
        let mut seq = String::new();
        let mut consumed = 0u64;
        let mut line_no = 0usize;
        let mut line = String::new();
        // `read_line` rather than `lines()`, so `consumed` counts the bytes the file holds rather
        // than the bytes left over after a split has discarded every line ending.
        loop {
            line.clear();
            let read = reader
                .read_line(&mut line)
                .with_context(|| format!("reading {}:{}", path.display(), line_no + 1))?;
            if read == 0 {
                break;
            }
            consumed += read as u64;
            line_no += 1;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Some(header) = trimmed.strip_prefix('>') {
                let accession = header
                    .split_whitespace()
                    .next()
                    .context("empty FASTA header")?
                    .to_string();
                if let Some(previous) = id.replace(accession) {
                    builder.protein(previous, &seq)?;
                    seq.clear();
                    reporter.at(Phase::Digesting, consumed, total);
                }
            } else {
                if id.is_none() {
                    bail!(
                        "{}:{} has sequence before the first FASTA header",
                        path.display(),
                        line_no
                    );
                }
                seq.push_str(&trimmed.to_ascii_uppercase());
            }
        }
        if let Some(id) = id {
            builder.protein(id, &seq)?;
        }
        let digest = builder.finish(path)?;
        reporter.at(Phase::Digesting, total, total);
        Ok(digest)
    }

    pub(crate) fn proteins(&self) -> usize {
        self.records
    }

    pub(crate) fn peptides(&self) -> usize {
        self.peptides.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.peptides.is_empty()
    }

    fn residues_at(&self, index: usize) -> &str {
        self.proteome.slice(self.peptides[index].residues)
    }

    /// Whether the digest produced this exact sequence.
    ///
    /// Binary search over the sorted peptide table rather than a set built from it: the decoy
    /// check asks this once per peptide, and a `BTreeSet<String>` built to answer it would be a
    /// second copy of every peptide in the run. Compared residue by residue, so a caller asking
    /// about a reversed reading never has to materialize it.
    pub(crate) fn contains(&self, residues: Residues<'_>) -> bool {
        self.peptides
            .binary_search_by(|peptide| {
                self.proteome
                    .slice(peptide.residues)
                    .bytes()
                    .cmp(residues.iter())
            })
            .is_ok()
    }
}

/// Every peptide in a digest, each as a handle into the shared blob.
pub(crate) fn peptides(digest: &Arc<Digest>) -> impl Iterator<Item = PeptideRef> + '_ {
    (0..digest.peptides()).map(|index| PeptideRef {
        digest: Arc::clone(digest),
        index: index as u32,
    })
}

/// One digested peptide, as a location in a shared [`Digest`] rather than a copy of it.
#[derive(Clone)]
pub(crate) struct PeptideRef {
    digest: Arc<Digest>,
    index: u32,
}

impl PeptideRef {
    pub(crate) fn residues(&self) -> &str {
        self.digest.residues_at(self.index as usize)
    }

    /// The proteins this peptide came from, borrowed from the digest.
    ///
    /// `decoy` picks the spelling rather than the membership: a decoy is the reversal of this
    /// peptide, so it belongs to the same proteins and differs only in carrying the prefix.
    pub(crate) fn proteins(&self, decoy: bool) -> ProteinGroup<'_> {
        let span = self.digest.peptides[self.index as usize].proteins;
        ProteinGroup::new(
            &self.digest.proteome.accessions,
            &self.digest.memberships[span.start as usize..span.end as usize],
            decoy,
        )
    }
}

/// Accumulates the blob and the ungrouped peptide table while the FASTA is being read.
struct Builder<'a> {
    rules: &'a DigestRules,
    proteome: Proteome,
    /// Every tryptic peptide the digest found, with the protein it came from, before grouping.
    /// Duplicates are expected: grouping them is what discovers a shared peptide.
    found: Vec<(Span, u32)>,
    /// Identifier to its index, so one identifier is one entry in the table however many records
    /// carry it. Without this a FASTA listing a protein twice under one name puts that name in a
    /// group twice, and a search engine counting protein evidence double-counts it.
    by_identifier: HashMap<String, u32>,
    /// Records read, which is not `accessions.len()` once a FASTA repeats an identifier. Reported
    /// as the protein count because it is what the file contained.
    records: usize,
}

impl<'a> Builder<'a> {
    fn new(rules: &'a DigestRules) -> Self {
        Self {
            rules,
            proteome: Proteome {
                residues: String::new(),
                accessions: Vec::new(),
            },
            found: Vec::new(),
            by_identifier: HashMap::new(),
            records: 0,
        }
    }

    fn protein(&mut self, accession: String, sequence: &str) -> Result<()> {
        let start = u32::try_from(self.proteome.residues.len()).map_err(|_| too_many_residues())?;
        self.proteome.residues.push_str(sequence);
        let end = u32::try_from(self.proteome.residues.len()).map_err(|_| too_many_residues())?;
        let protein = self.identifier(accession)?;
        self.records += 1;
        for peptide in tryptic_spans(
            &self.proteome.residues[start as usize..end as usize],
            start,
            self.rules,
        ) {
            self.found.push((peptide, protein));
        }
        Ok(())
    }

    /// The table index for one identifier, assigning it on first sight.
    fn identifier(&mut self, accession: String) -> Result<u32> {
        if let Some(&protein) = self.by_identifier.get(&accession) {
            return Ok(protein);
        }
        let protein =
            u32::try_from(self.proteome.accessions.len()).map_err(|_| too_many_proteins())?;
        self.by_identifier.insert(accession.clone(), protein);
        self.proteome.accessions.push(accession);
        Ok(protein)
    }

    /// Group the found peptides by their residues, which is what discovers that one peptide came
    /// from several proteins.
    fn finish(mut self, path: &Path) -> Result<Digest> {
        if self.records == 0 {
            bail!("FASTA {} contains no records", path.display());
        }
        let proteome = self.proteome;
        let records = self.records;
        // Ordered by residues, then by identifier text rather than by table index, so a group
        // reads alphabetically however the FASTA happened to list its records. The identifier
        // comparison only runs when two peptides are the same, which is rare.
        self.found.sort_unstable_by(|a, b| {
            proteome.slice(a.0).cmp(proteome.slice(b.0)).then_with(|| {
                proteome.accessions[a.1 as usize].cmp(&proteome.accessions[b.1 as usize])
            })
        });

        let mut peptides = Vec::new();
        let mut memberships: Vec<u32> = Vec::new();
        for (span, protein) in self.found {
            let residues = proteome.slice(span);
            let same_peptide = peptides
                .last()
                .is_some_and(|last: &DigestedPeptide| proteome.slice(last.residues) == residues);
            if same_peptide {
                let last = peptides.last_mut().expect("just checked");
                // Identifiers are interned, so one identifier is one index and the sort above
                // puts every repeat of it next to itself: a protein reached twice for the same
                // peptide is listed once.
                if memberships.last() != Some(&protein) {
                    memberships.push(protein);
                    last.proteins.end = memberships.len() as u32;
                }
                continue;
            }
            let start = memberships.len() as u32;
            memberships.push(protein);
            peptides.push(DigestedPeptide {
                residues: span,
                proteins: Span {
                    start,
                    end: memberships.len() as u32,
                },
            });
        }
        Ok(Digest {
            proteome,
            records,
            peptides,
            memberships,
        })
    }
}

fn too_many_residues() -> anyhow::Error {
    anyhow::anyhow!(
        "FASTA holds more than {} residues, which is more than this digest can address",
        u32::MAX
    )
}

fn too_many_proteins() -> anyhow::Error {
    anyhow::anyhow!(
        "FASTA holds more than {} distinct protein identifiers, which is more than this digest \
         can address",
        u32::MAX
    )
}

/// Tryptic peptides of one protein, as spans offset by where that protein sits in the blob.
fn tryptic_spans(sequence: &str, base: u32, rules: &DigestRules) -> Vec<Span> {
    let bytes = sequence.as_bytes();
    let mut sites = vec![0usize];
    for (i, &aa) in bytes.iter().enumerate() {
        if (aa == b'K' || aa == b'R') && bytes.get(i + 1) != Some(&b'P') {
            sites.push(i + 1);
        }
    }
    if sites.last().copied() != Some(bytes.len()) {
        sites.push(bytes.len());
    }
    let mut out = Vec::new();
    for start in 0..sites.len().saturating_sub(1) {
        for mc in 0..=rules.missed_cleavages {
            let end = start + mc + 1;
            if end >= sites.len() {
                break;
            }
            let peptide = &sequence[sites[start]..sites[end]];
            if (rules.min_length..=rules.max_length).contains(&peptide.len())
                && peptide.bytes().all(|aa| VALID_AA.as_bytes().contains(&aa))
            {
                out.push(Span {
                    start: base + sites[start] as u32,
                    end: base + sites[end] as u32,
                });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::scratch::Scratch;

    fn rules(missed: usize, min: usize, max: usize) -> DigestRules {
        DigestRules {
            missed_cleavages: missed,
            min_length: min,
            max_length: max,
        }
    }

    fn digest(contents: &str, rules: DigestRules) -> Arc<Digest> {
        let scratch = Scratch::holding("digest.fasta", contents);
        Arc::new(Digest::read(scratch.path(), &rules, &Reporter::new(None)).unwrap())
    }

    /// `unwrap_err` would need `Digest: Debug`, and a `Debug` that prints a whole proteome is a
    /// worse failure message than none.
    fn read_err(path: &Path) -> anyhow::Error {
        match Digest::read(path, &rules(0, 5, 30), &Reporter::new(None)) {
            Err(error) => error,
            Ok(digest) => panic!("expected a refusal, got {} peptides", digest.peptides()),
        }
    }

    fn group(peptide: &PeptideRef, decoy: bool) -> Vec<String> {
        peptide
            .proteins(decoy)
            .iter()
            .map(|id| id.to_string())
            .collect()
    }

    fn sequences(digest: &Arc<Digest>) -> Vec<String> {
        peptides(digest).map(|p| p.residues().to_string()).collect()
    }

    #[test]
    fn tryptic_digest_honors_proline_and_missed_cleavages() {
        let digest = digest(">p desc\nAAAKPLLLRCCCK\n", rules(1, 3, 30));
        // `KP` is not a cleavage site, so the first peptide runs through it.
        assert_eq!(
            sequences(&digest),
            vec!["AAAKPLLLR", "AAAKPLLLRCCCK", "CCCK"]
        );
    }

    #[test]
    fn a_peptide_in_two_proteins_lists_both_once() {
        let digest = digest(
            ">first d\nSAMPLEPEPTIDEKLNQAEDNTER\n>second d\nSAMPLEPEPTIDEKMTHEDTAILR\n",
            rules(0, 5, 30),
        );
        let shared = peptides(&digest)
            .find(|p| p.residues() == "SAMPLEPEPTIDEK")
            .expect("shared peptide missing");
        assert_eq!(group(&shared, false), ["first", "second"]);
        // A decoy belongs to the same proteins and differs only in the prefix, which is applied
        // per member rather than to a joined string.
        assert_eq!(group(&shared, true), ["DECOY_first", "DECOY_second"]);
        let unique = peptides(&digest)
            .find(|p| p.residues() == "LNQAEDNTER")
            .expect("unique peptide missing");
        assert_eq!(group(&unique, false), ["first"]);
    }

    /// A self-concatenated database, or one carrying a protein under a name it already used, must
    /// not put that name in a group twice: a search engine counting protein evidence would count
    /// it twice too.
    #[test]
    fn a_repeated_identifier_is_one_member_of_a_group() {
        let digest = digest(
            ">dup d\nPEPTIDEKSAMPLERTAILR\n>dup d\nPEPTIDEKSAMPLERTAILR\n>bbb d\nPEPTIDEKQQQQQR\n",
            rules(0, 5, 30),
        );
        // Three records, two distinct identifiers.
        assert_eq!(digest.proteins(), 3);
        let shared = peptides(&digest)
            .find(|p| p.residues() == "PEPTIDEK")
            .unwrap();
        assert_eq!(group(&shared, false), ["bbb", "dup"]);
    }

    /// Alphabetical, not FASTA order, so a regenerated library does not diff against an older one
    /// on every shared peptide just because the records were reordered.
    #[test]
    fn a_group_is_ordered_by_identifier_not_by_position() {
        let digest = digest(
            ">zzz d\nPEPTIDEKQQQQQR\n>aaa d\nPEPTIDEKWWWWWR\n",
            rules(0, 5, 30),
        );
        let shared = peptides(&digest)
            .find(|p| p.residues() == "PEPTIDEK")
            .unwrap();
        assert_eq!(group(&shared, false), ["aaa", "zzz"]);
    }

    #[test]
    fn peptides_come_out_sorted_so_a_rerun_numbers_them_the_same_way() {
        let digest = digest(">p d\nYYYYKAAAAKMMMMK\n", rules(0, 4, 30));
        let listed = sequences(&digest);
        assert_eq!(listed, vec!["AAAAK", "MMMMK", "YYYYK"]);
    }

    /// The digest reads bytes rather than lines, so the line endings and the missing final
    /// newline a real FASTA arrives with have to leave the same peptides behind.
    #[test]
    fn digestion_survives_crlf_and_a_missing_final_newline() {
        let digest = digest(
            ">protein_one d\r\nPEPTIDEMR\r\n\r\n>protein_two d\r\nSAMPLETID",
            rules(0, 9, 9),
        );
        assert_eq!(digest.proteins(), 2);
        assert_eq!(sequences(&digest), vec!["PEPTIDEMR", "SAMPLETID"]);
    }

    #[test]
    fn a_sequence_that_was_never_digested_is_not_reported_as_a_target() {
        let digest = digest(">p d\nPEPTIDEKSAMPLER\n", rules(0, 5, 30));
        assert!(digest.contains(Residues::target("PEPTIDEK")));
        assert!(!digest.contains(Residues::target("PEPTIDE")));
        // The question the decoy path actually asks, answered without the reversal being built.
        assert!(!digest.contains(Residues::pseudo_reversed("PEPTIDEK")));
        assert!(digest.contains(Residues::pseudo_reversed("PEDITPEK")));
    }

    /// The property `run_library`'s decoy check rests on; see the comment there.
    #[test]
    fn pseudo_reverse_is_an_involution() {
        let reverse = |s: &str| Residues::pseudo_reversed(s).to_string();
        for sequence in [
            "PEPTIDEK",
            "AK",
            "K",
            "",
            "AAK",
            "PEPTIDERPEPTIDEK",
            "MCMCMCMR",
        ] {
            assert_eq!(reverse(&reverse(sequence)), sequence, "{sequence}");
        }
    }

    #[test]
    fn reversed_residues_pin_the_termini_and_need_no_string() {
        let read = |residues: Residues<'_>| residues.iter().map(|b| b as char).collect::<String>();
        assert_eq!(read(Residues::pseudo_reversed("PEPTIDEK")), "PEDITPEK");
        assert_eq!(read(Residues::target("PEPTIDEK")), "PEPTIDEK");
        // Two residues or fewer are all termini, so there is nothing to turn around.
        for short in ["", "K", "AK"] {
            assert_eq!(read(Residues::pseudo_reversed(short)), short);
        }
        // `Display` and the iterator have to agree, since sinks use whichever suits them.
        let reversed = Residues::pseudo_reversed("PEPTIDEK");
        assert_eq!(reversed.to_string(), read(reversed));
        assert_eq!(reversed.len(), 8);
    }

    #[test]
    fn a_fasta_with_no_records_is_refused() {
        let scratch = Scratch::holding("empty.fasta", "\n\n");
        let error = read_err(scratch.path());
        assert!(error.to_string().contains("no records"), "{error}");
    }

    #[test]
    fn residues_before_a_header_are_refused_with_the_line() {
        let scratch = Scratch::holding("headerless.fasta", "PEPTIDEK\n>p d\nSAMPLER\n");
        let error = read_err(scratch.path());
        assert!(error.to_string().contains(":1"), "{error}");
    }
}
