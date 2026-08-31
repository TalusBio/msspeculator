//! The FASTA and the peptides it digests to, stored as one blob and a table of locations.
//!
//! A tryptic digest of a proteome produces far more peptides than the proteome has residues, and
//! almost every one of them is a substring of something already in memory. Storing each as its
//! own `String`, mapped to a set of owned accession strings, spends hundreds of megabytes
//! restating bytes the FASTA already holds. Here a peptide is two `u32` offsets into one shared
//! blob, and its proteins are `u32` indices into one shared table.

use std::collections::HashMap;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};

use crate::progress::{Phase, Reporter};

const VALID_AA: &str = "GASPVTCLINDQKEMHFRYW";

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
    /// Accession per protein, in the order the FASTA listed them.
    accessions: Vec<String>,
    /// Where each protein's residues are. Deliberately not a prefix sum: two accessions with
    /// byte-identical sequences share one span, so a database concatenated with a contaminant
    /// panel, or one carrying a protein under two names, stores those residues once.
    spans: Vec<Span>,
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
    pub(crate) fn read(path: &Path, rules: &DigestRules, reporter: Reporter<'_>) -> Result<Self> {
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
        self.proteome.accessions.len()
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

    /// Whether the digest produced this exact sequence, by binary search rather than by a set of
    /// copies: the decoy check asks this once per peptide, and a `BTreeSet<String>` built to
    /// answer it would be a second copy of every peptide in the run.
    pub(crate) fn contains(&self, residues: &str) -> bool {
        self.peptides
            .binary_search_by(|peptide| self.proteome.slice(peptide.residues).cmp(residues))
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
pub(crate) struct PeptideRef {
    digest: Arc<Digest>,
    index: u32,
}

impl PeptideRef {
    pub(crate) fn residues(&self) -> &str {
        self.digest.residues_at(self.index as usize)
    }

    /// The accessions this peptide appears in, joined with `;`, in FASTA order.
    ///
    /// Built on demand rather than stored, because only the peptides that survive to a spectrum
    /// need it and a joined string per digested peptide is the allocation this module exists to
    /// avoid.
    pub(crate) fn protein_group(&self) -> String {
        let span = self.digest.peptides[self.index as usize].proteins;
        let members = &self.digest.memberships[span.start as usize..span.end as usize];
        let mut out = String::new();
        for (i, &protein) in members.iter().enumerate() {
            if i > 0 {
                out.push(';');
            }
            out.push_str(&self.digest.proteome.accessions[protein as usize]);
        }
        out
    }
}

/// Accumulates the blob and the ungrouped peptide table while the FASTA is being read.
struct Builder<'a> {
    rules: &'a DigestRules,
    proteome: Proteome,
    /// Every tryptic peptide the digest found, with the protein it came from, before grouping.
    /// Duplicates are expected: grouping them is what discovers a shared peptide.
    found: Vec<(Span, u32)>,
    /// Sequence hash to the proteins already holding those residues, so an exactly repeated
    /// protein sequence is stored once. Holds hashes and indices, never a copy of a sequence.
    by_sequence: HashMap<u64, Vec<u32>>,
}

impl<'a> Builder<'a> {
    fn new(rules: &'a DigestRules) -> Self {
        Self {
            rules,
            proteome: Proteome {
                residues: String::new(),
                accessions: Vec::new(),
                spans: Vec::new(),
            },
            found: Vec::new(),
            by_sequence: HashMap::new(),
        }
    }

    fn protein(&mut self, accession: String, sequence: &str) -> Result<()> {
        let span = self.intern(sequence)?;
        let protein = u32::try_from(self.proteome.accessions.len())
            .map_err(|_| anyhow::anyhow!("FASTA holds more than {} proteins", u32::MAX))?;
        self.proteome.accessions.push(accession);
        self.proteome.spans.push(span);
        for peptide in tryptic_spans(self.proteome.slice(span), span.start, self.rules) {
            self.found.push((peptide, protein));
        }
        Ok(())
    }

    /// Place a protein's residues in the blob, reusing the span of an identical sequence.
    fn intern(&mut self, sequence: &str) -> Result<Span> {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        sequence.hash(&mut hasher);
        let key = hasher.finish();
        if let Some(candidates) = self.by_sequence.get(&key) {
            // A hash match is a candidate, not an answer: compare the residues before sharing a
            // span, or a collision would silently merge two different proteins.
            for &protein in candidates {
                let span = self.proteome.spans[protein as usize];
                if self.proteome.slice(span) == sequence {
                    return Ok(span);
                }
            }
        }
        let start = u32::try_from(self.proteome.residues.len()).map_err(|_| too_many_residues())?;
        self.proteome.residues.push_str(sequence);
        let end = u32::try_from(self.proteome.residues.len()).map_err(|_| too_many_residues())?;
        self.by_sequence
            .entry(key)
            .or_default()
            .push(self.proteome.accessions.len() as u32);
        Ok(Span { start, end })
    }

    /// Group the found peptides by their residues, which is what discovers that one peptide came
    /// from several proteins.
    fn finish(mut self, path: &Path) -> Result<Digest> {
        if self.proteome.accessions.is_empty() {
            bail!("FASTA {} contains no records", path.display());
        }
        let proteome = self.proteome;
        self.found.sort_unstable_by(|a, b| {
            proteome
                .slice(a.0)
                .cmp(proteome.slice(b.0))
                .then(a.1.cmp(&b.1))
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
                // Sorted by protein within a peptide, so a protein listed twice for the same
                // peptide (a repeated sequence, or a missed cleavage reaching the same span)
                // lands next to itself.
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

    /// A scratch path that removes itself, so the tests need no dev-dependency for it.
    struct Scratch(std::path::PathBuf);

    impl Scratch {
        fn new(name: &str, contents: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "msspeculator-proteome-{}-{}",
                std::process::id(),
                name
            ));
            std::fs::write(&path, contents).unwrap();
            Self(path)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn rules(missed: usize, min: usize, max: usize) -> DigestRules {
        DigestRules {
            missed_cleavages: missed,
            min_length: min,
            max_length: max,
        }
    }

    fn digest(contents: &str, rules: DigestRules) -> Arc<Digest> {
        let scratch = Scratch::new("digest.fasta", contents);
        Arc::new(Digest::read(&scratch.0, &rules, Reporter::new(None)).unwrap())
    }

    /// `unwrap_err` would need `Digest: Debug`, and a `Debug` that prints a whole proteome is a
    /// worse failure message than none.
    fn read_err(path: &Path) -> anyhow::Error {
        match Digest::read(path, &rules(0, 5, 30), Reporter::new(None)) {
            Err(error) => error,
            Ok(digest) => panic!("expected a refusal, got {} peptides", digest.peptides()),
        }
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
        assert_eq!(shared.protein_group(), "first;second");
        let unique = peptides(&digest)
            .find(|p| p.residues() == "LNQAEDNTER")
            .expect("unique peptide missing");
        assert_eq!(unique.protein_group(), "first");
    }

    /// The reason the span table is not a prefix sum: a contaminant panel concatenated onto a
    /// database repeats whole sequences, and there is no reason to store them twice.
    #[test]
    fn an_identical_sequence_under_two_accessions_is_stored_once() {
        let digest = digest(
            ">first d\nPEPTIDEKSAMPLERTAILR\n>second d\nPEPTIDEKSAMPLERTAILR\n",
            rules(0, 5, 30),
        );
        assert_eq!(digest.proteins(), 2);
        assert_eq!(digest.proteome.residues.len(), 20);
        assert_eq!(digest.proteome.spans[0], digest.proteome.spans[1]);
        let peptide = peptides(&digest)
            .find(|p| p.residues() == "PEPTIDEK")
            .unwrap();
        assert_eq!(peptide.protein_group(), "first;second");
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
        assert!(digest.contains("PEPTIDEK"));
        assert!(!digest.contains("PEPTIDE"));
        assert!(!digest.contains("KEDITPEP"));
    }

    #[test]
    fn a_fasta_with_no_records_is_refused() {
        let scratch = Scratch::new("empty.fasta", "\n\n");
        let error = read_err(&scratch.0);
        assert!(error.to_string().contains("no records"), "{error}");
    }

    #[test]
    fn residues_before_a_header_are_refused_with_the_line() {
        let scratch = Scratch::new("headerless.fasta", "PEPTIDEK\n>p d\nSAMPLER\n");
        let error = read_err(&scratch.0);
        assert!(error.to_string().contains(":1"), "{error}");
    }
}
