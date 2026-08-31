//! The DIA-NN TSV serialization: one row per transition, one flat table.
//!
//! Everything here is this format's own spelling. `SpectrumRow` carries the peptidoform and the
//! protein group as structured values; the joining, the `(UniMod:N)` annotation syntax, and the
//! `Decoy=1` column are DIA-NN's conventions and live with DIA-NN's writer.

use std::io::Write;

use anyhow::Result;
use msspeculator_core::peptide::{ModSpec, Peptide, Site};

use crate::library::{LibrarySink, SpectrumRow};
use crate::proteome::ProteinGroup;
use crate::provenance::LibraryProvenance;

pub(crate) struct DiannSink<W: Write> {
    pub(crate) writer: W,
}

impl<W: Write + Send> LibrarySink for DiannSink<W> {
    fn header(&mut self, _provenance: &LibraryProvenance) -> Result<()> {
        writeln!(self.writer, "ModifiedPeptide\tStrippedPeptide\tPrecursorMz\tPrecursorCharge\tTr_recalibrated\tIonMobility\tProteinID\tDecoy\tFragmentMz\tFragmentType\tFragmentNumber\tFragmentCharge\tFragmentLossType\tRelativeIntensity")?;
        Ok(())
    }

    fn spectrum(&mut self, row: &SpectrumRow<'_>) -> Result<()> {
        // Both spellings this format needs, built once per precursor rather than per transition.
        let proteins = SemicolonJoined(row.proteins);
        let modified = ModifiedPeptide(row.peptide);
        for peak in &row.peaks {
            writeln!(
                self.writer,
                "{}\t{}\t{:.8}\t{}\t{:.6}\t{:.8}\t{}\t{}\t{:.8}\t{}\t{}\t{}\tnoloss\t{:.8}",
                modified,
                row.stripped,
                row.precursor_mz,
                row.charge,
                row.rt,
                row.mobility,
                proteins,
                u8::from(row.decoy),
                peak.mz,
                peak.ion,
                peak.ordinal,
                peak.charge,
                peak.intensity,
            )?;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

/// A protein group in DIA-NN's single protein column.
///
/// The one place `;` belongs. A `Display` adapter rather than a joined `String`, so the separator
/// reaches the output stream without a `Vec` and a `String` built to hold it on the way.
#[derive(Clone, Copy)]
struct SemicolonJoined<'a>(ProteinGroup<'a>);

impl std::fmt::Display for SemicolonJoined<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, protein) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(";")?;
            }
            write!(f, "{protein}")?;
        }
        Ok(())
    }
}

/// A peptidoform in DIA-NN's modified-peptide spelling, `PEPC(UniMod:4)IDER`.
///
/// Rendered here rather than carried on the row: it is one format's notation, and mzSpecLib spells
/// the same peptidoform as ProForma. Written straight to the stream for the same reason as the
/// protein group.
struct ModifiedPeptide<'a>(&'a Peptide);

impl std::fmt::Display for ModifiedPeptide<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Peptide { sequence, mods, .. } = self.0;
        for (site, spec) in mods {
            if *site == Site::NTerm {
                write!(f, "{}", Annotation(spec))?;
            }
        }
        for (i, aa) in sequence.chars().enumerate() {
            write!(f, "{aa}")?;
            for (site, spec) in mods {
                if *site == Site::Residue(i) {
                    write!(f, "{}", Annotation(spec))?;
                }
            }
        }
        for (site, spec) in mods {
            if *site == Site::CTerm {
                write!(f, "{}", Annotation(spec))?;
            }
        }
        Ok(())
    }
}

/// One modification in DIA-NN's notation.
struct Annotation<'a>(&'a ModSpec);

impl std::fmt::Display for Annotation<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            ModSpec::Unimod { accession, .. } => write!(f, "(UniMod:{accession})"),
            ModSpec::MassOnly(mass) => write!(f, "({mass:+})"),
            ModSpec::Formula { formula, .. } => write!(f, "[Formula:{formula}]"),
        }
    }
}

#[cfg(test)]
pub(crate) fn modified_peptide(peptide: &Peptide) -> String {
    ModifiedPeptide(peptide).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use msspeculator_core::proforma::parse_descriptor;

    fn oxidation() -> ModSpec {
        parse_descriptor("UNIMOD:35").unwrap()
    }

    #[test]
    fn modifications_render_in_diann_notation_at_their_site() {
        let peptide = Peptide::new(
            "PEPTIDEMR".to_string(),
            vec![(Site::Residue(7), oxidation())],
        );
        assert_eq!(modified_peptide(&peptide), "PEPTIDEM(UniMod:35)R");
    }

    #[test]
    fn terminal_modifications_sit_outside_the_sequence() {
        let peptide = Peptide::new(
            "PEPTIDEK".to_string(),
            vec![(Site::NTerm, oxidation()), (Site::CTerm, oxidation())],
        );
        assert_eq!(modified_peptide(&peptide), "(UniMod:35)PEPTIDEK(UniMod:35)");
    }
}
