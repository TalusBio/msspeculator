# Reference peptide panels

`prtc.tsv` is a transcription of Table 1 in Thermo Fisher manual MAN0011736 for
the Pierce Retention Time Calibration Mixture (products 88320 and 88321):

<https://documents.thermofisher.com/TFS-Assets/LSG/manuals/MAN0011736_Pierce_Retention_Time_Calib_Mix_UG.pdf>

The source table marks every C-terminal lysine or arginine as heavy. The
`proforma_sequence` column represents heavy lysine (`13C6 15N2`, +8 Da) as
`[UNIMOD:259]` and heavy arginine (`13C6 15N4`, +10 Da) as `[UNIMOD:267]`.
These are explicit CV accessions and therefore do not depend on interpreting
the approximate "+8" and "+10" labels.

`hydrophobicity_factor` preserves the vendor's column name. It is an ordering
yardstick for this mixture, not a claim that the values use the Biognosys iRT
scale.

`biognosys_irt.tsv` transcribes the standard, Q1, and iRT columns from the
`RT->iRT->RT` worksheet in the Biognosys iRT kit reference sheet:

<https://biognosys.com/content/uploads/2021/03/irt-kit-reference-sheet.xls>

`diagnostic_spectra.tsv` is the fixed MS2 panel the model doctor scores against. Unlike the
tables above it transcribes no vendor sheet: it is three real spectra drawn from the prepared
corpus by `tools/vendor_reference_spectra.py`, which is how it gets regenerated.

    uv run --extra etl python tools/vendor_reference_spectra.py \
        --chunks '<prepared-prefix>/shards/prospect_TUM_first_pool_1/*/data.parquet' \
        --chunks '<prepared-prefix>/shards/multi_ptm_TUM_mod_pS/*/data.parquet' \
        --chunks '<prepared-prefix>/shards/prospect_TUM_HLA/*/data.parquet' \
        --out data/reference_peptides/diagnostic_spectra.tsv

Four properties make the panel usable as a fixed reference:

- **Validation split.** Validation already drives early stopping and checkpoint selection, so
  reading it every epoch shows the run nothing it was not already seeing. Test stays untouched,
  and train would not show a fit going wrong.
- **Ranked on quality, not position.** Backbone coverage first, then Andromeda score. All three
  currently cover every fragmentation site and score 366 to 455, against a corpus median nearer
  120. A spectrum missing half its backbone would let a model be wrong in the gaps for free.
  Both numbers are columns in the file, so a later run that picks differently is a visible change
  of evidence rather than an unexplained diff.
- **One spectrum per dataset**, across a phosphopeptide, a non-tryptic HLA peptide, and a plain
  tryptic one. Three spectra from one dataset would be three views of one acquisition.
- **FTMS only.** An ion trap discards fragments below roughly a third of the precursor m/z, so an
  ITMS spectrum can put its base peak at the detection edge, and then a failing check cannot be
  told from an instrument limit.

Peaks are struct-of-arrays within a row (`annotations`, `fragment_mz`, `relative_intensity` are
`;`-separated and parallel), so a sequence is written once and the file stays legible in a diff.
Intensities are relative to each spectrum's base peak. The underlying spectra come from
ProteomeTools via PROSPECT, which is publicly available.

This panel is real experimental data, which `biognosys_irt_transitions.tsv` is not: that one is a
triple-quadrupole method sheet, an acquisition the model is never trained on, so it pins m/z
arithmetic and nothing about predicted intensity.

`biognosys_irt_transitions.tsv` transcribes the neighboring `iRT Kit
transitions` worksheet. Its relative-intensity column intentionally follows the
integer display precision of that worksheet and is explicitly approximate in
the source. The standards are unmodified peptides, so their bare amino-acid
sequences are valid ProForma strings.
