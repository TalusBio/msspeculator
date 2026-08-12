# Reference peptide panels

`prtc.tsv` is a transcription of Table 1 in Thermo Fisher manual MAN0011736 for
the Pierce Retention Time Calibration Mixture (products 88320 and 88321):

<https://documents.thermofisher.com/TFS-Assets/LSG/manuals/MAN0011736_Pierce_Retention_Time_Calib_Mix_UG.pdf>

The source table marks every C-terminal lysine or arginine as heavy. The
`proforma_sequence` column represents heavy lysine (`13C6 15N2`, +8 Da) as
`[UNIMOD:259]` and heavy arginine (`13C6 15N4`, +10 Da) as `[UNIMOD:267]`.
These are explicit CV accessions and therefore do not depend on interpreting
the approximate “+8” and “+10” labels.

`hydrophobicity_factor` preserves the vendor's column name. It is an ordering
yardstick for this mixture, not a claim that the values use the Biognosys iRT
scale.

`biognosys_irt.tsv` transcribes the standard, Q1, and iRT columns from the
`RT->iRT->RT` worksheet in the Biognosys iRT kit reference sheet:

<https://biognosys.com/content/uploads/2021/03/irt-kit-reference-sheet.xls>

`biognosys_irt_transitions.tsv` transcribes the neighboring `iRT Kit
transitions` worksheet. Its relative-intensity column intentionally follows the
integer display precision of that worksheet and is explicitly approximate in
the source. The standards are unmodified peptides, so their bare amino-acid
sequences are valid ProForma strings.
