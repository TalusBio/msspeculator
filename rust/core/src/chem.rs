//! Monoisotopic mass and m/z arithmetic — a 1:1 port of `pepdistill.chem`.
//!
//! v1 supports the 20 standard amino acids and no modifications. Fragment ordering matches
//! `chem.ION_TYPES` = (b,1),(y,1),(b,2),(y,2).

pub const PROTON: f64 = 1.007_276_466_879;
pub const H2O: f64 = 18.010_564_684_25;

/// (ion kind is_b, fragment charge) columns, in the exact order of `chem.ION_TYPES`.
pub const ION_TYPES: [(bool, u8); 4] = [(true, 1), (false, 1), (true, 2), (false, 2)];

/// Monoisotopic residue mass (Da) for a standard amino acid, or `None` if unsupported.
pub fn residue_mass(aa: u8) -> Option<f64> {
    Some(match aa {
        b'G' => 57.021_463_723,
        b'A' => 71.037_113_787,
        b'S' => 87.032_028_409,
        b'P' => 97.052_763_851,
        b'V' => 99.068_413_915,
        b'T' => 101.047_678_473,
        b'C' => 103.009_184_477,
        b'L' => 113.084_064_015,
        b'I' => 113.084_064_015,
        b'N' => 114.042_927_446,
        b'D' => 115.026_943_031,
        b'Q' => 128.058_577_540,
        b'K' => 128.094_963_016,
        b'E' => 129.042_593_095,
        b'M' => 131.040_484_605,
        b'H' => 137.058_911_861,
        b'F' => 147.068_413_915,
        b'R' => 156.101_111_050,
        b'Y' => 163.063_328_575,
        b'W' => 186.079_312_952,
        _ => return None,
    })
}

/// Per-position residue masses for a bare (unmodified) peptide.
pub fn residue_masses(seq: &[u8]) -> anyhow::Result<Vec<f64>> {
    seq.iter()
        .map(|&aa| {
            residue_mass(aa).ok_or_else(|| anyhow::anyhow!("unsupported residue {:?}", aa as char))
        })
        .collect()
}

/// Precursor m/z for a peptide of the given residue masses at `charge`.
pub fn precursor_mz(rm: &[f64], charge: i64) -> f64 {
    let total: f64 = rm.iter().sum();
    (total + H2O + charge as f64 * PROTON) / charge as f64
}

/// Fragment m/z matrix, shape `(L-1, ION_TYPES.len())`, matching `chem.fragment_mz_matrix`.
///
/// Row `i` (0-based) is the site after residue `i+1`: b ordinal `i+1`, y ordinal `L-1-i`.
pub fn fragment_mz_matrix(rm: &[f64]) -> Vec<Vec<f64>> {
    let n = rm.len();
    // prefix[i] = sum of residues 0..=i (cumulative), matching fast.py's cumsum.
    let mut prefix = vec![0.0_f64; n];
    let mut acc = 0.0;
    for (p, &m) in prefix.iter_mut().zip(rm.iter()) {
        acc += m;
        *p = acc;
    }
    let total = prefix[n - 1];
    let mut out = Vec::with_capacity(n.saturating_sub(1));
    // Row i is the site after residue i+1: b ordinal i+1, y ordinal L-1-i (ordinals assigned
    // by the caller). prefix[i] = sum residues 0..=i.
    for &pfx in &prefix[..n - 1] {
        let b_neutral = pfx;
        let y_neutral = total - pfx + H2O;
        let mut row = Vec::with_capacity(ION_TYPES.len());
        for &(is_b, z) in ION_TYPES.iter() {
            let neutral = if is_b { b_neutral } else { y_neutral };
            row.push((neutral + z as f64 * PROTON) / z as f64);
        }
        out.push(row);
    }
    out
}

/// Monoisotopic mass delta (Da) for a named modification, or None if unknown.
pub fn mod_delta(name: &str) -> Option<f64> {
    Some(match name {
        "Carbamidomethyl@C" => 57.021_463_723,
        "Oxidation@M" => 15.994_914_622,
        "Phospho" => 79.966_331_2,
        "TMT6plex" => 229.162_932_1,
        _ => return None,
    })
}

/// Per-position residue masses with modification deltas applied at their sites.
pub fn residue_masses_mod(seq: &[u8], mods: &[(usize, String)]) -> anyhow::Result<Vec<f64>> {
    let mut rm = residue_masses(seq)?;
    for (site, name) in mods {
        let d = mod_delta(name).ok_or_else(|| anyhow::anyhow!("unknown modification {name}"))?;
        rm[*site] += d;
    }
    Ok(rm)
}

pub fn mono_mass(rm: &[f64]) -> f64 {
    rm.iter().sum::<f64>() + H2O
}

/// m/z of a single fragment ion. `ordinal` is 1-based.
pub fn fragment_mz(rm: &[f64], ion: &str, ordinal: usize, charge: i64) -> anyhow::Result<f64> {
    let n = rm.len();
    let neutral = match ion {
        "b" => rm[..ordinal].iter().sum::<f64>(),
        "y" => rm[n - ordinal..].iter().sum::<f64>() + H2O,
        _ => anyhow::bail!("unknown ion type {ion}"),
    };
    Ok((neutral + charge as f64 * PROTON) / charge as f64)
}

/// (n_fragment_positions, n_ion_types) for a peptide of the given length.
pub fn ms2_target_shape(length: usize) -> (usize, usize) {
    (length - 1, ION_TYPES.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) { assert!((a - b).abs() < tol, "{a} vs {b}"); }

    #[test]
    fn mono_mass_peptide() {
        let rm = residue_masses(b"PEPTIDE").unwrap();
        approx(mono_mass(&rm), 799.35997, 0.01);
    }

    #[test]
    fn y1_c_terminal_lysine() {
        let rm = residue_masses(b"SAMPLEK").unwrap();
        approx(fragment_mz(&rm, "y", 1, 1).unwrap(), 128.094963 + H2O + PROTON, 0.001);
    }

    #[test]
    fn b_plus_y_complementarity() {
        let rm = residue_masses(b"SAMPLER").unwrap();
        let n = rm.len();
        let m = mono_mass(&rm);
        for i in 1..n {
            let b = fragment_mz(&rm, "b", i, 1).unwrap();
            let y = fragment_mz(&rm, "y", n - i, 1).unwrap();
            approx(b + y, m + 2.0 * PROTON, 0.001);
        }
    }

    #[test]
    fn fixed_mod_shifts_mass() {
        let base = mono_mass(&residue_masses(b"ACDEK").unwrap());
        let mods = vec![(1usize, "Carbamidomethyl@C".to_string())];
        let modded = mono_mass(&residue_masses_mod(b"ACDEK", &mods).unwrap());
        approx(modded - base, 57.021_463_723, 1e-6);
    }

    #[test]
    fn target_shape() {
        assert_eq!(ms2_target_shape(7), (6, ION_TYPES.len()));
    }
}
