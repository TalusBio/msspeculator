//! Deterministic train/val/test assignment, ported from `src/pepdistill/data/split.py`.
//!
//! Hashing rather than sampling is what makes a peptide keep its assignment as the corpus grows,
//! across machines and across runs. Every modified form and charge state of one bare sequence
//! hashes together, so nothing leaks between splits through a modification.
//!
//! This is a second implementation of a contract, which is a thing this codebase otherwise avoids
//!, but the fit runs here and the corpus is prepared there, and both have to agree about which
//! peptides a model was allowed to see. It is pinned from Python against golden values; changing
//! either side without the other silently invalidates every held-out number the project reports.

use blake2::digest::consts::U8;
use blake2::{Blake2b, Digest};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    Train,
    Val,
    Test,
}

#[derive(Debug, Clone)]
pub struct SplitConfig {
    pub train: f64,
    pub val: f64,
    pub test: f64,
    pub salt: String,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            train: 0.8,
            val: 0.1,
            test: 0.1,
            salt: "pepdistill-v1".to_string(),
        }
    }
}

/// Map a sequence to a stable float in [0, 1).
///
/// blake2b with an 8-byte digest over `"{salt}:{sequence}"`, read big-endian and divided by 2^64,
/// each step matching `_unit_hash`, since any difference silently reassigns peptides.
pub fn unit_hash(sequence: &str, salt: &str) -> f64 {
    let mut hasher = Blake2b::<U8>::new();
    hasher.update(format!("{salt}:{sequence}").as_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest);
    u64::from_be_bytes(bytes) as f64 / 18446744073709551616.0 // 2^64
}

/// Assign a bare (unmodified) sequence to a split.
pub fn assign_split(sequence: &str, cfg: &SplitConfig) -> Split {
    let h = unit_hash(sequence, &cfg.salt);
    if h < cfg.train {
        Split::Train
    } else if h < cfg.train + cfg.val {
        Split::Val
    } else {
        Split::Test
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden values produced by the Python implementation. Regenerate with:
    ///
    /// ```text
    /// uv run python -c "
    /// from pepdistill.data.split import _unit_hash
    /// for s in ['PEPTIDEK','ACDEFGHIK','SAMPLERK','TESTPEPTIDER','K']:
    ///     print(s, repr(_unit_hash(s, 'pepdistill-v1')))"
    /// ```
    const GOLDEN: [(&str, f64); 5] = [
        ("PEPTIDEK", 0.001_923_595_662_914_6),
        ("ACDEFGHIK", 0.148_034_139_029_839_6),
        ("SAMPLERK", 0.717_579_479_388_606_4),
        ("TESTPEPTIDER", 0.264_004_869_540_141_9),
        ("K", 0.012_600_369_097_530_3),
    ];

    #[test]
    fn unit_hash_matches_python() {
        for (sequence, expected) in GOLDEN {
            let got = unit_hash(sequence, "pepdistill-v1");
            assert!(
                (got - expected).abs() < 1e-15,
                "{sequence}: rust {got} != python {expected}"
            );
        }
    }

    #[test]
    fn splits_follow_the_configured_fractions() {
        let cfg = SplitConfig::default();
        // Hashes 0.0399 / 0.8067 / 0.9882 against boundaries at 0.8 and 0.9.
        assert_eq!(assign_split("PEPTIDECK", &cfg), Split::Train);
        assert_eq!(assign_split("PEPTIDEYK", &cfg), Split::Val);
        assert_eq!(assign_split("PEPTIDERK", &cfg), Split::Test);
    }

    #[test]
    fn the_salt_is_part_of_the_contract() {
        // Changing it reassigns every peptide, which is why it travels in SplitConfig rather
        // than being a constant someone could tune locally.
        let other = SplitConfig {
            salt: "different".to_string(),
            ..SplitConfig::default()
        };
        assert_ne!(
            unit_hash("PEPTIDEK", "pepdistill-v1"),
            unit_hash("PEPTIDEK", &other.salt)
        );
    }
}
