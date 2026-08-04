//! A fitted model's NAME, computed from its content.
//!
//! This is the risk plane's construction (`cloud/apps/risk/address.go`) applied to
//! fitted estimators, and the reasoning transfers unchanged:
//!
//!   * The address covers everything that makes two models answer the same input
//!     differently, and NOTHING else. The hyperparameters are in it even though they
//!     are not consulted at prediction time, because two models fitted under different
//!     configurations are two different values even in the rare case their parameters
//!     coincide — a name that ignored the config would call them one.
//!
//!   * Floats are hashed as IEEE-754 BITS and never as text. A decimal rendering is a
//!     lossy function of an `f64`, so two runs holding the same number could print it
//!     differently and be named apart.
//!
//!   * No tenant, no clock, no counter. It is a pure function of the value, so two
//!     processes agree on the name without talking and publication is idempotent.
//!     Isolation is a predicate held elsewhere; a name is never an authority.
//!
//! # Length prefixes are load-bearing
//!
//! Every sequence is written with its length ahead of it. Without that, concatenation
//! is ambiguous: `[[1],[2,3]]` and `[[1,2],[3]]` flatten to the same bytes, so two
//! models of genuinely different shape would collide. The prefix is what makes the
//! encoding injective, and injectivity is the whole property a content address rests
//! on.

use sha2::{Digest as _, Sha256};

/// A model value's name: SHA-256 over its content.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Address([u8; 32]);

impl Address {
    /// The raw digest.
    pub fn bytes(&self) -> [u8; 32] {
        self.0
    }

    /// Lower-case hex, which is the form a store keys on and a report cites.
    pub fn hex(&self) -> String {
        let mut s = String::with_capacity(64);
        for b in self.0 {
            s.push(char::from_digit((b >> 4) as u32, 16).unwrap());
            s.push(char::from_digit((b & 0xf) as u32, 16).unwrap());
        }
        s
    }
}

impl std::fmt::Display for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.hex())
    }
}

impl std::fmt::Debug for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

/// Builds one [`Address`].
///
/// ONE implementation, shared by every model in this crate, so the domain separation,
/// the float encoding and the length prefixing are a single decision rather than three
/// that happen to agree today.
pub struct Digest(Sha256);

impl Digest {
    /// Open a digest in one model kind's own domain.
    ///
    /// The domain is separated with a NUL, so a digest from `linear` can never be
    /// mistaken for one from `boost` — even if their remaining bytes coincided, which
    /// for a two-parameter linear model and a one-node tree is not far-fetched.
    pub fn new(domain: &str) -> Self {
        let mut h = Sha256::new();
        h.update(domain.as_bytes());
        h.update([0u8]);
        Self(h)
    }

    /// A count or a dimension.
    pub fn size(mut self, v: usize) -> Self {
        self.0.update((v as u64).to_be_bytes());
        self
    }

    /// An integer — a class label, or a hyperparameter that counts.
    pub fn int(mut self, v: i64) -> Self {
        self.0.update(v.to_be_bytes());
        self
    }

    /// A flag.
    pub fn flag(mut self, v: bool) -> Self {
        self.0.update([u8::from(v)]);
        self
    }

    /// A real number, as its bits.
    pub fn real(mut self, v: f64) -> Self {
        self.0.update(v.to_bits().to_be_bytes());
        self
    }

    /// A sequence of real numbers, length-prefixed.
    pub fn reals(mut self, v: &[f64]) -> Self {
        self.0.update((v.len() as u64).to_be_bytes());
        for x in v {
            self.0.update(x.to_bits().to_be_bytes());
        }
        self
    }

    /// A sequence of integers, length-prefixed.
    pub fn ints(mut self, v: &[i64]) -> Self {
        self.0.update((v.len() as u64).to_be_bytes());
        for x in v {
            self.0.update(x.to_be_bytes());
        }
        self
    }

    /// A sequence of sizes, length-prefixed.
    pub fn sizes(mut self, v: &[usize]) -> Self {
        self.0.update((v.len() as u64).to_be_bytes());
        for x in v {
            self.0.update((*x as u64).to_be_bytes());
        }
        self
    }

    /// Seal it.
    pub fn finish(self) -> Address {
        Address(self.0.finalize().into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_address_is_sixty_four_hex_characters() {
        let a = Digest::new("hanzo.learn.test").real(1.5).finish();
        assert_eq!(a.hex().len(), 64);
        assert!(a
            .hex()
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()));
    }

    #[test]
    fn the_domain_separates() {
        let a = Digest::new("hanzo.learn.linear").reals(&[1.0]).finish();
        let b = Digest::new("hanzo.learn.boost").reals(&[1.0]).finish();
        assert_ne!(a, b);
    }

    #[test]
    fn length_prefixes_make_the_encoding_injective() {
        // The claim this file rests on: two different shapes cannot flatten to one
        // name. Without the prefix these two are the same byte stream.
        let a = Digest::new("d").reals(&[1.0]).reals(&[2.0, 3.0]).finish();
        let b = Digest::new("d").reals(&[1.0, 2.0]).reals(&[3.0]).finish();
        assert_ne!(a, b);
    }

    #[test]
    fn zero_and_minus_zero_are_different_bit_patterns_and_stay_different_names() {
        // They compare equal as numbers, so a text encoding could easily fold them
        // together. They are different bits and this is a bit-level encoding.
        assert_ne!(
            Digest::new("d").real(0.0).finish(),
            Digest::new("d").real(-0.0).finish()
        );
    }

    #[test]
    fn a_nan_names_itself_consistently() {
        // Floats that are not equal to themselves would break a naming scheme built on
        // numeric comparison. This one is built on bits, so it does not care.
        assert_eq!(
            Digest::new("d").real(f64::NAN).finish(),
            Digest::new("d").real(f64::NAN).finish()
        );
    }
}
