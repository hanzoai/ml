//! The Mersenne twister, drawn exactly as `numpy` draws it.
//!
//! # Why bit-exact matters here and nowhere else
//!
//! A resampling plan is not a number, it is a DECISION about which rows a model may see.
//! Two implementations that disagree about that decision cannot be compared: a Rust model
//! scoring better than a Python baseline on "the same" five folds has proved nothing if
//! the folds differ. So the fold assignment is held to `numpy`'s stream bit for bit, which
//! makes a scikit-learn baseline and a `hanzo-learn` challenger comparable on identical
//! data — and makes the fixtures in `tests/fixture` assert equality of INDICES rather than
//! a distributional property that any shuffle would satisfy.
//!
//! Nothing else in this crate draws random numbers, so this is the only place the choice of
//! generator is visible, and no fit is at the mercy of it.
//!
//! # What this is not
//!
//! Not for anything that must be unpredictable. MT19937's state is recoverable from its
//! output, which is exactly why it is reproducible, and reproducibility is the property
//! wanted here. Keys, tokens and nonces come from the platform generator.
//!
//! Clean-room from the published algorithm (Matsumoto and Nishimura 1998) and `numpy`'s
//! documented legacy seeding and bounded-draw behaviour, verified against fixtures its own
//! implementation produced.

/// Degree of the recurrence: how many words of state.
const N: usize = 624;
/// The middle word of the recurrence.
const M: usize = 397;
/// Coefficients of the twist.
const A: u32 = 0x9908_b0df;
const UPPER: u32 = 0x8000_0000;
const LOWER: u32 = 0x7fff_ffff;

/// A reproducible stream of words, seeded the way `numpy.random.RandomState` seeds one.
#[derive(Debug, Clone)]
pub struct Twister {
    state: [u32; N],
    at: usize,
}

impl Twister {
    /// Seed from one integer, as `RandomState(seed)` does.
    pub fn seed(seed: u32) -> Self {
        let mut state = [0u32; N];
        let mut s = seed;
        for (position, word) in state.iter_mut().enumerate() {
            *word = s;
            // The seeding recurrence advances AFTER storing, with the position plus one
            // added in. Writing it the other way round shifts the whole stream by a word.
            s = 1_812_433_253u32
                .wrapping_mul(s ^ (s >> 30))
                .wrapping_add(position as u32 + 1);
        }
        Self { state, at: N }
    }

    /// One tempered word.
    pub fn next_word(&mut self) -> u32 {
        if self.at >= N {
            self.twist();
        }
        let mut y = self.state[self.at];
        self.at += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    fn twist(&mut self) {
        for i in 0..N - M {
            let y = (self.state[i] & UPPER) | (self.state[i + 1] & LOWER);
            self.state[i] = self.state[i + M] ^ (y >> 1) ^ ((y & 1).wrapping_neg() & A);
        }
        for i in N - M..N - 1 {
            let y = (self.state[i] & UPPER) | (self.state[i + 1] & LOWER);
            self.state[i] = self.state[i + M - N] ^ (y >> 1) ^ ((y & 1).wrapping_neg() & A);
        }
        let y = (self.state[N - 1] & UPPER) | (self.state[0] & LOWER);
        self.state[N - 1] = self.state[M - 1] ^ (y >> 1) ^ ((y & 1).wrapping_neg() & A);
        self.at = 0;
    }

    /// A word in `0..=most`, by masked rejection — `numpy`'s bounded draw.
    ///
    /// Rejection and not a modulus. A modulus would be faster and would bias the low
    /// values, and it would put this stream permanently out of step with `numpy`'s, which
    /// is the one property the type exists to have.
    pub fn below(&mut self, most: u32) -> u32 {
        if most == 0 {
            return 0;
        }
        let mut mask = most;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        loop {
            let v = self.next_word() & mask;
            if v <= most {
                return v;
            }
        }
    }

    /// A real in `[0, 1)`, as `RandomState.random_sample` returns one.
    ///
    /// Two words, not one: 53 bits is the whole mantissa of an `f64`, and one 32-bit word
    /// would leave the low 21 bits permanently zero. The high word contributes 27 bits and
    /// the low word 26, in that order, because the order is the difference between this
    /// stream and `numpy`'s.
    pub fn next_real(&mut self) -> f64 {
        let high = (self.next_word() >> 5) as f64;
        let low = (self.next_word() >> 6) as f64;
        (high * 67_108_864.0 + low) / 9_007_199_254_740_992.0
    }

    /// Shuffle in place, as `RandomState.shuffle` does: a Fisher-Yates sweep from the top,
    /// each step drawing a position in `0..=i`.
    pub fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let j = self.below(i as u32) as usize;
            values.swap(i, j);
        }
    }

    /// `0..n` shuffled, as `RandomState.permutation(n)` returns it.
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut order: Vec<usize> = (0..n).collect();
        self.shuffle(&mut order);
        order
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference stream of MT19937 seeded with 5489 — the algorithm's own published
    /// test vector, which pins the recurrence and the tempering independently of any
    /// fixture.
    #[test]
    fn the_published_test_vector_matches() {
        let mut t = Twister::seed(5489);
        let first: Vec<u32> = (0..5).map(|_| t.next_word()).collect();
        assert_eq!(
            first,
            vec![3499211612, 581869302, 3890346734, 3586334585, 545404204]
        );
    }

    #[test]
    fn a_seed_is_the_whole_state_so_two_streams_agree() {
        let mut a = Twister::seed(42);
        let mut b = Twister::seed(42);
        for _ in 0..2000 {
            assert_eq!(a.next_word(), b.next_word());
        }
        assert_ne!(Twister::seed(42).next_word(), Twister::seed(43).next_word());
    }

    #[test]
    fn a_bounded_draw_stays_in_range_and_reaches_both_ends() {
        let mut t = Twister::seed(7);
        let mut low = false;
        let mut high = false;
        for _ in 0..10_000 {
            let v = t.below(9);
            assert!(v <= 9);
            low |= v == 0;
            high |= v == 9;
        }
        assert!(low && high);
        assert_eq!(t.below(0), 0);
    }

    /// `numpy.random.RandomState(seed).random_sample(k)`, to the last bit. Two seeds,
    /// because a single one cannot distinguish the right word order from a lucky one.
    #[test]
    fn a_real_draw_matches_numpy_bit_for_bit() {
        let mut t = Twister::seed(7);
        let got: Vec<f64> = (0..5).map(|_| t.next_real()).collect();
        assert_eq!(
            got,
            vec![
                0.076_308_289_373_957_17,
                0.779_918_792_240_114_6,
                0.438_409_231_440_893_5,
                0.723_465_177_830_941_2,
                0.977_989_511_996_602_7
            ]
        );
        let mut t = Twister::seed(0);
        let got: Vec<f64> = (0..3).map(|_| t.next_real()).collect();
        assert_eq!(
            got,
            vec![
                0.548_813_503_927_324_8,
                0.715_189_366_372_419_5,
                0.602_763_376_071_643_9
            ]
        );
    }

    #[test]
    fn a_real_draw_stays_below_one() {
        let mut t = Twister::seed(11);
        for _ in 0..50_000 {
            let v = t.next_real();
            assert!((0.0..1.0).contains(&v), "{v}");
        }
    }

    #[test]
    fn a_permutation_is_a_permutation() {
        let mut t = Twister::seed(0);
        let p = t.permutation(500);
        let mut sorted = p.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..500).collect::<Vec<_>>());
        assert_ne!(p, (0..500).collect::<Vec<_>>());
    }
}
