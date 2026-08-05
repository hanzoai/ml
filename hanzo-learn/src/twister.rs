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

    /// Two words as one draw, HIGH WORD FIRST — `numpy`'s 64-bit draw from this generator.
    ///
    /// The order is the entire content of the function. Swapping it leaves a generator that
    /// is still uniform and still reproducible, and that disagrees with `numpy` on every
    /// draw wide enough to need two words.
    fn next_pair(&mut self) -> u64 {
        let high = self.next_word() as u64;
        let low = self.next_word() as u64;
        (high << 32) | low
    }

    /// A value in `0..=most`, by masked rejection — `numpy`'s bounded draw.
    ///
    /// Rejection and not a modulus. A modulus would be faster and would bias the low
    /// values, and it would put this stream permanently out of step with `numpy`'s, which
    /// is the one property the type exists to have.
    ///
    /// # Two widths, because `numpy` has two
    ///
    /// `numpy`'s `random_interval` takes ONE word when the bound fits in one and a PAIR
    /// when it does not, so the width of the bound decides how many words leave the stream.
    /// A single-width version of this is not a simplification of it: from the first bound
    /// that crosses `u32::MAX` it is a different stream, and every draw after that
    /// disagrees. The boundary is `most <= u32::MAX` — a bound of exactly `u32::MAX` still
    /// takes one word, and one more takes two.
    ///
    /// The bound is a `u64` for the same reason. A `u32` bound would not merely refuse
    /// large arguments, it would silently truncate them: shuffling `2³² + 5` rows would
    /// draw positions in `0..=4`.
    pub fn below(&mut self, most: u64) -> u64 {
        if most == 0 {
            return 0;
        }
        let mut mask = most;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        mask |= mask >> 32;
        loop {
            let drawn = if most <= u32::MAX as u64 {
                self.next_word() as u64
            } else {
                self.next_pair()
            };
            let v = drawn & mask;
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
            let j = self.below(i as u64) as usize;
            values.swap(i, j);
        }
    }

    /// `0..n` shuffled, as `RandomState.permutation(n)` returns it.
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut order: Vec<usize> = (0..n).collect();
        self.shuffle(&mut order);
        order
    }

    /// `take` distinct values from `0..n`, in `take` draws and `O(take)` space.
    ///
    /// # Why this is not `permutation(n)` truncated, and why it can exist at all
    ///
    /// Subsampling 256 rows out of 10⁶ by shuffling all 10⁶ costs a million swaps and eight
    /// megabytes to throw 999,744 of them away. Across a hundred trees on every core that is
    /// the difference between a forest whose fit is `O(trees · sample)` and one that only
    /// claims to be.
    ///
    /// It works because [`Twister::shuffle`] sweeps from the TOP: step one settles position
    /// `n-1`, step two settles `n-2`, and a position is never touched again once settled. So
    /// the LAST `take` entries of the shuffle are fully determined after `take` draws, while
    /// the FIRST `take` are not determined until the sweep ends — which is why this returns
    /// the tail and a truncation of the head could not. The positions it never reaches are
    /// never materialised; only the ones a swap actually touched are remembered.
    ///
    /// Identical draws to `permutation(n)`, so the two agree: this is the tail of that
    /// permutation, reversed, and `a_partial_choice_is_the_tail_of_the_whole_shuffle` pins it.
    pub fn choose(&mut self, n: usize, take: usize) -> Vec<usize> {
        let take = take.min(n);
        // Only positions a swap disturbed. Everything else still holds its own index.
        let mut moved: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::with_capacity(take * 2);
        let mut out = Vec::with_capacity(take);
        for step in 0..take {
            let i = n - 1 - step;
            if i == 0 {
                out.push(*moved.get(&0).unwrap_or(&0));
                break;
            }
            let j = self.below(i as u64) as usize;
            let vi = *moved.get(&i).unwrap_or(&i);
            let vj = *moved.get(&j).unwrap_or(&j);
            moved.insert(i, vj);
            moved.insert(j, vi);
            out.push(vj);
        }
        out
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

    /// The width of the draw is decided by the BOUND, not by the type of the argument, and
    /// `u32::MAX` is the last bound that costs one word.
    ///
    /// Asserted as a word COUNT rather than as values, because the count is the property:
    /// a version that took a pair either side of the boundary would still return numbers in
    /// range, and would still be uniform, while standing one word away from `numpy` for the
    /// rest of the stream.
    #[test]
    fn the_bound_decides_whether_a_draw_costs_one_word_or_two() {
        let spent = |most: u64| {
            let mut t = Twister::seed(1);
            t.next_word(); // past the first twist, so `at` counts words within the block
            let before = t.at;
            t.below(most);
            t.at - before
        };
        // A bound of the form 2^k - 1 is all mask, so nothing is ever rejected and the
        // words spent are exactly the words one draw needs.
        assert_eq!(spent(255), 1);
        assert_eq!(
            spent(u32::MAX as u64),
            1,
            "the widest bound that costs one word"
        );
        assert_eq!(spent((1 << 33) - 1), 2, "past it, a draw is a pair");
        assert_eq!(spent(u64::MAX >> 1), 2);
        // The first bound over the boundary rejects about half its attempts, so the count
        // is not fixed — but every attempt is a PAIR, so it is even, and never one.
        let crossed = spent(1 << 32);
        assert!(
            crossed >= 2 && crossed % 2 == 0,
            "a bound of 2^32 spent {crossed} words, so it did not draw pairs"
        );
    }

    /// A bound above `2^32` draws over the whole of it, not over a truncation of it.
    ///
    /// Narrowing this bound to 32 bits leaves `most = 4`, so the narrowed version can only
    /// ever answer `0..=4`. The bar is therefore that the draws REACH — over 64 of them,
    /// a correct draw fails to pass halfway once in `2^64` seeds, and a truncated one
    /// cannot pass at all.
    #[test]
    fn a_bound_above_thirty_two_bits_is_not_truncated() {
        let most = (1u64 << 32) + 4;
        let mut t = Twister::seed(0);
        let drawn: Vec<u64> = (0..64).map(|_| t.below(most)).collect();
        assert!(drawn.iter().all(|&v| v <= most));
        assert!(
            drawn.iter().copied().max().unwrap() > u32::MAX as u64 / 2,
            "every draw landed in the handful of values a 32-bit bound would leave: {drawn:?}"
        );
    }

    /// A partial choice of a design too large to permute is still a choose OF IT.
    #[test]
    fn a_choice_of_a_design_larger_than_four_billion_rows_spans_it() {
        let n = (1usize << 32) + 5;
        let picked = Twister::seed(0).choose(n, 32);
        assert_eq!(picked.len(), 32);
        assert!(picked.iter().all(|&v| v < n));
        let mut sorted = picked.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 32, "a choice repeated a row");
        assert!(
            picked.iter().copied().max().unwrap() > u32::MAX as usize / 2,
            "the row indices collapsed into the low ones: {picked:?}"
        );
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

    /// The partial choice IS the tail of the whole shuffle, which is what makes it a
    /// uniform subsample rather than merely a cheap one.
    #[test]
    fn a_partial_choice_is_the_tail_of_the_whole_shuffle() {
        for n in [1usize, 2, 5, 40, 1000] {
            let whole = Twister::seed(9).permutation(n);
            for take in [1usize, 2, 7, 40] {
                if take > n {
                    continue;
                }
                let part = Twister::seed(9).choose(n, take);
                let want: Vec<usize> = whole[n - take..].iter().rev().cloned().collect();
                assert_eq!(part, want, "n={n} take={take}");
            }
            // Asking for everything reproduces the whole shuffle, reversed.
            let all = Twister::seed(9).choose(n, n);
            let want: Vec<usize> = whole.iter().rev().cloned().collect();
            assert_eq!(all, want, "n={n} take=n");
        }
    }

    #[test]
    fn a_partial_choice_is_distinct_and_in_range() {
        let mut t = Twister::seed(4);
        let picked = t.choose(1_000_000, 256);
        assert_eq!(picked.len(), 256);
        assert!(picked.iter().all(|&v| v < 1_000_000));
        let mut sorted = picked.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 256, "a choice repeated a row");
        // Clamped, not an error, and not a panic on the boundary.
        assert_eq!(t.choose(3, 10).len(), 3);
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
