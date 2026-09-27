//! Keys to table rows, computed on the host in u64.
//!
//! A table is cut into heads that sit end to end, each a prime number of rows: head `h` owns rows
//! `offset[h] .. offset[h] + size[h]`. A key lands in one row of every head that reads it, `key
//! mod size[h] + offset[h]` after its head's mixing, so two keys that collide in one head rarely
//! collide in another.
//!
//! - [`Ngram`]: each token's n-grams, orders 2..=`order`, `heads` heads per order. This is the
//!   hash of Qwen3.8-Flash-Next's per-layer embedding (vLLM `ple_layer.py:336-435`), bit for bit.
//! - [`Keyed`]: any u64 key (an entity, a feature combination), one row in each of its heads,
//!   `splitmix64(key ^ seed[h]) mod size[h] + offset[h]`.

use crate::memory::NONE;
use hanzo_ml::{bail, Result};

/// `n` if prime; trial division, for the head sizes of a table.
pub fn prime(n: u64) -> bool {
    n > 1 && (2..).take_while(|d| d * d <= n).all(|d| n % d != 0)
}

/// `count` consecutive primes from the first at or above `from`: the sizes of `count` heads.
pub fn primes(from: u64, count: usize) -> Vec<u64> {
    let mut out = Vec::with_capacity(count);
    let mut p = from.max(2);
    while out.len() < count {
        if prime(p) {
            out.push(p);
        }
        p += 1;
    }
    out
}

/// Heads of `sizes` laid end to end from row `start`: each head's first row.
pub fn chain(start: u64, sizes: &[u64]) -> Vec<u64> {
    let mut at = start;
    sizes
        .iter()
        .map(|s| {
            let o = at;
            at += s;
            o
        })
        .collect()
}

/// splitmix64's finalizer: every input bit moves every output bit.
pub fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// `seed`'s i-th draw from splitmix64: the constants a table derives from its seed.
pub fn draw(seed: u64, i: u64) -> u64 {
    mix(seed.wrapping_add(i.wrapping_add(1).wrapping_mul(0x9e37_79b9_7f4a_7c15)))
}

/// The n-gram hash: token ids to table rows.
#[derive(Debug, Clone, PartialEq)]
pub struct Ngram {
    /// Segment separator: a token before a sequence's start reads as it, and it cuts off every
    /// older token for the positions after it.
    pub eos: u32,
    /// Heads per order.
    pub heads: usize,
    /// One multiplier per n-gram position, `mult[i]` for the token `i` places back; the order is
    /// its length.
    pub mult: Vec<u64>,
    /// Per head, orders 2..=order in turn: the head's first row and its (prime) row count.
    pub offset: Vec<u64>,
    pub size: Vec<u64>,
}

impl Ngram {
    /// The hash from its constants, checked: the heads sit end to end from row 0, every row fits
    /// a u32 id, and the counts agree.
    pub fn new(
        eos: u32,
        heads: usize,
        mult: Vec<u64>,
        offset: Vec<u64>,
        size: Vec<u64>,
    ) -> Result<Self> {
        let hash = Self {
            eos,
            heads,
            mult,
            offset,
            size,
        };
        let order = hash.mult.len();
        let end = hash
            .offset
            .iter()
            .zip(&hash.size)
            .try_fold(0u64, |at, (&o, &s)| (o == at).then_some(at + s));
        if order < 2
            || hash.size.len() != (order - 1) * hash.heads
            || hash.offset.len() != hash.size.len()
            || !end.is_some_and(|end| end < u64::from(NONE))
        {
            bail!(
                "n-gram hash is inconsistent: order {order}, {} heads, {} offsets, {} sizes",
                hash.heads,
                hash.offset.len(),
                hash.size.len()
            );
        }
        Ok(hash)
    }

    /// A hash drawn from `seed`: orders 2..=`order`, `heads` heads per order of prime sizes from
    /// `rows`, laid from row 0; multipliers odd and below 2⁶³ / `vocab`, so no product of a token
    /// id and a multiplier overflows.
    pub fn seeded(
        seed: u64,
        eos: u32,
        order: usize,
        heads: usize,
        rows: u64,
        vocab: u64,
    ) -> Result<Self> {
        let bound = (1u64 << 63) / vocab.max(1);
        let mult = (0..order as u64)
            .map(|i| (draw(seed, i) % bound) | 1)
            .collect();
        let size = primes(rows, (order.saturating_sub(1)) * heads);
        let offset = chain(0, &size);
        Self::new(eos, heads, mult, offset, size)
    }

    /// The n-gram order.
    pub fn order(&self) -> usize {
        self.mult.len()
    }

    /// Rows each token reads: `(order − 1) · heads`.
    pub fn width(&self) -> usize {
        self.size.len()
    }

    /// One past the last row a head reaches.
    pub fn end(&self) -> u64 {
        self.offset
            .last()
            .zip(self.size.last())
            .map_or(0, |(o, s)| o + s)
    }

    /// The table rows of each token of `chunk`, [`Ngram::width`] per token, orders 2..=order in
    /// turn. `prior` is the sequence before the chunk; only its last `order − 1` tokens matter,
    /// and missing ones read as EOS (vLLM `model_state.py:65-89`).
    pub fn rows(&self, prior: &[u32], chunk: &[u32]) -> Vec<u32> {
        let mut out = Vec::with_capacity(chunk.len() * self.width());
        self.extend(prior, chunk, &mut out);
        out
    }

    /// [`Ngram::rows`], appended to `out`; allocates nothing when `out` has room.
    pub fn extend(&self, prior: &[u32], chunk: &[u32], out: &mut Vec<u32>) {
        // the token `back` places before chunk[t]: the chunk's, then the prior's, else EOS
        let at = |t: usize, back: usize| -> u32 {
            if back <= t {
                chunk[t - back]
            } else {
                let i = back - t;
                prior.len().checked_sub(i).map_or(self.eos, |j| prior[j])
            }
        };
        for t in 0..chunk.len() {
            // mixed_n = XOR over i < n of x[t-i]·mult[i] (ple_layer.py:426-431). An EOS at t-i cuts
            // off every older token, which then reads as EOS (:337-369); the token at t is never
            // cut. No product overflows: each multiplier is below 2^63 / vocab (:219).
            let mut mixed = u64::from(chunk[t]) * self.mult[0];
            let mut cut = false;
            for n in 2..=self.order() {
                let back = at(t, n - 1);
                mixed ^= u64::from(if cut { self.eos } else { back }) * self.mult[n - 1];
                cut |= back == self.eos;
                // row = mixed_n mod size + offset, per head (:433).
                for h in (n - 2) * self.heads..(n - 1) * self.heads {
                    out.push((mixed % self.size[h] + self.offset[h]) as u32);
                }
            }
        }
    }
}

/// A keyed stream: each key one row in each of its heads.
#[derive(Debug, Clone, PartialEq)]
pub struct Keyed {
    /// Per head: its seed, first row and (prime) row count.
    pub seed: Vec<u64>,
    pub offset: Vec<u64>,
    pub size: Vec<u64>,
}

impl Keyed {
    /// `heads` heads of prime sizes from `rows`, laid from row `start`, seeded from `seed`.
    pub fn seeded(seed: u64, heads: usize, rows: u64, start: u64) -> Keyed {
        let size = primes(rows, heads);
        Keyed {
            seed: (0..heads as u64).map(|h| draw(seed, h)).collect(),
            offset: chain(start, &size),
            size,
        }
    }

    pub fn heads(&self) -> usize {
        self.size.len()
    }

    /// One past the last row a head reaches.
    pub fn end(&self) -> u64 {
        self.offset
            .last()
            .zip(self.size.last())
            .map_or(0, |(o, s)| o + s)
    }

    /// `key`'s row in head `h`.
    pub fn row(&self, key: u64, h: usize) -> u32 {
        (mix(key ^ self.seed[h]) % self.size[h] + self.offset[h]) as u32
    }
}

/// FNV-1a over `bytes`, from `basis`: a stable u64 of a string, for keys.
pub fn fnv(basis: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(basis, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// FNV-1a's offset basis.
pub const BASIS: u64 = 0xcbf2_9ce4_8422_2325;
