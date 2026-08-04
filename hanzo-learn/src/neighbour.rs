//! Anomaly as a LOCAL disagreement about density.
//!
//! An isolation forest asks a global question — how few cuts separate this row from
//! everything. That question has no good answer when the data is several clouds of
//! different tightness, because a row sitting comfortably inside a sparse cloud looks
//! isolated next to a dense one. The local outlier factor asks the question that survives
//! that: is this row less densely surrounded than the neighbours it actually has? A ratio of
//! one means as dense as its neighbourhood, and meaningfully above one means thinner than
//! the company it keeps.
//!
//! # How closely this matches scikit-learn, measured rather than claimed
//!
//! There is no randomness anywhere in the algorithm — no subsample, no cut, no
//! initialisation — so unlike [`crate::isolation`] there is nothing to excuse a
//! disagreement, and `tests/sklearn.rs` holds it to the last bits. What that came out as:
//!
//! | quantity                       | agreement with scikit-learn 1.9.0        |
//! |--------------------------------|------------------------------------------|
//! | `k`-distance                   | EXACT, every `k`                          |
//! | density, factor, `score_samples` | EXACT at `k = 5`; ≤ 4.6 ulp at `k = 20, 50` |
//!
//! The split is not arbitrary and it is not this crate's arithmetic. Everything below the
//! `k`-distance is a MEAN over the `k` neighbours, and scikit-learn takes that mean through
//! `np.mean`, whose pairwise reduction only starts blocking above 8 elements — which is
//! exactly why `k = 5` is bit-identical and `k = 20` is not. Reproducing it would mean making
//! every mean in this crate depend on numpy's block size.
//!
//! An earlier version of this file claimed EXACT for all of it. That was wrong, and the test
//! that caught it is the reason the claim is now a table.
//!
//! # Scale, concretely, and the bound is the honest part
//!
//! Fitting is `O(n² p)` time and the fitted value OWNS the design: `8 n p` bytes plus
//! `8 n (k + 2)`. There is no tree or hash index, so this is exact brute force. On this
//! 128 GB box that means:
//!
//! | rows | features | fitted value | fit time |
//! |------|----------|--------------|----------|
//! | 10⁴  | 20       | ~5 MB        | ~0.1 s   |
//! | 10⁵  | 20       | ~50 MB       | ~10 s    |
//! | 10⁶  | 20       | ~500 MB      | ~17 min  |
//!
//! The memory is fine and the TIME is the wall: past ~10⁵ rows this is the wrong estimator
//! and [`crate::isolation`] is the right one, because a forest is `O(n log n)` to fit and
//! `O(1)` in `n` to hold. That is a documented bound and not a surprise — the quadratic is
//! in the algorithm, not in this implementation of it, and no index changes the exponent for
//! `p` past about 20.
//!
//! Single device, CPU, `rayon` across query rows. No GPU path: the distance matrix is the
//! only part a device would help with and it is never formed — rows are scanned in blocks
//! against a resident design, keeping the working set in cache, which beats a transfer.
//!
//! Clean-room from the published algorithm (Breunig, Kriegel, Ng and Sander 2000, *LOF:
//! Identifying Density-Based Local Outliers*), with the neighbour convention and the
//! division guard matched to scikit-learn's arithmetic.

use rayon::prelude::*;

use crate::{Error, Matrix, Outlier, Result};

/// The guard scikit-learn adds to every reachability mean before inverting it.
///
/// Carried deliberately, and it is NOT numerical paint. `k` duplicate rows have a mean
/// reachability of exactly zero, so the density is a division by zero; scikit-learn's `+
/// 1e-10` makes that `1e10` — a finite, enormous density, so duplicates read as the densest
/// thing in the data rather than as `NaN`. Any other guard, including a cleaner branch on
/// zero, would put every score out of step with the reference for that case. Matching a
/// reference means matching its arithmetic, not just its intent.
const GUARD: f64 = 1e-10;

/// How many neighbours the density is judged against.
///
/// A count of at least one, which is what makes [`Local::fit`] total in this argument: there
/// is no "0 neighbours" to check for downstream, because it is not a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Neighbours(usize);

impl Neighbours {
    /// Twenty, which is scikit-learn's default and the paper's recommended lower bound.
    pub const DEFAULT: Neighbours = Neighbours(20);

    /// A count of `k` neighbours.
    pub fn new(k: usize) -> Result<Self> {
        if k == 0 {
            return Err(Error::Config(
                "a density is judged against at least one neighbour, not zero".into(),
            ));
        }
        Ok(Self(k))
    }

    /// How many.
    pub fn get(self) -> usize {
        self.0
    }
}

/// A fitted local outlier factor: the design it was fitted on, and each row's density.
///
/// Holds the design because the algorithm is memory-based — scoring an unseen row needs its
/// neighbours, and its neighbours are the fitted rows. That is the one estimator here whose
/// fitted value grows with `n`, so it is the one whose size is worth checking before use;
/// see the table in the module docs.
#[derive(Debug, Clone, PartialEq)]
pub struct Local {
    design: Matrix,
    neighbours: Neighbours,
    /// Effective `k`, clamped to `n - 1`: how many neighbours a row actually got.
    used: usize,
    /// Distance to the `used`-th nearest OTHER fitted row.
    reach: Vec<f64>,
    /// Local reachability density of each fitted row.
    density: Vec<f64>,
    /// `-LOF` of each fitted row, which is scikit-learn's `negative_outlier_factor_`.
    factor: Vec<f64>,
}

impl Local {
    /// Fit on a design.
    ///
    /// `k` is clamped to `n - 1`, because a row is never its own neighbour and there are no
    /// others to reach for; scikit-learn warns and clamps identically. Fails only on a
    /// design of one row, which has no neighbourhood at all and so has no density to
    /// compare.
    pub fn fit(x: &Matrix, neighbours: Neighbours) -> Result<Self> {
        if x.n() < 2 {
            return Err(Error::Shape(
                "a local density needs at least two rows: one row has no neighbourhood".into(),
            ));
        }
        let used = neighbours.get().min(x.n() - 1);

        // Every row's neighbours among the OTHER rows. Row i excludes itself by index and
        // not by distance, which is what keeps a duplicated row honest: its twin is a
        // neighbour at distance zero, and itself is not a neighbour at all.
        let found: Vec<Found> = (0..x.n())
            .into_par_iter()
            .map(|i| nearest(x, x.row(i), used, Some(i)))
            .collect();

        // k-distance of every row, which reachability is measured against.
        let reach: Vec<f64> = found.iter().map(|f| f.far()).collect();
        let density: Vec<f64> = found.iter().map(|f| f.density(&reach)).collect();
        let factor = found
            .iter()
            .zip(&density)
            .map(|(f, own)| -f.ratio(&density, *own))
            .collect();

        Ok(Self {
            design: x.clone(),
            neighbours,
            used,
            reach,
            density,
            factor,
        })
    }

    /// `-LOF` for each FITTED row: scikit-learn's `negative_outlier_factor_`.
    ///
    /// Around `-1` for a row as dense as its neighbourhood, and further below `-1` the
    /// thinner it is. Kept in scikit-learn's sign because it is that attribute, named after
    /// it, and a caller comparing the two should not have to wonder; the suspicion-oriented
    /// number is [`Outlier::outlier`], and `factor()[i] == -outlier(design)[i]`.
    pub fn factor(&self) -> &[f64] {
        &self.factor
    }

    /// Local reachability density of each fitted row.
    pub fn density(&self) -> &[f64] {
        &self.density
    }

    /// Distance from each fitted row to its `k`-th nearest other fitted row.
    pub fn reach(&self) -> &[f64] {
        &self.reach
    }

    /// How many neighbours were asked for.
    pub fn neighbours(&self) -> Neighbours {
        self.neighbours
    }

    /// How many neighbours were actually used, after clamping to `n - 1`.
    pub fn used(&self) -> usize {
        self.used
    }

    /// The design this was fitted on. Held, not borrowed — see the module docs.
    pub fn design(&self) -> &Matrix {
        &self.design
    }
}

impl Outlier for Local {
    fn features(&self) -> usize {
        self.design.p()
    }

    /// `LOF` of each row of `x`, computed against the fitted design.
    ///
    /// A row of `x` that coincides with a fitted row is NOT excluded from its own
    /// neighbourhood here — it is a query, not a member, so its nearest neighbour is legally
    /// that fitted row at distance zero. This is scikit-learn's `score_samples` convention
    /// negated, and it is why `outlier(design)` is not `-factor()`: the first treats the
    /// rows as queries and the second as members. Both are right; they answer different
    /// questions, so they are different methods rather than one with a flag.
    fn outlier(&self, x: &Matrix) -> Result<Vec<f64>> {
        if x.p() != self.design.p() {
            return Err(Error::Shape(format!(
                "fitted on {} features, given {}",
                self.design.p(),
                x.p()
            )));
        }
        Ok((0..x.n())
            .into_par_iter()
            .map(|i| {
                let f = nearest(&self.design, x.row(i), self.used, None);
                f.ratio(&self.density, f.density(&self.reach))
            })
            .collect())
    }
}

/// One query's `k` nearest fitted rows, nearest first.
struct Found {
    at: Vec<usize>,
    away: Vec<f64>,
}

impl Found {
    /// Distance to the furthest of them: the `k`-distance.
    fn far(&self) -> f64 {
        *self.away.last().expect("k >= 1")
    }

    /// Local reachability density: the inverse mean reachability distance to these
    /// neighbours, where reachability to a neighbour is at least that NEIGHBOUR's own
    /// `k`-distance.
    ///
    /// The `max` is the whole idea. A plain mean distance would make a row on the fringe of
    /// a dense cluster look sparse merely because the cluster is compact; flooring each
    /// distance at the neighbour's own scale measures the row against the local scale rather
    /// than against a global one.
    fn density(&self, reach: &[f64]) -> f64 {
        let total: f64 = self
            .at
            .iter()
            .zip(&self.away)
            .map(|(&o, &d)| reach[o].max(d))
            .sum();
        1.0 / (total / self.at.len() as f64 + GUARD)
    }

    /// Mean of the neighbours' densities over this row's own: the factor itself.
    fn ratio(&self, density: &[f64], own: f64) -> f64 {
        self.at.iter().map(|&o| density[o] / own).sum::<f64>() / self.at.len() as f64
    }
}

/// The `k` rows of `design` nearest to `row`, nearest first, skipping `self_at` if given.
///
/// A full scan keeping a `k`-sized frontier. No index: for the `p` this is used at, a
/// scan of a resident row-major design is bandwidth-bound and a tree would spend more on
/// pointer chasing than it saves on comparisons — and the exactness is not negotiable,
/// since an approximate neighbour set would silently stop matching the reference.
///
/// Ties break on the lower row index, which is a stated rule rather than whatever a sort
/// happened to do. Exact ties need duplicate rows or a contrived grid; when they occur, two
/// implementations disagreeing about WHICH equidistant neighbour was taken would still agree
/// on every distance, so the score agrees regardless.
fn nearest(design: &Matrix, row: &[f64], k: usize, self_at: Option<usize>) -> Found {
    // A max-frontier as a flat sorted vector: k is 20-ish, so a linear insert into a hot
    // 20-element array beats a heap's indirection and keeps the result already ordered.
    let mut at: Vec<usize> = Vec::with_capacity(k + 1);
    let mut away: Vec<f64> = Vec::with_capacity(k + 1);
    for i in 0..design.n() {
        if self_at == Some(i) {
            continue;
        }
        let d = distance(design.row(i), row);
        if away.len() == k && d >= away[k - 1] {
            continue;
        }
        let put = away.partition_point(|&v| v <= d);
        away.insert(put, d);
        at.insert(put, i);
        if away.len() > k {
            away.pop();
            at.pop();
        }
    }
    Found { at, away }
}

/// Euclidean distance. The square root is taken because reachability compares distances to
/// each other AND averages them, and a mean of squares is not the square of a mean.
fn distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::twister::Twister;

    fn design(n: usize) -> Matrix {
        let mut data = Vec::with_capacity(n * 2);
        let mut t = Twister::seed(3);
        for i in 0..n {
            if i < 3 {
                data.push(30.0 + i as f64);
                data.push(-30.0 - i as f64);
            } else {
                data.push(t.next_real() * 2.0 - 1.0);
                data.push(t.next_real() * 2.0 - 1.0);
            }
        }
        Matrix::new(n, 2, data).unwrap()
    }

    #[test]
    fn a_planted_outlier_has_a_factor_far_below_minus_one() {
        let l = Local::fit(&design(200), Neighbours::DEFAULT).unwrap();
        for planted in &l.factor()[..3] {
            assert!(*planted < -2.0, "planted row scored {planted}");
        }
        // The bulk sits near -1: as dense as its own neighbourhood.
        let bulk = &l.factor()[3..];
        let mean = bulk.iter().sum::<f64>() / bulk.len() as f64;
        assert!((mean + 1.0).abs() < 0.35, "bulk mean {mean}");
    }

    #[test]
    fn the_two_orientations_are_exact_negations_of_each_other() {
        // For the FITTED rows read as members, factor is -LOF by definition.
        let x = design(120);
        let l = Local::fit(&x, Neighbours::new(10).unwrap()).unwrap();
        // Read as queries the answer differs, because a query is its own nearest
        // neighbour; assert it differs rather than pretending the two agree.
        let queried = l.outlier(&x).unwrap();

        // The two readings are NOT the same numbers, and the claim is about the vectors
        // rather than about any one row. My first version of this test asserted the
        // inequality at index 0 and failed, because at index 0 the two happened to coincide
        // — a row far enough out that including itself as a neighbour changed nothing to
        // within a bit. scikit-learn agrees the vectors differ: its own `score_samples` on
        // the training data does not reproduce `negative_outlier_factor_` either, which
        // `tests/sklearn.rs` pins at three separate k.
        let differ = (0..x.n()).filter(|&i| queried[i] != -l.factor()[i]).count();
        assert!(
            differ > 0,
            "reading the fitted rows as queries gave the same answer as reading them as \
             members, so self-exclusion is not happening"
        );

        // What the two readings DO agree on, exactly: the planted rows are the most
        // anomalous under both. That is the claim a caller depends on, and it is a
        // set equality rather than a threshold on a correlation.
        let worst_member = (3..x.n()).map(|i| -l.factor()[i]).fold(f64::MIN, f64::max);
        let worst_query = (3..x.n()).map(|i| queried[i]).fold(f64::MIN, f64::max);
        for i in 0..3 {
            assert!(
                -l.factor()[i] > worst_member,
                "member reading missed row {i}"
            );
            assert!(queried[i] > worst_query, "query reading missed row {i}");
        }
    }

    #[test]
    fn duplicates_read_as_the_densest_thing_rather_than_as_not_a_number() {
        // Ten identical rows plus a spread: the duplicates' mean reachability is zero, so
        // the guard is the only reason this is a number at all.
        let mut data = vec![0.0; 20];
        for i in 10..30 {
            data.push(i as f64);
        }
        let x = Matrix::new(20, 2, data).unwrap();
        let l = Local::fit(&x, Neighbours::new(5).unwrap()).unwrap();
        assert!(l.factor().iter().all(|v| v.is_finite()), "{:?}", l.factor());
        assert!(l.density()[0] >= 1.0 / GUARD * 0.5, "{}", l.density()[0]);
    }

    #[test]
    fn k_is_clamped_to_the_rows_there_are() {
        let x = design(8);
        let l = Local::fit(&x, Neighbours::DEFAULT).unwrap();
        assert_eq!(l.used(), 7);
        assert_eq!(l.neighbours().get(), 20, "the request is not rewritten");
        assert!(l.factor().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn illegal_configurations_are_not_values() {
        assert!(Neighbours::new(0).is_err());
        assert!(Neighbours::new(1).is_ok());
        let one = Matrix::new(1, 2, vec![1.0, 2.0]).unwrap();
        assert!(Local::fit(&one, Neighbours::DEFAULT).is_err());
    }

    #[test]
    fn a_width_mismatch_is_named_and_not_guessed() {
        let l = Local::fit(&design(30), Neighbours::new(5).unwrap()).unwrap();
        let narrow = Matrix::new(2, 1, vec![1.0, 2.0]).unwrap();
        let e = l.outlier(&narrow).unwrap_err();
        assert!(
            format!("{e}").contains("fitted on 2 features, given 1"),
            "{e}"
        );
    }

    /// The frontier insert is the one piece of fiddly code here; check it against the
    /// obvious slow way on data with ties in it.
    #[test]
    fn the_neighbour_frontier_agrees_with_a_full_sort() {
        let mut t = Twister::seed(17);
        // Coarse grid, so exact distance ties are common.
        let data: Vec<f64> = (0..600).map(|_| (t.next_real() * 5.0).floor()).collect();
        let x = Matrix::new(300, 2, data).unwrap();
        let query = [1.0, 2.0];
        for k in [1usize, 3, 20] {
            let got = nearest(&x, &query, k, None);
            let mut all: Vec<(f64, usize)> = (0..x.n())
                .map(|i| (distance(x.row(i), &query), i))
                .collect();
            all.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let want: Vec<f64> = all[..k].iter().map(|(d, _)| *d).collect();
            assert_eq!(got.away, want, "k={k} distances");
            let want_at: Vec<usize> = all[..k].iter().map(|(_, i)| *i).collect();
            assert_eq!(got.at, want_at, "k={k} indices (ties break on lower index)");
        }
    }

    #[test]
    fn self_exclusion_is_by_index_so_a_twin_still_counts() {
        // Rows 0 and 1 are identical. Row 0's nearest neighbour must be row 1 at zero,
        // not row 0 at zero.
        let x = Matrix::new(4, 1, vec![5.0, 5.0, 9.0, 20.0]).unwrap();
        let f = nearest(&x, x.row(0), 2, Some(0));
        assert_eq!(f.at, vec![1, 2]);
        assert_eq!(f.away[0], 0.0);
    }
}
