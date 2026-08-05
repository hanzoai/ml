//! Isolating a point rather than modelling the crowd.
//!
//! An isolation forest asks how few random cuts it takes to separate a row from the rest.
//! Anomalies fall out early, so a SHORT expected path is the evidence. The estimator is
//! therefore a depth measurement, normalised so that forests grown on different sample
//! sizes are comparable, and nothing else.
//!
//! # Where the threshold went
//!
//! scikit-learn's `IsolationForest` takes a `contamination` and reports `predict` as ±1.
//! That braids two decisions together: how anomalous each row is, which the data settles,
//! and how much alarm the operator can afford, which the data cannot. Here the fitted value
//! answers only the first. The second belongs to [`crate::metric::Curve`], where
//! [`crate::metric::Curve::cheapest`] picks the threshold from the cost of a false alarm
//! against the cost of a miss — the terms a risk desk actually knows. So there is no
//! `contamination` on [`Config`], no `offset_`, and no `predict`.
//!
//! # Scale, concretely
//!
//! Fitting BORROWS a [`Matrix`] and keeps none of it. Each tree reads a subsample of
//! [`Config::sample`] rows — 256 by default, whatever `n` is — so the fitted value is
//! `O(trees · sample)` and INDEPENDENT of `n`: about 1.6 MB for the default forest, at
//! `n = 10³` or `n = 10⁹`. Fitting is one pass to draw the subsample per tree, so a design
//! too large to hold can be fitted from a reservoir sample without the forest changing
//! shape. Scoring is `O(trees · depth)` per row with no allocation per row.
//!
//! Single device, CPU, `rayon` across trees. There is no GPU path: a tree descent is a
//! dependent chain of unpredictable branches, which is the one shape a wide machine cannot
//! help with. The measured comparison is in `oracle/bench.py`.
//!
//! Clean-room from the published algorithm (Liu, Ting and Zhou 2008, *Isolation Forest*),
//! with the normalisation and the leaf correction matched to scikit-learn's arithmetic.

use rayon::prelude::*;

use crate::twister::Twister;
use crate::{Error, Matrix, Outlier, Result};

/// The Euler–Mascheroni constant, to the bit `numpy` carries it as `euler_gamma`.
const EULER: f64 = 0.577_215_664_901_532_9;

/// The expected path length to an unsuccessful search in a binary search tree over `m`
/// points: Liu's `c(m)`.
///
/// This is the term that makes depths comparable. A leaf holding `m > 1` points was not
/// grown further — either the depth limit stopped it or the points are identical — so the
/// depth reached UNDERSTATES how long isolating one of them would have taken. `c(m)` is the
/// expected remainder, and adding it back is what lets one forest's depths be read against
/// another's.
///
/// Exact at the boundaries: a leaf of one point has nothing left to isolate, and a leaf of
/// two takes exactly one more cut.
pub fn average_path(m: u64) -> f64 {
    match m {
        0 | 1 => 0.0,
        2 => 1.0,
        m => {
            let m = m as f64;
            2.0 * ((m - 1.0).ln() + EULER) - 2.0 * (m - 1.0) / m
        }
    }
}

/// One node of one tree.
///
/// A leaf carries a number and no test; a split carries a test and no number. Neither can
/// hold the other's fields, so "a leaf whose threshold was read" is not a state that exists.
#[derive(Debug, Clone, PartialEq)]
enum Node {
    /// This leaf's depth plus [`average_path`] of the rows that reached it — the whole
    /// contribution of this tree, precomputed at fit time so scoring is one lookup.
    Leaf { credit: f64 },
    /// Rows with `x[feature] <= threshold` go left.
    Split {
        feature: u32,
        threshold: f64,
        left: u32,
        right: u32,
    },
}

/// One tree: nodes in one vector, the root at zero.
#[derive(Debug, Clone, PartialEq)]
struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    /// Grow one tree over `rows`, which this consumes as scratch: the partition is done in
    /// place, so the whole build allocates the node vector and nothing else per node.
    fn grow(x: &Matrix, rows: &mut [usize], limit: u32, rng: &mut Twister) -> Self {
        let mut nodes = Vec::new();
        split(x, rows, 0, limit, &mut nodes, rng);
        Self { nodes }
    }

    /// The path length this tree assigns to one row.
    fn credit(&self, row: &[f64]) -> f64 {
        let mut at = 0usize;
        loop {
            match self.nodes[at] {
                Node::Leaf { credit } => return credit,
                Node::Split {
                    feature,
                    threshold,
                    left,
                    right,
                } => {
                    at = if row[feature as usize] <= threshold {
                        left as usize
                    } else {
                        right as usize
                    }
                }
            }
        }
    }
}

/// Grow the subtree over `rows`, returning its node index.
///
/// # Why a split can never be degenerate
///
/// The threshold is drawn in `[low, high)` — closed below, OPEN above — and rows go left on
/// `<=`. So the row holding `low` always goes left and the row holding `high` always goes
/// right: both children are non-empty, every recursion strictly shrinks, and there is no
/// need for the retry loop that a `[low, high]` draw would require. A node whose chosen
/// feature is constant has `low == high` and becomes a leaf instead, which is the only way
/// the split can decline.
fn split(
    x: &Matrix,
    rows: &mut [usize],
    depth: u32,
    limit: u32,
    nodes: &mut Vec<Node>,
    rng: &mut Twister,
) -> u32 {
    let me = nodes.len() as u32;
    let leaf = |nodes: &mut Vec<Node>| {
        nodes.push(Node::Leaf {
            credit: depth as f64 + average_path(rows.len() as u64),
        });
        me
    };
    if rows.len() <= 1 || depth >= limit {
        return leaf(nodes);
    }
    let feature = rng.below(x.p() as u64 - 1) as usize;
    let (low, high) = rows.iter().fold((f64::MAX, f64::MIN), |(lo, hi), &i| {
        let v = x.at(i, feature);
        (lo.min(v), hi.max(v))
    });
    if low >= high {
        return leaf(nodes);
    }
    let threshold = low + rng.next_real() * (high - low);
    let cut = partition(x, rows, feature, threshold);

    nodes.push(Node::Leaf { credit: 0.0 }); // reserved; overwritten once children exist
    let (left_rows, right_rows) = rows.split_at_mut(cut);
    let left = split(x, left_rows, depth + 1, limit, nodes, rng);
    let right = split(x, right_rows, depth + 1, limit, nodes, rng);
    nodes[me as usize] = Node::Split {
        feature: feature as u32,
        threshold,
        left,
        right,
    };
    me
}

/// Move rows with `x[feature] <= threshold` to the front, returning how many there are.
fn partition(x: &Matrix, rows: &mut [usize], feature: usize, threshold: f64) -> usize {
    let mut cut = 0;
    for at in 0..rows.len() {
        if x.at(rows[at], feature) <= threshold {
            rows.swap(cut, at);
            cut += 1;
        }
    }
    cut
}

/// How to grow a forest.
///
/// A plain value with no learned state and no `outlier`, so it is not a thing that can be
/// asked a question. [`Config::fit`] is the only way to get something that can.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    trees: usize,
    sample: usize,
    seed: u32,
}

impl Default for Config {
    /// A hundred trees of 256 rows under seed zero: Liu's own recommendation, and
    /// scikit-learn's default.
    ///
    /// 256 is not a rounding of `n`. The paper's finding is that isolation depth SATURATES:
    /// past a few hundred rows a bigger subsample buys resolution on the crowd, which is
    /// the part nobody is asking about, while making the anomalies harder to cut out early.
    fn default() -> Self {
        Self {
            trees: 100,
            sample: 256,
            seed: 0,
        }
    }
}

impl Config {
    /// A forest of `trees` trees, each grown on `sample` rows, under `seed`.
    ///
    /// Fails on zero of either, because a forest with no trees has no depth to average and
    /// a tree with no rows has nothing to isolate.
    pub fn new(trees: usize, sample: usize, seed: u32) -> Result<Self> {
        if trees == 0 || sample == 0 {
            return Err(Error::Config(format!(
                "a forest needs at least one tree over at least one row, not {trees} over \
                 {sample}"
            )));
        }
        Ok(Self {
            trees,
            sample,
            seed,
        })
    }

    /// How many trees will be grown.
    pub fn trees(&self) -> usize {
        self.trees
    }

    /// How many rows each tree will see, before clamping to the design.
    pub fn sample(&self) -> usize {
        self.sample
    }

    /// The seed the subsamples and the cuts are drawn under.
    pub fn seed(&self) -> u32 {
        self.seed
    }

    /// Grow the forest.
    ///
    /// Each tree gets its own seeded stream, derived from [`Config::seed`] and the tree's
    /// index, so the forest is reproducible AND the trees are independent of the order they
    /// are grown in. That is what lets the growth run over every core and still answer the
    /// same as a serial fit, bit for bit — the tree that saw stream `k` is the same tree
    /// whichever thread grew it.
    pub fn fit(&self, x: &Matrix) -> Result<Forest> {
        let sample = self.sample.min(x.n());
        let limit = (sample.max(2) as f64).log2().ceil() as u32;
        let trees = (0..self.trees)
            .into_par_iter()
            .map(|k| {
                let mut rng = Twister::seed(self.seed.wrapping_add(k as u32));
                // `subsample` and not `permutation(n).truncate(sample)`: the subsample is
                // 256 rows whatever `n` is, so drawing all of `n` would make the FIT `O(n)`
                // per tree in both time and memory — 8 MB and a million swaps per tree at
                // n = 10⁶, on every core at once. This is what makes the cost claim in the
                // module docs true rather than aspirational.
                let mut rows = rng.subsample(x.n(), sample);
                Tree::grow(x, &mut rows, limit, &mut rng)
            })
            .collect();
        Ok(Forest {
            trees,
            features: x.p(),
            denominator: self.trees as f64 * average_path(sample as u64),
            sample,
            config: *self,
        })
    }
}

/// A grown forest: the fitted value.
///
/// Immutable, `O(trees · sample)` in size whatever `n` was, and obtainable only from
/// [`Config::fit`].
#[derive(Debug, Clone, PartialEq)]
pub struct Forest {
    trees: Vec<Tree>,
    features: usize,
    denominator: f64,
    sample: usize,
    config: Config,
}

impl Forest {
    /// The config this was grown under.
    pub fn config(&self) -> Config {
        self.config
    }

    /// How many rows each tree actually saw, after clamping to the design.
    pub fn sample(&self) -> usize {
        self.sample
    }

    /// How many trees there are.
    pub fn trees(&self) -> usize {
        self.trees.len()
    }

    /// The mean path length over the forest for each row, before normalisation.
    ///
    /// Exposed because it is the quantity in Liu's paper and the one to reach for when
    /// asking WHY a row scored as it did: a depth is interpretable in cuts, where the
    /// normalised score is not.
    pub fn depth(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.width(x)?;
        Ok((0..x.n())
            .map(|i| {
                let row = x.row(i);
                self.trees.iter().map(|t| t.credit(row)).sum::<f64>() / self.trees.len() as f64
            })
            .collect())
    }

    fn width(&self, x: &Matrix) -> Result<()> {
        if x.p() != self.features {
            return Err(Error::Shape(format!(
                "fitted on {} features, given {}",
                self.features,
                x.p()
            )));
        }
        Ok(())
    }
}

impl Outlier for Forest {
    fn features(&self) -> usize {
        self.features
    }

    /// `2^(-E[h(x)] / c(sample))`, which is Liu's `s(x)` and the negation of
    /// scikit-learn's `score_samples`.
    fn outlier(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.width(x)?;
        Ok((0..x.n())
            .map(|i| {
                let row = x.row(i);
                let total: f64 = self.trees.iter().map(|t| t.credit(row)).sum();
                (-total / self.denominator).exp2()
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `sklearn.ensemble._iforest._average_path_length`, evaluated by scikit-learn 1.9.0.
    /// Pins the boundaries and the constant independently of any forest.
    #[test]
    fn the_path_length_correction_matches_scikit_learn() {
        let want = [
            (0u64, 0.0),
            (1, 0.0),
            (2, 1.0),
            (3, 1.207_392_357_589_623),
            (10, 3.748_880_484_475_505),
            (256, 10.244_770_920_119_917),
            (1000, 12.969_940_887_100_174),
            (100_000, 22.180_282_259_643_523),
        ];
        for (m, c) in want {
            let got = average_path(m);
            assert!((got - c).abs() < 1e-12, "c({m}) = {got}, want {c}");
        }
    }

    fn design(n: usize) -> Matrix {
        // A dense blob with three points thrown far out of it.
        let mut data = Vec::with_capacity(n * 2);
        let mut t = Twister::seed(1);
        for i in 0..n {
            if i < 3 {
                data.push(40.0 + i as f64);
                data.push(-35.0 - i as f64);
            } else {
                data.push(t.next_real() * 2.0 - 1.0);
                data.push(t.next_real() * 2.0 - 1.0);
            }
        }
        Matrix::new(n, 2, data).unwrap()
    }

    #[test]
    fn the_planted_outliers_score_above_every_inlier() {
        let x = design(400);
        let f = Config::default().fit(&x).unwrap();
        let s = f.outlier(&x).unwrap();
        let worst_inlier = s[3..].iter().cloned().fold(f64::MIN, f64::max);
        for (i, planted) in s[..3].iter().enumerate() {
            assert!(
                *planted > worst_inlier,
                "planted {i} scored {planted}, below the worst inlier {worst_inlier}"
            );
        }
    }

    #[test]
    fn a_score_is_a_probability_shaped_number_and_a_depth_is_in_cuts() {
        let x = design(300);
        let f = Config::default().fit(&x).unwrap();
        for s in f.outlier(&x).unwrap() {
            assert!((0.0..=1.0).contains(&s), "{s}");
        }
        // The depth limit is ceil(log2(256)) = 8, and a leaf adds at most c(256).
        for d in f.depth(&x).unwrap() {
            assert!(d > 0.0 && d < 8.0 + average_path(256), "{d}");
        }
    }

    #[test]
    fn the_forest_is_reproducible_and_the_seed_changes_it() {
        let x = design(200);
        let a = Config::default().fit(&x).unwrap();
        let b = Config::default().fit(&x).unwrap();
        assert_eq!(a, b, "the same seed grew a different forest");
        let c = Config::new(100, 256, 9).unwrap().fit(&x).unwrap();
        assert_ne!(a.outlier(&x).unwrap(), c.outlier(&x).unwrap());
    }

    /// The growth runs over `rayon`; the answer must not depend on how many threads it got.
    #[test]
    fn a_parallel_fit_is_bit_identical_to_a_one_thread_fit() {
        let x = design(500);
        let whole = Config::default().fit(&x).unwrap();
        let serial = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| Config::default().fit(&x).unwrap());
        assert_eq!(whole, serial);
        assert_eq!(whole.outlier(&x).unwrap(), serial.outlier(&x).unwrap());
    }

    #[test]
    fn a_sample_larger_than_the_design_clamps_rather_than_failing() {
        let x = design(40);
        let f = Config::default().fit(&x).unwrap();
        assert_eq!(f.sample(), 40);
        assert_eq!(f.config().sample(), 256, "the config is not rewritten");
    }

    #[test]
    fn a_width_mismatch_is_named_and_not_guessed() {
        let f = Config::default().fit(&design(50)).unwrap();
        let narrow = Matrix::new(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
        let e = f.outlier(&narrow).unwrap_err();
        assert!(
            format!("{e}").contains("fitted on 2 features, given 1"),
            "{e}"
        );
        assert!(f.depth(&narrow).is_err());
    }

    #[test]
    fn an_empty_forest_is_not_a_value() {
        assert!(Config::new(0, 256, 0).is_err());
        assert!(Config::new(100, 0, 0).is_err());
        assert!(Config::new(1, 1, 0).is_ok());
    }

    /// A constant column cannot be cut, so the tree must decline the split rather than
    /// draw a threshold equal to the only value there is and recurse forever.
    #[test]
    fn a_constant_design_terminates_and_scores_everything_alike() {
        let x = Matrix::new(50, 2, vec![7.0; 100]).unwrap();
        let f = Config::default().fit(&x).unwrap();
        let s = f.outlier(&x).unwrap();
        assert!(s.iter().all(|v| (v - s[0]).abs() < 1e-15));
    }
}
