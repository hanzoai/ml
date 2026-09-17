//! Regression trees: the base learner, and the irregular half of this crate.
//!
//! A CART regression tree, greedy, exhaustive over every feature and every distinct
//! value. Its split rule matches scikit-learn's `squared_error` splitter term for term
//! — including the two conventions that look like details and are not: the near-equal
//! grouping tolerance, and the exact arithmetic of the threshold midpoint.
//!
//! # A leaf has no threshold, and that is in the type
//!
//! scikit-learn stores one flat array per field and writes `-2` into `feature` and
//! `threshold` at leaves. That is an illegal state made representable: `-2` is a
//! perfectly good feature index in a language without sum types, and every consumer
//! must know the sentinel. [`Node`] is an enum, so a leaf HAS no threshold to read and
//! a split HAS no value to mistake for a prediction.
//!
//! # THE FEATURE SPACE IS SINGLE PRECISION, AND THAT IS THE ALGORITHM
//!
//! A tree never does arithmetic on a feature value. It only ORDERS values and places
//! thresholds BETWEEN them, so the only thing a feature's precision decides is which
//! side of a boundary a sample falls on. Every serious tree implementation therefore
//! reads features at binary32 — scikit-learn casts `X` to `float32` before its splitter
//! sees it, XGBoost and LightGBM do the same, and `ai.onnx.ml`'s `TreeEnsembleRegressor`
//! stores `nodes_values` as `floats`, which IS binary32. Single precision is the
//! interchange format of this model family, not a scikit-learn quirk.
//!
//! So [`single`] is applied wherever a feature value is read, and the consequence is
//! measurable. From this crate's own fixture, the two values straddling the root split of
//! the first boosted tree:
//!
//! ```text
//!   a  = -1.5903684017801702      b = -1.5771856983463417
//!   a/2 + b/2                  = -1.583777050063256      <- double precision
//!   single(a)/2 + single(b)/2  = -1.5837770700454712      <- scikit-learn records THIS
//! ```
//!
//! Those differ by 2e-8 — nowhere near a rounding difference, and enough to send a
//! sample down the other branch. Measured over the 40 depth-1 trees of the `stumps`
//! fixture, whose roots all see the full sample set: the single-precision midpoint
//! reproduces scikit-learn's threshold bit-for-bit 40 times out of 40, and the
//! double-precision midpoint 0 times out of 40. That is why `tests/sklearn.rs` can
//! assert thresholds with `==` and not a tolerance.
//!
//! The targets stay `f64`. Only the feature side is single, because only the feature
//! side is ordered rather than summed — leaf values are means of `y` and want every bit.
//!
//! # WHERE THIS DIVERGES FROM SCIKIT-LEARN, DELIBERATELY
//!
//! Ties are broken by the LOWEST FEATURE INDEX, always. scikit-learn breaks them with a
//! `random_state`-seeded partial Fisher-Yates shuffle over the features, which means a
//! scikit-learn tree IS NOT A FUNCTION OF ITS TRAINING DATA at the default
//! `min_samples_leaf = 1`:
//!
//! ```text
//!   DecisionTreeRegressor(max_depth=3)          20 structures over 40 seeds,
//!                                               prediction spread 3.9e+00
//!   GradientBoostingRegressor(max_depth=5)      12 structures over 12 seeds,
//!                                               prediction spread 1.1e+00
//! ```
//!
//! measured on this crate's own fixture data. That spread is O(1) on a target whose own
//! spread is O(1) — a DIFFERENT MODEL, not a different rounding.
//!
//! The tie is not exotic, it is structural: at `min_samples_leaf = 1` the recursion
//! reaches nodes holding two samples, and EVERY feature splits two samples into the
//! same two singletons, so every feature earns an identical score and the shuffle
//! decides. Deeper trees reach more such nodes.
//!
//! Determinism is not a nicety here, it is what makes [`crate::Address`] mean anything:
//! a content address over a model that a hidden RNG helped choose names one draw, not
//! one fit. So the tie-break is the lowest index, the fit is a pure function of the
//! data, and the same data always names the same model.
//!
//! This is why the fixture pins a leaf bound on every boosted case and PROVES
//! scikit-learn deterministic there before recording it: at the defaults there is no
//! well-defined answer to match.
//!
//! # WHERE A GPU DSL DOES NOT REACH — the honest report
//!
//! The split search is `argmax` over `features x candidate positions` of a score built
//! from prefix sums, which IS data-parallel and would lower to `hanzo-kernel` cleanly.
//! The RECURSION around it does not:
//!
//!   * The work at each node depends on a partition computed by the node above it, so the
//!     tree's shape is not known until it is built — and a launch geometry has to be
//!     stated BEFORE the launch. This is not a guess about the DSL, it is what the DSL
//!     is: `hanzo-kernel` exposes the launch shape as `Grid` (`hanzo-kernel/src/lib.rs:35`,
//!     re-exporting `cubecl::CubeCount`), and across all 69 launch sites in that crate
//!     `Grid::Static(..)` is the ONLY variant that appears — every geometry is computed on
//!     the host from a count it already knows, e.g. `Grid::Static(rows as u32, 1, 1)` in
//!     `hanzo-kernel/src/norm.rs:248`. A CART node's sample count is not such a number.
//!     Dispatching per node would need either a device-side launch (CUDA dynamic
//!     parallelism, which ROCm, Metal and Vulkan do not offer alike, so a DSL that lowers
//!     to all four cannot expose it) or a host round trip per tree level, which serialises
//!     the build behind one synchronisation per level.
//!   * The subproblems are wildly unbalanced. A node holding 3 samples and one holding
//!     3000 are the same dispatch, so a warp-per-node scheme idles almost all of it.
//!   * Every node needs its samples ordered by the candidate feature. Sorting a
//!     shrinking, irregular subset per node is the actual cost, and it is a
//!     permutation-heavy scatter rather than arithmetic.
//!
//! The tractable lowering is the HISTOGRAM formulation — bin the features once, then a
//! node's split search is a fixed-size reduction over `bins x features` and the recursion
//! only reads histograms. That is exactly what dissolves the obstacle above: the bin count
//! is a HYPERPARAMETER, so `Grid::Static(nodes_at_level * features, 1, 1)` over a fixed
//! `bins`-wide reduction is host-knowable before the level runs, and one launch per level
//! replaces one per node. It is also why `HistGradientBoosting` is the family that gets
//! accelerated everywhere and exhaustive CART is not.
//!
//! But it is a DIFFERENT ALGORITHM with different answers — binning quantises the
//! thresholds, so it cannot reproduce the node-for-node agreement this module is held to.
//! It therefore belongs behind its own config rather than silently under this one. It is
//! NOT built. Naming the obstacle precisely is the finding; a slow port pretending to be
//! accelerated would not be.

use crate::data::Matrix;
use crate::error::{Error, Result};

/// A feature value as a tree reads it: rounded to IEEE-754 binary32.
///
/// ONE function, so the crate has exactly one answer to "what does the splitter see".
/// Applied when ordering values, when grouping near-equal ones, when placing a threshold
/// between two of them, and when walking a fitted tree — the four places a feature value
/// is read. Anywhere it were skipped, a fit and a prediction would disagree about which
/// side of a boundary a sample lies on, which is the one thing a tree must never be
/// vague about. Returned as `f64` because the midpoint of two binary32 values needs one
/// more mantissa bit than binary32 has; the threshold is genuinely double.
///
/// See the module header for the measurement that forces this.
fn single(v: f64) -> f64 {
    v as f32 as f64
}

/// The tolerance for calling two feature values the same, in the single-precision space
/// [`single`] puts them in. Values within it are grouped and never split between.
///
/// It is 1e-7 because that is scikit-learn's own constant, and its size is not arbitrary:
/// binary32 carries ~7 decimal digits, so 1e-7 is the scale at which "these are two
/// values" and "these are one value the cast happened to separate" stop being
/// distinguishable.
const FEATURE_THRESHOLD: f64 = 1e-7;

/// A node's impurity at or below this is treated as pure. scikit-learn uses the same
/// constant for the same purpose, and it matters: without it a node of identical
/// targets splits forever on rounding noise.
const PURE: f64 = f64::EPSILON;

/// Below this many samples the split search runs sequentially. Thread dispatch costs
/// more than the scan at small `n`, and a boosted ensemble spends most of its nodes
/// there. Both paths reduce in ascending feature order, so the answer does not depend
/// on which one ran.
const PARALLEL_FLOOR: usize = 512;

/// How to grow one tree.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Config {
    /// Longest root-to-leaf path. Depth 1 is a single split.
    pub depth: usize,

    /// Fewest samples a node may hold and still be considered for splitting.
    pub min_split: usize,

    /// Fewest samples either side of a split may hold.
    ///
    /// Raising this above 1 is what makes a fit REPRODUCIBLE — see the module header.
    /// It is not only a regularisation knob.
    pub min_leaf: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            depth: 3,
            min_split: 2,
            min_leaf: 1,
        }
    }
}

impl Config {
    fn validate(&self) -> Result<()> {
        if self.depth == 0 {
            return Err(Error::Config(
                "depth 0 is a tree with no splits; use depth 1 for a single split".into(),
            ));
        }
        if self.min_leaf == 0 || self.min_split == 0 {
            return Err(Error::Config(
                "min_leaf and min_split count samples and must be at least 1".into(),
            ));
        }
        Ok(())
    }
}

/// One node. A split or a leaf, never both and never neither.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Node {
    /// Go left when `row[feature] <= threshold`.
    Split {
        /// Which feature is tested.
        feature: usize,
        /// The value it is tested against.
        threshold: f64,
        /// Index of the `<=` child.
        left: usize,
        /// Index of the `>` child.
        right: usize,
    },
    /// What this region answers.
    Leaf {
        /// The mean of the targets that reached it.
        value: f64,
    },
}

/// A fitted regression tree: a VALUE.
///
/// Nodes are in pre-order, left subtree before right — the same order scikit-learn's
/// depth-first builder emits, so a node-for-node comparison against its arrays needs no
/// remapping.
#[derive(Clone, PartialEq, Debug)]
pub struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    /// The nodes, pre-order.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// What this tree answers for one row.
    ///
    /// Total: the walk starts at a node that exists and every `Split` names children
    /// that exist, both established when the tree was built.
    pub fn value(&self, row: &[f64]) -> f64 {
        let mut at = 0;
        loop {
            match self.nodes[at] {
                Node::Leaf { value } => return value,
                Node::Split {
                    feature,
                    threshold,
                    left,
                    right,
                    // `single` here for the same reason it is in the split search: the
                    // threshold was placed between two single-precision values, so the
                    // comparison must happen in that space or a row whose f64 and f32
                    // values straddle the boundary would predict from the wrong leaf.
                } => {
                    at = if single(row[feature]) <= threshold {
                        left
                    } else {
                        right
                    }
                }
            }
        }
    }

    /// Fold this tree's shape and parameters into a model's name.
    pub(crate) fn digest(&self, d: crate::Digest) -> crate::Digest {
        let mut d = d.size(self.nodes.len());
        for node in &self.nodes {
            d = match *node {
                // The discriminant is written, so a leaf and a split can never hash
                // alike however their numbers line up.
                Node::Leaf { value } => d.flag(false).real(value),
                Node::Split {
                    feature,
                    threshold,
                    left,
                    right,
                } => d
                    .flag(true)
                    .size(feature)
                    .real(threshold)
                    .size(left)
                    .size(right),
            };
        }
        d
    }
}

/// Grow one tree against arbitrary targets.
///
/// `y` is taken as a plain slice rather than as [`crate::Samples`] because the boosting
/// loop above fits each tree to a RESIDUAL it just computed, which is not the caller's
/// data and has no business being wrapped as though it were.
pub(crate) fn grow(x: &Matrix, y: &[f64], config: &Config) -> Result<Tree> {
    config.validate()?;
    // The feature space is binary32, so a value beyond binary32's range becomes an
    // infinity the instant the splitter reads it, and every ordering, grouping and
    // threshold below would then be arithmetic over infinities. Refused here, naming the
    // cell, rather than answered from a tree built on them.
    //
    // `Matrix` is the wrong place for this check even though it is where finiteness is
    // established: `linear` and `logistic` work in double precision and are right to
    // accept a 1e300 feature. The narrower domain belongs to the estimator that has it.
    for i in 0..x.n() {
        for (j, v) in x.row(i).iter().enumerate() {
            if !single(*v).is_finite() {
                return Err(Error::Shape(format!(
                    "value at row {i} column {j} is {v:e}, outside binary32's range; \
                     a tree reads features at single precision"
                )));
            }
        }
    }
    let mut nodes = Vec::new();
    let mut index: Vec<usize> = (0..x.n()).collect();
    build(x, y, config, &mut index, 0, &mut nodes);
    Ok(Tree { nodes })
}

/// Emits one node, then its left subtree, then its right — scikit-learn's own order.
///
/// Returns nothing: the node's own index is `nodes.len()` before it is pushed, and the
/// children's indices are only known after they are built, so the slot is reserved and
/// patched. That is why `Node::Split` is written twice.
fn build(
    x: &Matrix,
    y: &[f64],
    config: &Config,
    index: &mut [usize],
    depth: usize,
    nodes: &mut Vec<Node>,
) {
    let n = index.len();
    let total: f64 = index.iter().map(|i| y[*i]).sum();
    let value = total / n as f64;

    // The leaf tests, in scikit-learn's order and with its constant. `impurity <= PURE`
    // is what stops a node of identical targets from splitting on rounding noise.
    let square: f64 = index.iter().map(|i| y[*i] * y[*i]).sum();
    let impurity = square / n as f64 - value * value;
    let bounded = depth >= config.depth
        || n < config.min_split
        || n < 2 * config.min_leaf
        || impurity <= PURE;

    let split = if bounded {
        None
    } else {
        best(x, y, index, config, total)
    };

    let Some(found) = split else {
        nodes.push(Node::Leaf { value });
        return;
    };

    // Reorder this node's samples so the two sides are contiguous, then recurse into
    // the halves. The order within a side is the chosen feature's ascending order,
    // which is also the order the child's own sums accumulate in.
    index.copy_from_slice(&found.order);
    let at = nodes.len();
    nodes.push(Node::Leaf { value });
    let (left_index, right_index) = index.split_at_mut(found.position);

    let left = at + 1;
    build(x, y, config, left_index, depth + 1, nodes);
    let right = nodes.len();
    build(x, y, config, right_index, depth + 1, nodes);

    nodes[at] = Node::Split {
        feature: found.feature,
        threshold: found.threshold,
        left,
        right,
    };
}

/// The winning split at one node, or `None` when no candidate is admissible.
struct Split {
    feature: usize,
    threshold: f64,
    /// Where the left side ends in `order`.
    position: usize,
    /// This node's samples, ascending in `feature`.
    order: Vec<usize>,
    score: f64,
}

/// Searches every feature and returns the best split.
///
/// Ties go to the LOWEST FEATURE INDEX — `>` rather than `>=`, folded in ascending
/// order. That is the whole of the determinism claim in the module header.
fn best(x: &Matrix, y: &[f64], index: &[usize], config: &Config, total: f64) -> Option<Split> {
    let scan = |feature: usize| scan_feature(x, y, index, config, total, feature);

    let candidates: Vec<Option<Split>> = if index.len() >= PARALLEL_FLOOR && x.p() > 1 {
        use rayon::prelude::*;
        (0..x.p()).into_par_iter().map(scan).collect()
    } else {
        (0..x.p()).map(scan).collect()
    };

    candidates
        .into_iter()
        .flatten()
        .fold(None, |best: Option<Split>, next| match best {
            Some(b) if b.score >= next.score => Some(b),
            _ => Some(next),
        })
}

/// The best split on ONE feature.
fn scan_feature(
    x: &Matrix,
    y: &[f64],
    index: &[usize],
    config: &Config,
    total: f64,
    feature: usize,
) -> Option<Split> {
    let n = index.len();
    let mut order: Vec<usize> = index.to_vec();
    // Ascending in this feature. `sort_unstable_by` is safe here: values that compare
    // equal are grouped and never split between, so their relative order cannot reach
    // the answer.
    order.sort_unstable_by(|a, b| {
        single(x.at(*a, feature))
            .partial_cmp(&single(x.at(*b, feature)))
            .expect("grow refuses features that do not fit binary32")
    });

    let low = single(x.at(order[0], feature));
    let high = single(x.at(order[n - 1], feature));
    // A constant feature — within the same tolerance the candidate scan groups by, so a
    // feature that offers no admissible split is skipped rather than scanned.
    if high <= low + FEATURE_THRESHOLD {
        return None;
    }

    // Prefix sums in ascending order, which is the order scikit-learn's criterion
    // accumulates `sum_left` in as it advances the boundary.
    let mut prefix = Vec::with_capacity(n);
    let mut running = 0.0;
    for i in &order {
        running += y[*i];
        prefix.push(running);
    }

    let mut best: Option<Split> = None;
    let mut at = 0usize;
    while at < n {
        // Advance past every value within tolerance of this one, so a candidate is only
        // ever placed between two genuinely distinct values.
        while at + 1 < n
            && single(x.at(order[at + 1], feature))
                <= single(x.at(order[at], feature)) + FEATURE_THRESHOLD
        {
            at += 1;
        }
        at += 1;
        if at >= n {
            break;
        }

        let (left, right) = (at, n - at);
        if left < config.min_leaf || right < config.min_leaf {
            continue;
        }

        // The proxy score for squared error: maximising it is equivalent to minimising
        // the weighted variance of the two sides, with the terms that do not depend on
        // the split dropped. scikit-learn maximises exactly this expression.
        let sum_left = prefix[at - 1];
        let sum_right = total - sum_left;
        let score = sum_left * sum_left / left as f64 + sum_right * sum_right / right as f64;

        if best.as_ref().is_none_or(|b| score > b.score) {
            let (before, after) = (
                single(x.at(order[at - 1], feature)),
                single(x.at(order[at], feature)),
            );
            // `a/2 + b/2`, NOT `(a+b)/2`. The two differ in the last bit and can differ
            // by more when `a+b` overflows toward infinity; scikit-learn writes the
            // former and a threshold that differs in its last bit sends a boundary
            // sample down the other branch. The halves are `f64` over `single` operands,
            // which is exactly what scikit-learn's Cython computes when it divides a
            // `float32` cell by a double literal.
            let mut threshold = before / 2.0 + after / 2.0;
            if threshold == after || !threshold.is_finite() {
                threshold = before;
            }
            best = Some(Split {
                feature,
                threshold,
                position: at,
                order: order.clone(),
                score,
            });
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: &[&[f64]]) -> Matrix {
        Matrix::rows(&rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>()).unwrap()
    }

    #[test]
    fn a_pure_node_does_not_split() {
        let x = matrix(&[&[0.0], &[1.0], &[2.0], &[3.0]]);
        let t = grow(&x, &[5.0; 4], &Config::default()).unwrap();
        assert_eq!(t.nodes(), &[Node::Leaf { value: 5.0 }]);
    }

    #[test]
    fn a_constant_feature_offers_no_split() {
        let x = matrix(&[&[1.0], &[1.0], &[1.0], &[1.0]]);
        let t = grow(&x, &[1.0, 2.0, 3.0, 4.0], &Config::default()).unwrap();
        // Impure targets but nothing to split ON, so one leaf at the mean.
        assert_eq!(t.nodes().len(), 1);
        assert!(matches!(t.nodes()[0], Node::Leaf { value } if (value - 2.5).abs() < 1e-15));
    }

    #[test]
    fn a_single_split_lands_on_the_midpoint_of_the_two_straddling_values() {
        let x = matrix(&[&[0.0], &[1.0], &[2.0], &[3.0]]);
        let t = grow(
            &x,
            &[0.0, 0.0, 10.0, 10.0],
            &Config {
                depth: 1,
                ..Config::default()
            },
        )
        .unwrap();
        match t.nodes()[0] {
            Node::Split {
                feature, threshold, ..
            } => {
                assert_eq!(feature, 0);
                assert_eq!(threshold, 1.0 / 2.0 + 2.0 / 2.0);
            }
            n => panic!("expected a split, got {n:?}"),
        }
        assert_eq!(t.value(&[0.5]), 0.0);
        assert_eq!(t.value(&[2.5]), 10.0);
    }

    #[test]
    fn nodes_are_pre_order_with_the_left_subtree_first() {
        // scikit-learn's own emission order; a node-for-node comparison against its
        // arrays depends on it.
        let x = matrix(&[
            &[0.0],
            &[1.0],
            &[2.0],
            &[3.0],
            &[4.0],
            &[5.0],
            &[6.0],
            &[7.0],
        ]);
        let y = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0];
        let t = grow(
            &x,
            &y,
            &Config {
                depth: 2,
                ..Config::default()
            },
        )
        .unwrap();
        // Root at 0, its left child at 1. Whatever the shape, the node right after a
        // split is that split's left child.
        match t.nodes()[0] {
            Node::Split { left, .. } => assert_eq!(left, 1),
            n => panic!("expected a split at the root, got {n:?}"),
        }
    }

    #[test]
    fn the_leaf_bound_is_respected_on_both_sides() {
        let x = matrix(&[&[0.0], &[1.0], &[2.0], &[3.0], &[4.0], &[5.0]]);
        let y = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let t = grow(
            &x,
            &y,
            &Config {
                depth: 9,
                min_split: 2,
                min_leaf: 3,
            },
        )
        .unwrap();
        // 6 samples with a floor of 3 admits exactly one split, so 3 nodes.
        assert_eq!(t.nodes().len(), 3, "{:?}", t.nodes());
    }

    #[test]
    fn ties_go_to_the_lowest_feature_index_so_a_fit_is_a_function_of_its_data() {
        // Two features that partition identically: every split scores the same. This is
        // the case scikit-learn resolves with its RNG and this crate resolves by index.
        let x = matrix(&[&[0.0, 0.0], &[1.0, 1.0], &[2.0, 2.0], &[3.0, 3.0]]);
        let y = [0.0, 0.0, 10.0, 10.0];
        for _ in 0..16 {
            let t = grow(
                &x,
                &y,
                &Config {
                    depth: 1,
                    ..Config::default()
                },
            )
            .unwrap();
            match t.nodes()[0] {
                Node::Split { feature, .. } => assert_eq!(feature, 0),
                n => panic!("expected a split, got {n:?}"),
            }
        }
    }

    #[test]
    fn a_two_sample_node_is_the_tie_that_makes_sklearn_irreproducible() {
        // Documents the mechanism named in the module header, as a live check: with two
        // samples every feature yields the same proxy score, so only the tie-break rule
        // decides. Ours is total; scikit-learn's is a draw.
        let x = matrix(&[&[0.0, 5.0], &[1.0, -5.0]]);
        let y = [1.0, 2.0];
        let t = grow(
            &x,
            &y,
            &Config {
                depth: 1,
                ..Config::default()
            },
        )
        .unwrap();
        match t.nodes()[0] {
            Node::Split { feature, .. } => assert_eq!(feature, 0, "lowest index must win"),
            n => panic!("expected a split, got {n:?}"),
        }
    }

    #[test]
    fn the_feature_space_is_single_precision_and_a_threshold_shows_it() {
        // The two values straddling the root split of the first boosted fixture tree.
        // Their two midpoints differ in the 8th significant digit, and scikit-learn
        // records the single-precision one — the measurement the module header cites,
        // kept here as a live check so a regression to double precision fails loudly
        // instead of drifting a fixture's tolerance.
        let (a, b) = (-1.5903684017801702f64, -1.5771856983463417f64);
        assert_eq!(a / 2.0 + b / 2.0, -1.583777050063256, "double midpoint");
        assert_eq!(
            single(a) / 2.0 + single(b) / 2.0,
            -1.5837770700454712,
            "single midpoint — this is scikit-learn's threshold"
        );

        // And the splitter must produce the latter. Two samples, so the only admissible
        // split is between them.
        let x = matrix(&[&[a], &[b]]);
        let t = grow(
            &x,
            &[0.0, 10.0],
            &Config {
                depth: 1,
                ..Config::default()
            },
        )
        .unwrap();
        match t.nodes()[0] {
            Node::Split { threshold, .. } => assert_eq!(threshold, -1.5837770700454712),
            n => panic!("expected a split, got {n:?}"),
        }
    }

    #[test]
    fn two_values_that_collapse_to_one_in_binary32_are_not_split_between() {
        // Distinct as f64, the SAME value as f32. A double-precision splitter would place
        // a threshold between them and then be unable to honour it, because the walk
        // reads the feature at single precision too and both rows land on one side.
        let a = 1.0f64;
        let b = f64::from_bits(a.to_bits() + 1); // 1.0000000000000002
        assert_ne!(a, b);
        assert_eq!(single(a), single(b));
        let x = matrix(&[&[a], &[b]]);
        let t = grow(
            &x,
            &[0.0, 10.0],
            &Config {
                depth: 1,
                ..Config::default()
            },
        )
        .unwrap();
        assert_eq!(
            t.nodes(),
            &[Node::Leaf { value: 5.0 }],
            "one leaf at the mean"
        );
    }

    #[test]
    fn a_feature_beyond_binary32_is_refused_rather_than_read_as_an_infinity() {
        // Finite as f64, an infinity the moment a tree reads it: 1e39 is just past
        // binary32's 3.4e38 ceiling. `Matrix` accepts it because a double-precision fit
        // legitimately can; the tree refuses it because it cannot.
        let x = matrix(&[&[1.0], &[1e39], &[2.0], &[3.0]]);
        let e = grow(&x, &[1.0, 2.0, 3.0, 4.0], &Config::default()).unwrap_err();
        let text = format!("{e}");
        assert!(text.contains("row 1 column 0"), "{text}");
        assert!(text.contains("binary32"), "{text}");
        // The same matrix is fine for a double-precision estimator.
        use crate::Fit as _;
        assert!(crate::linear::Config::new()
            .fit(&crate::Samples::new(x, vec![1.0, 2.0, 3.0, 4.0]).unwrap())
            .is_ok());
    }

    #[test]
    fn depth_zero_is_refused_rather_than_silently_yielding_a_stump() {
        assert!(grow(
            &matrix(&[&[0.0], &[1.0]]),
            &[0.0, 1.0],
            &Config {
                depth: 0,
                ..Config::default()
            }
        )
        .is_err());
    }
}
