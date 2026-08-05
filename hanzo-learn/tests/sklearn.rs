//! The correctness bar: every estimator, against scikit-learn's own answer.
//!
//! The fixtures in `tests/fixture/` were produced by scikit-learn 1.9.0 / numpy 2.5.1
//! (`generate.py`, committed beside them) and are read here as data. No Python runs
//! during `cargo test`, so the bar holds on a box that has none and the oracle's version
//! is a recorded fact rather than whatever is installed.
//!
//! # The tolerances, and why each is the number it is
//!
//! A tolerance is a claim about how far two correct implementations may differ. Each one
//! below is derived from the mathematics, not tuned until the test passed. Every
//! assertion also prints the deviation it actually observed, so a regression that stays
//! inside a tolerance is still visible.
//!
//! | estimator | quantity      | tolerance | why                                          |
//! |-----------|---------------|-----------|----------------------------------------------|
//! | linear    | coefficients  | 1e-10     | unique minimiser at full rank; two backward- |
//! |           |               |           | stable solvers (QR here, SVD there) differ by |
//! |           |               |           | O(eps * cond) and cond is 1.31, recorded in  |
//! |           |               |           | the fixture. 1e-10 is ~1e5x that headroom.   |
//! | logistic  | coefficients  | 1e-7      | strictly convex, so ONE minimiser; both sides |
//! |           |               |           | stop near it (sklearn at tol 1e-12, Newton at |
//! |           |               |           | gradient 1e-11). The gap is two stopping      |
//! |           |               |           | distances, not an algorithmic difference.     |
//! | boost     | split feature | EXACT     | an integer index. Equality or the split rule  |
//! |           |               |           | differs.                                      |
//! | boost     | threshold     | EXACT     | `a/2 + b/2` over the same two data values in  |
//! |           |               |           | the SAME precision — binary32, which is where |
//! |           |               |           | every tree implementation reads features (see |
//! |           |               |           | `tree`'s header). Bit-identical, or the        |
//! |           |               |           | candidate scan picked a different pair.       |
//! | boost     | leaf value    | 1e-12     | a mean; differs only by summation order, since |
//! |           |               |           | sklearn accumulates over its partitioned      |
//! |           |               |           | sample array and this crate over the feature- |
//! |           |               |           | sorted one.                                   |
//! | boost     | prediction    | 1e-10     | 100 leaf values accumulated in round order;   |
//! |           |               |           | leaf error compounds at most linearly.        |

use hanzo_learn::{boost, linear, logistic, tree, Fit, Matrix, Model as _, Predict, Samples};
use serde_json::Value;

fn fixture(name: &str) -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixture/");
    let raw = std::fs::read_to_string(format!("{path}{name}"))
        .unwrap_or_else(|e| panic!("{name}: {e} — regenerate with tests/fixture/generate.py"));
    serde_json::from_str(&raw).expect("fixture is valid JSON")
}

/// Read an array of oracle numbers.
///
/// `as_f64` is bit-exact ONLY because this crate enables serde_json's `float_roundtrip`
/// feature; see the note on it in `Cargo.toml`. Without it this helper is accurate to about
/// one ulp, which is indistinguishable from an arithmetic bug in whatever it is checking.
/// Non-finite oracle values arrive as the strings `inf`/`-inf`/`nan`, because JSON has no
/// spelling for them.
fn reals(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| match x.as_str() {
            Some("inf") => f64::INFINITY,
            Some("-inf") => f64::NEG_INFINITY,
            Some("nan") => f64::NAN,
            Some(other) => panic!("not a number: {other}"),
            None => x.as_f64().expect("number"),
        })
        .collect()
}

fn rows(v: &Value) -> Matrix {
    Matrix::rows(
        &v.as_array()
            .expect("array")
            .iter()
            .map(reals)
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

/// Largest absolute deviation, or a panic naming the worst position.
fn agree(what: &str, got: &[f64], want: &[f64], tolerance: f64) -> f64 {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = (0usize, 0.0f64);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        if d > worst.1 {
            worst = (i, d);
        }
    }
    assert!(
        worst.1 <= tolerance,
        "{what}: differs by {:e} at index {} (got {}, sklearn {}), over the {:e} bound",
        worst.1,
        worst.0,
        got[worst.0],
        want[worst.0],
        tolerance
    );
    worst.1
}

const LINEAR_TOLERANCE: f64 = 1e-10;
const LOGISTIC_TOLERANCE: f64 = 1e-7;
const LEAF_TOLERANCE: f64 = 1e-12;
const BOOST_TOLERANCE: f64 = 1e-10;

#[test]
fn least_squares_matches_scikit_learn() {
    let f = fixture("linear.json");
    let x = rows(&f["x"]);
    let y = reals(&f["y"]);
    let x_test = rows(&f["x_test"]);
    let data = Samples::new(x, y).unwrap();
    println!(
        "oracle {} | cond(X) = {:.4}",
        f["oracle"]["sklearn"],
        f["cond"].as_f64().unwrap()
    );

    for (label, config) in [
        ("intercept", linear::Config::new()),
        ("no_intercept", linear::Config::through_origin()),
    ] {
        let case = &f["cases"][label];
        let m = config.fit(&data).unwrap();

        let coef = agree(
            &format!("{label} coefficients"),
            m.coefficients(),
            &reals(&case["coef"]),
            LINEAR_TOLERANCE,
        );
        let want = case["intercept"].as_f64().unwrap();
        let inter = (m.intercept() - want).abs();
        assert!(
            inter <= LINEAR_TOLERANCE,
            "{label} intercept: {} vs sklearn {want}",
            m.intercept()
        );
        let pred = agree(
            &format!("{label} predictions"),
            &m.predict(&x_test).unwrap(),
            &reals(&case["predict_test"]),
            LINEAR_TOLERANCE,
        );
        println!(
            "  {label:13} coefficients {coef:.3e}  intercept {inter:.3e}  predictions {pred:.3e}  \
             (bound {LINEAR_TOLERANCE:e})"
        );
        println!("    address {}", m.address());
    }
}

#[test]
fn logistic_regression_matches_scikit_learn() {
    let f = fixture("logistic.json");
    let x = rows(&f["x"]);
    let labels: Vec<i64> = reals(&f["y"]).into_iter().map(|v| v as i64).collect();
    let x_test = rows(&f["x_test"]);
    let data = Samples::new(x, labels).unwrap();
    println!(
        "oracle {} | class balance {}",
        f["oracle"]["sklearn"], f["class_balance"]
    );

    for label in ["c1", "c100"] {
        let case = &f["cases"][label];
        let c = case["c"].as_f64().unwrap();
        let m = logistic::Config::penalty(c).fit(&data).unwrap();

        let coef = agree(
            &format!("{label} coefficients"),
            m.coefficients(),
            &reals(&case["coef"]),
            LOGISTIC_TOLERANCE,
        );
        let want = case["intercept"].as_f64().unwrap();
        let inter = (m.intercept() - want).abs();
        assert!(
            inter <= LOGISTIC_TOLERANCE,
            "{label} intercept: {} vs sklearn {want}",
            m.intercept()
        );
        let proba = agree(
            &format!("{label} probabilities"),
            &m.probability(&x_test).unwrap(),
            &reals(&case["proba_test"]),
            LOGISTIC_TOLERANCE,
        );

        // The hard label, resolved through the model's own class list, must match
        // sklearn's label EXACTLY — it is a discrete decision, so "close" is meaningless.
        let got: Vec<i64> = m
            .predict(&x_test)
            .unwrap()
            .into_iter()
            .map(|k| m.label(k))
            .collect();
        let sklearn: Vec<i64> = reals(&case["predict_test"])
            .into_iter()
            .map(|v| v as i64)
            .collect();
        assert_eq!(
            got, sklearn,
            "{label}: predicted classes must match exactly"
        );

        println!(
            "  C={c:<6} coefficients {coef:.3e}  intercept {inter:.3e}  probabilities {proba:.3e}  \
             classes exact  ({} Newton steps, bound {LOGISTIC_TOLERANCE:e})",
            m.iterations()
        );
        println!("    address {}", m.address());
    }
}

#[test]
fn boosted_trees_match_scikit_learn_node_for_node() {
    let f = fixture("boosted.json");
    let x = rows(&f["x"]);
    let y = reals(&f["y"]);
    let x_test = rows(&f["x_test"]);
    let data = Samples::new(x, y).unwrap();
    println!("oracle {}", f["oracle"]["sklearn"]);

    for label in ["default", "deep", "stumps", "leafbound"] {
        let case = &f["cases"][label];
        let cfg = &case["config"];
        let config = boost::Config {
            rounds: cfg["n_estimators"].as_u64().unwrap() as usize,
            rate: cfg["learning_rate"].as_f64().unwrap(),
            tree: tree::Config {
                depth: cfg["max_depth"].as_u64().unwrap() as usize,
                min_split: 2,
                min_leaf: cfg["min_samples_leaf"].as_u64().unwrap_or(1) as usize,
            },
        };
        let m = config.fit(&data).unwrap();

        // F_0.
        let base = case["init"].as_f64().unwrap();
        assert!(
            (m.base() - base).abs() <= LEAF_TOLERANCE,
            "{label}: base {} vs sklearn {base}",
            m.base()
        );

        // Node for node, tree for tree. This is the strong claim: it localises any
        // divergence to the first node that differs instead of surfacing one wrong
        // number at the end.
        let trees = case["trees"].as_array().unwrap();
        assert_eq!(m.trees().len(), trees.len(), "{label}: tree count");
        let mut worst_leaf = 0.0f64;
        let mut nodes = 0usize;
        for (r, (ours, theirs)) in m.trees().iter().zip(trees).enumerate() {
            let left = reals(&theirs["left"]);
            let right = reals(&theirs["right"]);
            let feature = reals(&theirs["feature"]);
            let threshold = reals(&theirs["threshold"]);
            let value = reals(&theirs["value"]);
            assert_eq!(
                ours.nodes().len(),
                left.len(),
                "{label} round {r}: node count — ours {}, sklearn {}",
                ours.nodes().len(),
                left.len()
            );
            for (i, node) in ours.nodes().iter().enumerate() {
                nodes += 1;
                // sklearn marks a leaf with children_left == -1.
                let is_leaf = left[i] < 0.0;
                match *node {
                    tree::Node::Leaf { value: v } => {
                        assert!(
                            is_leaf,
                            "{label} round {r} node {i}: ours leaf, sklearn split"
                        );
                        let d = (v - value[i]).abs();
                        assert!(
                            d <= LEAF_TOLERANCE,
                            "{label} round {r} node {i}: leaf {v} vs sklearn {} ({d:e})",
                            value[i]
                        );
                        worst_leaf = worst_leaf.max(d);
                    }
                    tree::Node::Split {
                        feature: fe,
                        threshold: th,
                        left: l,
                        right: rt,
                    } => {
                        assert!(
                            !is_leaf,
                            "{label} round {r} node {i}: ours split, sklearn leaf"
                        );
                        assert_eq!(
                            fe as f64, feature[i],
                            "{label} round {r} node {i}: split feature"
                        );
                        assert_eq!(
                            th, threshold[i],
                            "{label} round {r} node {i}: threshold must be bit-identical"
                        );
                        assert_eq!(l as f64, left[i], "{label} round {r} node {i}: left child");
                        assert_eq!(
                            rt as f64, right[i],
                            "{label} round {r} node {i}: right child"
                        );
                    }
                }
            }
        }

        let pred = agree(
            &format!("{label} predictions"),
            &m.predict(&x_test).unwrap(),
            &reals(&case["predict_test"]),
            BOOST_TOLERANCE,
        );
        println!(
            "  {label:10} {} trees / {nodes} nodes IDENTICAL (feature, threshold, children exact; \
             leaf {worst_leaf:.3e})  predictions {pred:.3e}",
            m.trees().len()
        );
        println!("    address {}", m.address());
    }
}

/// A fitted value's name is a function of its content, and it separates what it must.
#[test]
fn a_fitted_model_is_named_by_its_content() {
    let f = fixture("linear.json");
    let data = Samples::new(rows(&f["x"]), reals(&f["y"])).unwrap();

    let a = linear::Config::new().fit(&data).unwrap();
    let b = linear::Config::new().fit(&data).unwrap();
    assert_eq!(a.address(), b.address(), "same data, same name");

    let c = linear::Config::through_origin().fit(&data).unwrap();
    assert_ne!(a.address(), c.address(), "different config, different name");

    // Across estimator kinds the domain separator does the work: a linear model and a
    // boosted one could in principle hash the same numbers.
    let boosted = boost::Config {
        rounds: 1,
        rate: 1.0,
        tree: tree::Config::default(),
    }
    .fit(&data)
    .unwrap();
    assert_ne!(a.address(), boosted.address());
    println!("linear {}\nboost  {}", a.address(), boosted.address());
}

// ---------------------------------------------------------------------------------------
// Preprocessing, metrics and the anomaly detectors.
//
// EVERY assertion below is at 0.0 — exact equality — except the two named otherwise, and
// that is the point rather than a flourish. A scaler, an encoder, an imputer, a fold
// assignment and a ranking metric are all CLOSED FORMS: there is no optimiser to stop early
// and no tie for a seed to break, so two correct implementations cannot legitimately differ
// at all. Handing these a tolerance would hide the only bug they can have.
//
// | estimator        | quantity                  | bound | why                            |
// |------------------|---------------------------|-------|--------------------------------|
// | scale::Standard  | mean, var, scale, apply   | 1e-13 | a mean and a two-pass variance |
// |                  |                           |       | over the SAME summation order; |
// |                  |                           |       | differs only where numpy's     |
// |                  |                           |       | pairwise sum regroups. Data is |
// |                  |                           |       | scaled to 1e4, so 1e-13 is the |
// |                  |                           |       | last two bits of the largest.  |
// | scale::Range     | everything                | EXACT | a min, a max, a subtract and a |
// |                  |                           |       | divide. No summation at all.   |
// | encode, impute   | everything                | EXACT | order, count, and a median.    |
// | split, folds     | indices                   | EXACT | integers. A different fold is  |
// |                  |                           |       | a different experiment.        |
// | metric ranking   | roc_auc, average_precision| EXACT | a function of the score ORDER  |
// |                  |                           |       | and of counts.                 |
// | metric::log_loss | value                     | 1e-14 | a mean of logarithms; the same |
// |                  |                           |       | clip, differing regrouping.    |
// | neighbour::Local | factor, score_samples     | EXACT | NO randomness anywhere in the  |
// |                  |                           |       | algorithm. The strongest claim |
// |                  |                           |       | in this file.                  |
// | isolation scoring| score over sklearn's trees| 1e-15 | 2^(-d/c): the same divide and  |
// |                  |                           |       | the same exp2 over integers.   |
// | isolation builder| ranking agreement         | see   | NOT exact and cannot be: the   |
// |                  |                           | below | trees depend on numpy's stream.|

use hanzo_learn::{encode, impute, isolation, metric, neighbour, scale, split, Outlier as _};
use hanzo_learn::{Class, Transform as _};

/// Exact. Named, so that a diff which relaxes it is visible as a relaxation.
const EXACT: f64 = 0.0;
const MOMENT_TOLERANCE: f64 = 1e-13;

/// A few last bits, and the reason is always the same one.
///
/// `numpy` sums with PAIRWISE summation — it splits a reduction into blocks and adds the
/// partial sums in a tree — while this crate sums in index order, which `crate::data::mean`
/// fixes deliberately so that two fits here agree bit for bit with each other. The two
/// orders round differently: over `k` terms the tree accumulates about `log2(k)` roundings
/// where the sequential walk accumulates `k`. So any quantity that reaches sklearn through
/// a `np.mean` or a `np.sum` can differ in the last bit or two, and no amount of care in
/// this crate removes it — only reproducing numpy's blocking would, at the cost of making
/// every mean here depend on a block size.
///
/// Stated in ULPs rather than as an absolute number, because that is the unit the claim is
/// actually in: 8 ulp is the same statement about a quantity near 1 and one near 1e6.
const ULPS: f64 = 8.0;

/// Largest deviation in ulps of the oracle value, or a panic naming the worst position.
fn agree_ulp(what: &str, got: &[f64], want: &[f64], ulps: f64) -> f64 {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = (0usize, 0.0f64);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        // ulp of the oracle value: the gap between adjacent f64 at that magnitude.
        let ulp = if *w == 0.0 {
            f64::MIN_POSITIVE
        } else {
            (w.abs() * f64::EPSILON).max(f64::MIN_POSITIVE)
        };
        let d = (g - w).abs() / ulp;
        if d > worst.1 {
            worst = (i, d);
        }
    }
    assert!(
        worst.1 <= ulps,
        "{what}: differs by {:.2} ulp at index {} (got {:?}, sklearn {:?}), over the {} ulp          bound",
        worst.1,
        worst.0,
        got[worst.0],
        want[worst.0],
        ulps
    );
    worst.1
}

fn integers(v: &Value) -> Vec<i64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_i64().expect("integer"))
        .collect()
}

/// Read an array of counts, indices or draws.
///
/// Separate from [`integers`] because a bounded draw legitimately exceeds `i64::MAX` — the
/// stream fixture records one at `2^63 + 6` — and reading it as a signed number would fail
/// on the one case that exercises a mask spanning all 64 bits.
fn naturals(v: &Value) -> Vec<u64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_u64().expect("unsigned integer"))
        .collect()
}

/// Read an array of row indices.
fn places(v: &Value) -> Vec<usize> {
    naturals(v).into_iter().map(|i| i as usize).collect()
}

fn flat(m: &Matrix) -> Vec<f64> {
    (0..m.n()).flat_map(|i| m.row(i).to_vec()).collect()
}

fn flat_fixture(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .flat_map(|r| reals(r))
        .collect()
}

#[test]
fn the_scalers_match_scikit_learn() {
    let f = fixture("prepare.json");
    let x = rows(&f["x"]);
    let x_test = rows(&f["x_test"]);

    let s = scale::Standard::fit(&x);
    let mean = agree(
        "standard mean",
        s.centre(),
        &reals(&f["standard"]["mean"]),
        MOMENT_TOLERANCE,
    );
    let var = agree(
        "standard var",
        s.variance(),
        &reals(&f["standard"]["var"]),
        MOMENT_TOLERANCE,
    );
    // sklearn's scale_ is 1.0 for a zero-variance column, which is the convention
    // `scale::Standard` documents; this asserts we picked THEIR convention and not merely
    // a defensible one.
    let sc = agree(
        "standard scale",
        s.scale(),
        &reals(&f["standard"]["scale"]),
        MOMENT_TOLERANCE,
    );
    // The tolerance on the TRANSFORM is derived, not chosen. `apply` divides by scale_, so
    // whatever the two means disagree by is amplified by 1/min(scale_). This fixture has a
    // column deliberately scaled to 1e-2, so the bound is ~100x the bound on the mean —
    // and that amplification, not any extra error, is the whole of the difference.
    let smallest = s.scale().iter().cloned().fold(f64::MAX, f64::min);
    let transform_tolerance = MOMENT_TOLERANCE / smallest;
    let ap = agree(
        "standard apply",
        &flat(&s.apply(&x_test).unwrap()),
        &flat_fixture(&f["standard"]["apply"]),
        transform_tolerance,
    );
    let inv = agree(
        "standard invert",
        &flat(&s.invert(&s.apply(&x_test).unwrap()).unwrap()),
        &flat_fixture(&f["standard"]["invert"]),
        transform_tolerance,
    );
    println!(
        "standard: mean {mean:e} var {var:e} scale {sc:e} apply {ap:e} invert {inv:e} \
         (bound {transform_tolerance:e} = {MOMENT_TOLERANCE:e} / min scale {smallest:e})"
    );

    for (label, low, high) in [("unit", 0.0, 1.0), ("shifted", -3.0, 7.0)] {
        let case = &f["range"][label];
        let r = scale::Range::fit(&x, scale::Span::new(low, high).unwrap());
        let lo = agree(
            &format!("range {label} data_min"),
            r.low(),
            &reals(&case["data_min"]),
            EXACT,
        );
        let hi = agree(
            &format!("range {label} data_max"),
            r.high(),
            &reals(&case["data_max"]),
            EXACT,
        );
        let ap = agree(
            &format!("range {label} apply"),
            &flat(&r.apply(&x_test).unwrap()),
            &flat_fixture(&case["apply"]),
            EXACT,
        );
        println!("range {label}: min {lo:e} max {hi:e} apply {ap:e}");
    }
}

#[test]
fn the_encoders_match_scikit_learn() {
    let f = fixture("prepare.json");

    // LabelEncoder: classes ascending, codes as positions in that list.
    let values = integers(&f["label"]["values"]);
    let le = encode::Label::fit(&values).unwrap();
    assert_eq!(le.classes(), integers(&f["label"]["classes"]).as_slice());
    let got: Vec<i64> = le
        .codes(&values)
        .unwrap()
        .iter()
        .map(|c| c.index() as i64)
        .collect();
    assert_eq!(got, integers(&f["label"]["codes"]), "label codes");
    // A value the encoder never saw is refused rather than mapped to something.
    assert!(le.code(&1234).is_err());

    // OneHotEncoder over the same integer codes.
    let levels: Vec<usize> = integers(&f["onehot"]["levels"])
        .iter()
        .map(|v| *v as usize)
        .collect();
    let source: Vec<Vec<i64>> = f["onehot"]["codes"]
        .as_array()
        .unwrap()
        .iter()
        .map(integers)
        .collect();
    let mut columns: Vec<Vec<Class>> = vec![Vec::new(); levels.len()];
    for row in &source {
        for (j, v) in row.iter().enumerate() {
            // Route through a Label per column so the Class values are minted by a fitted
            // encoder, which is the only way to get one.
            let column: Vec<i64> = (0..levels[j] as i64).collect();
            columns[j].push(encode::Label::fit(&column).unwrap().code(v).unwrap());
        }
    }
    let codes = encode::Codes::columns(&columns).unwrap();
    let oh = encode::OneHot::of(&levels).unwrap();
    let got = oh.apply(&codes).unwrap();
    let want = flat_fixture(&f["onehot"]["apply"]);
    assert_eq!(got.p(), levels.iter().sum::<usize>(), "one-hot width");
    agree("onehot apply", &flat(&got), &want, EXACT);
    println!("onehot: {} columns exact", got.p());
}

#[test]
fn the_imputer_matches_scikit_learn() {
    let f = fixture("prepare.json");
    let gapped = &f["impute"]["gapped"].as_array().unwrap();
    let cols = gapped[0].as_array().unwrap().len();
    let cells: Vec<Option<f64>> = gapped
        .iter()
        .flat_map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|v| if v.is_null() { None } else { v.as_f64() })
                .collect::<Vec<_>>()
        })
        .collect();
    let partial = impute::Partial::new(gapped.len(), cols, cells).unwrap();

    for (label, statistic) in [
        ("mean", impute::Statistic::Mean),
        ("median", impute::Statistic::Median),
        ("most_frequent", impute::Statistic::Mode),
    ] {
        let case = &f["impute"][label];
        let fill = impute::Fill::fit(&partial, statistic).unwrap();
        let s = agree(
            &format!("impute {label} statistic"),
            fill.value(),
            &reals(&case["statistic"]),
            EXACT,
        );
        let a = agree(
            &format!("impute {label} apply"),
            &flat(&fill.apply(&partial).unwrap()),
            &flat_fixture(&case["apply"]),
            EXACT,
        );
        println!("impute {label}: statistic {s:e} apply {a:e}");
    }
}

/// numpy's generator itself, draw for draw.
///
/// Everything else in this file is a closed form — a mean, a fold boundary, a curve — and a
/// closed form consumes no random numbers. So until this existed, the whole of `twister`
/// was held up by nothing: a bounded draw taking a modulus instead of rejecting, or
/// spending one word where numpy spends two, produced byte-identical fixtures and stayed
/// green. Every assertion here is on INTEGERS and is exact; there is no tolerance to state,
/// because a stream is either numpy's or it is a different stream.
#[test]
fn the_generator_matches_numpys_stream_draw_for_draw() {
    use hanzo_learn::twister::Twister;
    let f = fixture("stream.json");

    let seed_of = |case: &Value| case["seed"].as_u64().expect("seed") as u32;

    // `RandomState.randint(0, most + 1)` IS `below(most)`. The cases straddle `2^32 - 1`,
    // where numpy stops spending one word per draw and starts spending two.
    for case in f["bounded"].as_array().unwrap() {
        let (seed, most) = (seed_of(case), case["most"].as_u64().unwrap());
        let want = naturals(&case["draws"]);
        let mut t = Twister::seed(seed);
        let got: Vec<u64> = (0..want.len()).map(|_| t.below(most)).collect();
        assert_eq!(got, want, "below({most}) under seed {seed}");
    }

    // The descending bounds a partial Fisher-Yates asks for, on a design too large to
    // permute. Every one of these needs the 64-bit branch, and a 32-bit bound would answer
    // every one of them with a number under five.
    for case in f["descending"].as_array().unwrap() {
        let (seed, top) = (seed_of(case), case["top"].as_u64().unwrap());
        let want = naturals(&case["draws"]);
        let mut t = Twister::seed(seed);
        let got: Vec<u64> = (0..want.len() as u64)
            .map(|step| t.below(top - step))
            .collect();
        assert_eq!(
            got,
            want,
            "the bounds a subsample of {} rows draws under seed {seed}",
            top + 1
        );
    }

    for case in f["permutation"].as_array().unwrap() {
        let (seed, n) = (seed_of(case), case["n"].as_u64().unwrap() as usize);
        assert_eq!(
            Twister::seed(seed).permutation(n),
            places(&case["order"]),
            "permutation({n}) under seed {seed}"
        );
    }

    // A subsample against NUMPY's permutation rather than against our own, so that "our
    // permutation is numpy's" and "our subsample is our permutation's tail" cannot hold
    // each other up.
    for case in f["subsample"].as_array().unwrap() {
        let seed = seed_of(case);
        let n = case["n"].as_u64().unwrap() as usize;
        let take = case["take"].as_u64().unwrap() as usize;
        assert_eq!(
            Twister::seed(seed).subsample(n, take),
            places(&case["rows"]),
            "subsample({n}, {take}) under seed {seed} is not numpy's permutation reversed"
        );
    }

    // Two calls off ONE generator. This is what catches a draw that is right in its VALUE
    // and wrong in how many words it spent: the bounded draws still match, and the
    // permutation that follows them does not.
    let c = &f["composed"];
    let mut t = Twister::seed(seed_of(c));
    let want = naturals(&c["first"]);
    let most = c["most"].as_u64().unwrap();
    let got: Vec<u64> = (0..want.len()).map(|_| t.below(most)).collect();
    assert_eq!(got, want, "the bounded draws before the permutation");
    assert_eq!(
        t.permutation(c["n"].as_u64().unwrap() as usize),
        places(&c["then"]),
        "the permutation that follows them left the stream in the wrong place"
    );

    println!(
        "stream: {} bounded cases (both sides of 2^32), {} descending, {} permutations, \
         {} subsamples, all exact",
        f["bounded"].as_array().unwrap().len(),
        f["descending"].as_array().unwrap().len(),
        f["permutation"].as_array().unwrap().len(),
        f["subsample"].as_array().unwrap().len(),
    );
}

#[test]
fn the_splitters_match_scikit_learn_index_for_index() {
    let f = fixture("prepare.json");
    let n = f["split"]["n"].as_u64().unwrap() as usize;

    // The seeds the shuffled cases were recorded under. Two, because one seed cannot tell a
    // generator that IS numpy's from one that happens to agree on a single stream.
    const SEEDS: [u32; 2] = [7, 11];

    for share in ["0.25", "0.1", "0.333"] {
        let case = &f["split"][format!("sequential_{share}")];
        let s = split::train_test(n, share.parse().unwrap(), split::Order::Sequential).unwrap();
        assert_eq!(
            s.train(),
            places(&case["train"]),
            "train_test {share} train"
        );
        assert_eq!(s.test(), places(&case["test"]), "train_test {share} test");

        for seed in SEEDS {
            let case = &f["split"][format!("shuffled_{share}_{seed}")];
            let s =
                split::train_test(n, share.parse().unwrap(), split::Order::Shuffled(seed)).unwrap();
            // sklearn hands these back in PERMUTATION order; a plan here is a set of rows,
            // ascending, which `split` documents. Sorting the oracle's copy is exactly that
            // normalisation — it reorders rows and cannot turn one set of rows into another,
            // so the claim "the same rows were held back" is untouched by it.
            let (mut want_train, mut want_test) = (places(&case["train"]), places(&case["test"]));
            want_train.sort_unstable();
            want_test.sort_unstable();
            assert_eq!(
                s.train(),
                want_train,
                "train_test {share} seed {seed} train"
            );
            assert_eq!(s.test(), want_test, "train_test {share} seed {seed} test");
        }
    }

    for k in [3usize, 5, 7] {
        let want = f["folds"][format!("sequential_{k}")].as_array().unwrap();
        let got = split::folds(n, k, split::Order::Sequential).unwrap();
        assert_eq!(got.len(), want.len(), "fold count");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(g.test(), places(&w["test"]), "KFold k={k} fold {i} test");
            assert_eq!(g.train(), places(&w["train"]), "KFold k={k} fold {i} train");
        }

        for seed in SEEDS {
            let want = f["folds"][format!("shuffled_{k}_{seed}")]
                .as_array()
                .unwrap();
            let got = split::folds(n, k, split::Order::Shuffled(seed)).unwrap();
            assert_eq!(got.len(), want.len(), "shuffled fold count");
            for (i, (g, w)) in got.iter().zip(want).enumerate() {
                // KFold masks its folds back onto 0..n, so sklearn's own answer is already
                // ascending on both sides. Exact, with nothing normalised away.
                assert_eq!(
                    g.test(),
                    places(&w["test"]),
                    "KFold k={k} seed={seed} fold {i} test"
                );
                assert_eq!(
                    g.train(),
                    places(&w["train"]),
                    "KFold k={k} seed={seed} fold {i} train"
                );
            }
        }
    }

    // StratifiedKFold on an 8%-positive label: the imbalanced case, where a plain KFold
    // can hand a fold with no positives in it at all.
    let labels = integers(&f["stratified"]["labels"]);
    let column: Vec<i64> = vec![0, 1];
    let le = encode::Label::fit(&column).unwrap();
    let classes: Vec<Class> = labels.iter().map(|v| le.code(v).unwrap()).collect();
    for k in [3usize, 5] {
        let want = f["stratified"]["folds"][format!("sequential_{k}")]
            .as_array()
            .unwrap();
        let got = split::stratified(&classes, k, split::Order::Sequential).unwrap();
        assert_eq!(got.len(), want.len(), "stratified fold count");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(
                g.test(),
                places(&w["test"]),
                "StratifiedKFold k={k} fold {i} test — a different fold is a different \
                 experiment, so this is exact"
            );
        }

        // The sharpest case here. A stratified shuffle draws once PER CLASS off ONE
        // generator: the second class's shuffle continues the stream the first left off.
        // Re-seeding per class is still reproducible and still balanced, and disagrees
        // with scikit-learn on every fold.
        for seed in SEEDS {
            let want = f["stratified"]["folds"][format!("shuffled_{k}_{seed}")]
                .as_array()
                .unwrap();
            let got = split::stratified(&classes, k, split::Order::Shuffled(seed)).unwrap();
            assert_eq!(got.len(), want.len(), "stratified shuffled fold count");
            for (i, (g, w)) in got.iter().zip(want).enumerate() {
                assert_eq!(
                    g.test(),
                    places(&w["test"]),
                    "StratifiedKFold k={k} seed={seed} fold {i} test"
                );
                assert_eq!(
                    g.train(),
                    places(&w["train"]),
                    "StratifiedKFold k={k} seed={seed} fold {i} train"
                );
            }
        }
    }
    println!(
        "splits: train_test, KFold(3,5,7), StratifiedKFold(3,5) exact, sequential and \
         shuffled under seeds {SEEDS:?}"
    );
}

#[test]
fn the_imbalanced_metrics_match_scikit_learn() {
    let f = fixture("measure.json");
    let truth: Vec<bool> = integers(&f["truth"]).iter().map(|v| *v == 1).collect();
    let score = reals(&f["score"]);
    let curve = metric::Curve::of(&truth, &score).unwrap();

    // ROC AUC is a trapezoid sum and average precision a weighted sum, so both reach
    // sklearn through numpy's pairwise reduction: ULPS, not EXACT. The COUNTS underneath
    // them are exact, which is what the curve assertions below check.
    let auc = agree_ulp(
        "roc_auc",
        &[curve.roc_auc()],
        &[f["roc_auc"].as_f64().unwrap()],
        ULPS,
    );
    let ap = agree_ulp(
        "average_precision — the metric that drifts when ties are handled as a trapezoid \
         instead of as a step",
        &[curve.average_precision()],
        &[f["average_precision"].as_f64().unwrap()],
        ULPS,
    );

    // The curves themselves, not just their summaries. Precision and recall are ratios of
    // integer counts, so EXACT is the only defensible bar for them.
    //
    // The mapping between the two conventions is written out here rather than hidden in the
    // library: ours starts at the flag-nothing endpoint and walks the threshold DOWN;
    // sklearn's runs the other way with the endpoint appended last and a thresholds array
    // one element shorter. Reversing our tail must reproduce sklearn's head, and our
    // endpoint must be sklearn's last element.
    let (precision, recall, thresholds) = curve.precision_recall();
    let want_p = reals(&f["pr"]["precision"]);
    let want_r = reals(&f["pr"]["recall"]);
    let want_t = reals(&f["pr"]["thresholds"]);
    assert_eq!(precision.len(), want_p.len(), "pr precision: length");
    assert_eq!(thresholds.len(), want_t.len() + 1, "pr thresholds: length");
    assert_eq!(
        (precision[0], recall[0]),
        (*want_p.last().unwrap(), *want_r.last().unwrap()),
        "the flag-nothing endpoint is ours at index 0 and sklearn's last"
    );
    assert!(
        thresholds[0].is_infinite(),
        "our endpoint threshold is +inf"
    );
    let mine_p: Vec<f64> = precision[1..].iter().rev().cloned().collect();
    let mine_r: Vec<f64> = recall[1..].iter().rev().cloned().collect();
    let mine_t: Vec<f64> = thresholds[1..].iter().rev().cloned().collect();
    agree("pr precision", &mine_p, &want_p[..want_p.len() - 1], EXACT);
    agree("pr recall", &mine_r, &want_r[..want_r.len() - 1], EXACT);
    agree("pr thresholds", &mine_t, &want_t, EXACT);
    let (fpr, tpr, roc_thresholds) = curve.roc();
    agree("roc fpr", &fpr, &reals(&f["roc"]["fpr"]), EXACT);
    agree("roc tpr", &tpr, &reals(&f["roc"]["tpr"]), EXACT);
    // sklearn's first ROC threshold is +inf: the operating point that flags nothing. Pinned
    // explicitly rather than through a subtraction, because inf - inf is NaN and a NaN
    // would slip past an absolute-difference check without ever being compared.
    let want_thresholds = reals(&f["roc"]["thresholds"]);
    assert!(
        roc_thresholds[0].is_infinite() && want_thresholds[0].is_infinite(),
        "the flags-nothing operating point must be at infinity on both sides, got {} vs {}",
        roc_thresholds[0],
        want_thresholds[0]
    );
    agree(
        "roc thresholds",
        &roc_thresholds[1..],
        &want_thresholds[1..],
        EXACT,
    );

    // Confusion at a threshold, in sklearn's own [[tn, fp], [fn, tp]] order.
    // Keys exactly as `str(float)` writes them in the generator: "0.0", not "0".
    for t in ["-0.5", "0.0", "0.2", "0.5"] {
        let want = integers(&f["confusion"][t]);
        let at = curve.at(t.parse().unwrap());
        let [tn, fp, fn_, tp] = at.matrix();
        assert_eq!(
            vec![tn as i64, fp as i64, fn_ as i64, tp as i64],
            want,
            "confusion at {t}"
        );
    }

    let prob = reals(&f["probability"]);
    let got = metric::log_loss(&truth, &prob).unwrap();
    let want = f["log_loss"].as_f64().unwrap();
    assert!(
        (got - want).abs() <= 1e-14,
        "log_loss {got} vs sklearn {want} (differs by {:e}) — a fixture with a 0.0 and a \
         1.0 in it, so this also pins the clip",
        (got - want).abs()
    );
    println!(
        "metrics: roc_auc {:.17} ({auc:.2} ulp) ap {:.17} ({ap:.2} ulp) log_loss {:.17}; \
         curves, thresholds and confusion counts all EXACT",
        curve.roc_auc(),
        curve.average_precision(),
        got
    );
}

/// The strongest claim in this file — LOF has no randomness, so there is nothing to excuse a
/// disagreement beyond the last bit.
///
/// MEASURED, and my first version of this test was wrong: I asserted 0.0 on every quantity
/// and it failed at 1 ulp on the density. The k-distance IS exact, because it is a sqrt of a
/// sum over the p features in the same order on both sides. The density and the factor are
/// MEANS over the k neighbours, and sklearn takes those means through `np.mean`, whose
/// pairwise reduction rounds differently from an index-order walk. So the honest bar is a
/// handful of ulps, and pretending otherwise would have meant tuning the arithmetic to a
/// claim instead of the claim to the arithmetic.
#[test]
fn the_local_outlier_factor_matches_scikit_learn_exactly() {
    let f = fixture("outlier.json");
    let x = rows(&f["x"]);
    let x_test = rows(&f["x_test"]);

    for k in [5usize, 20, 50] {
        let case = &f["lof"][k.to_string()];
        let l = neighbour::Local::fit(&x, neighbour::Neighbours::new(k).unwrap()).unwrap();

        agree(
            &format!("lof k={k} k-distance"),
            l.reach(),
            &reals(&case["k_distance"]),
            EXACT,
        );
        // The k-distance above is EXACT: a sqrt of a sum over p terms, in the same order.
        // Everything below it is a MEAN over the k neighbours, which is where numpy's
        // pairwise reduction parts company with an index-order walk — so these are in ulps.
        let lrd = agree_ulp(
            &format!("lof k={k} lrd"),
            l.density(),
            &reals(&case["lrd"]),
            ULPS,
        );
        let nof = agree_ulp(
            &format!("lof k={k} negative_outlier_factor_"),
            l.factor(),
            &reals(&case["negative_outlier_factor"]),
            ULPS,
        );
        // Outlier::outlier is -score_samples, on unseen rows and on the fitted rows read
        // as queries. Both directions of the sign convention, pinned.
        let got: Vec<f64> = l.outlier(&x_test).unwrap().iter().map(|v| -v).collect();
        let test = agree_ulp(
            &format!("lof k={k} score_samples(test)"),
            &got,
            &reals(&case["score_samples_test"]),
            ULPS,
        );
        let got: Vec<f64> = l.outlier(&x).unwrap().iter().map(|v| -v).collect();
        let train = agree_ulp(
            &format!("lof k={k} score_samples(train)"),
            &got,
            &reals(&case["score_samples_train"]),
            ULPS,
        );
        println!(
            "lof k={k}: k-distance EXACT; lrd {lrd:.2} ulp, factor {nof:.2} ulp, \
             score_samples(test) {test:.2} ulp, score_samples(train) {train:.2} ulp"
        );
    }
}

/// Isolation forest, split into the claim that CAN be exact and the one that cannot.
///
/// The scoring arithmetic is exercised against scikit-learn's OWN trees, read out of the
/// fixture node by node, so an exact bar is legitimate. The builder is then held to a
/// ranking bar, because a forest grown from a different random stream is a different forest
/// and pinning it against sklearn's would be pinning a coincidence.
#[test]
fn the_isolation_score_matches_scikit_learn_over_its_own_trees() {
    let f = fixture("outlier.json");

    // The correction term first: it is the whole normalisation, at the sizes the trees
    // actually reach.
    for (m, want) in f["average_path_length"].as_object().unwrap() {
        let m: u64 = m.parse().unwrap();
        let got = isolation::average_path(m);
        let want = want.as_f64().unwrap();
        assert!(
            (got - want).abs() <= 1e-14,
            "c({m}) = {got}, sklearn {want}"
        );
    }

    // Walk sklearn's trees to reproduce its score_samples. This is the arithmetic in
    // `Outlier for Forest`, done here over foreign trees: mean depth over the forest,
    // each leaf credited with average_path of the rows that reached it, normalised by
    // average_path(max_samples).
    let trees = f["forest"]["trees"].as_array().unwrap();
    let max_samples = f["forest"]["max_samples"].as_u64().unwrap();
    let denominator = trees.len() as f64 * isolation::average_path(max_samples);

    for (which, key) in [
        ("test", "score_samples_test"),
        ("train", "score_samples_train"),
    ] {
        let x = rows(&f[if which == "test" { "x_test" } else { "x" }]);
        let mut got = Vec::with_capacity(x.n());
        for i in 0..x.n() {
            let row = x.row(i);
            let mut total = 0.0;
            for t in trees {
                let left = integers(&t["left"]);
                let right = integers(&t["right"]);
                let feature = integers(&t["feature"]);
                let threshold = reals(&t["threshold"]);
                let size = integers(&t["n_node_samples"]);
                let mut at = 0usize;
                let mut depth = 0.0;
                // sklearn marks a leaf with children_left == -1.
                while left[at] != -1 {
                    at = if row[feature[at] as usize] <= threshold[at] {
                        left[at] as usize
                    } else {
                        right[at] as usize
                    };
                    depth += 1.0;
                }
                total += depth + isolation::average_path(size[at] as u64);
            }
            got.push(-(-total / denominator).exp2());
        }
        agree(
            &format!("isolation score_samples({which}) over sklearn's own trees"),
            &got,
            &reals(&f["forest"][key]),
            1e-15,
        );
        println!("isolation: score_samples({which}) over sklearn's trees matches to 1e-15");
    }
}

/// Our own forest, held to what a differently-seeded forest CAN be held to.
#[test]
fn our_isolation_forest_ranks_the_same_rows_as_scikit_learns() {
    let f = fixture("outlier.json");
    let x = rows(&f["x"]);
    let x_test = rows(&f["x_test"]);
    let forest = isolation::Config::new(100, 128, 0)
        .unwrap()
        .fit(&x)
        .unwrap();

    // The fixture's design has its first 6 rows planted far out, and x_test's last 5.
    // Agreement is measured as the metric a risk desk would use: does our score rank the
    // planted rows above the rest, as well as sklearn's does?
    for (label, data, planted) in [("train", &x, 0..6usize), ("test", &x_test, 30..35)] {
        let truth: Vec<bool> = (0..data.n()).map(|i| planted.contains(&i)).collect();
        let ours = forest.outlier(data).unwrap();
        let theirs: Vec<f64> = reals(
            &f["forest"][if label == "train" {
                "score_samples_train"
            } else {
                "score_samples_test"
            }],
        )
        .iter()
        .map(|v| -v)
        .collect();
        let our_auc = metric::Curve::of(&truth, &ours).unwrap().roc_auc();
        let their_auc = metric::Curve::of(&truth, &theirs).unwrap().roc_auc();
        println!("isolation {label}: our AUC {our_auc:.6}, sklearn AUC {their_auc:.6}");
        assert_eq!(
            our_auc, 1.0,
            "our forest failed to separate the planted rows at all"
        );
        assert!(
            our_auc >= their_auc,
            "our forest ranks worse than sklearn's: {our_auc} < {their_auc}"
        );
    }
}
