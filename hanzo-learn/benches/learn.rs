//! Fit and predict time for every estimator here, against scikit-learn on the same data.
//!
//! # Method, which is the repo's and not a new one
//!
//! Follows `hanzo-ml/benches/`: `criterion`, a NAMED SHAPE TABLE so a number always says
//! which problem it is for, `black_box` around the result so nothing is optimised away, and
//! the allocation of inputs kept outside the timed region. The scikit-learn side is measured
//! by `oracle/bench.py` beside it, and both are reported together in `oracle/README.md`.
//!
//! # The data is the same data, and that is provable rather than asserted
//!
//! Both sides draw from `numpy`'s legacy Mersenne stream at the same seed:
//! [`hanzo_learn::twister::Twister::next_real`] is `RandomState.random_sample` bit for bit,
//! which `twister`'s own test pins against numpy's output. So the Rust and Python
//! benchmarks see IDENTICAL values, not merely the same distribution — which matters for
//! the two estimators whose cost depends on the values (a tree's depth, a neighbour
//! frontier's insert rate) rather than only on the shape.
//!
//! Run with:
//!     cargo bench -p hanzo-learn --bench learn
//! or, for a number in a hurry:
//!     cargo bench -p hanzo-learn --bench learn -- --warm-up-time 1 --measurement-time 3
//!
//! `--bench learn` is not optional when passing those flags: a bare `cargo bench` also runs
//! the LIB target's built-in harness, which does not understand criterion's options and
//! fails the whole run with `Unrecognized option: 'warm-up-time'`.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use std::hint::black_box;

use hanzo_learn::twister::Twister;
use hanzo_learn::{encode, impute, isolation, metric, neighbour, scale, split};
use hanzo_learn::{Class, Matrix, Outlier as _, Transform as _};

/// The shapes every `O(n·p)` estimator is measured at.
///
/// `p = 20` throughout: a risk feature vector is tens of columns, not thousands, and holding
/// it fixed makes the `n` scaling readable down the column.
const SHAPES: &[(&str, usize, usize)] = &[
    ("10k_x_20", 10_000, 20),
    ("100k_x_20", 100_000, 20),
    ("1m_x_20", 1_000_000, 20),
];

/// The shapes the QUADRATIC estimator is measured at.
///
/// Stops at 20k on purpose. `neighbour::Local` is `O(n²p)` by algorithm, so 10⁶ rows is
/// about 17 minutes of one core — the bound its module docs state. Benchmarking it there
/// would measure a decision nobody should make rather than a cost anybody should pay.
const NEAR_SHAPES: &[(&str, usize, usize)] = &[("2k_x_20", 2_000, 20), ("20k_x_20", 20_000, 20)];

/// A design from `numpy`'s stream, so Python sees the same numbers.
fn design(n: usize, p: usize, seed: u32) -> Matrix {
    let mut t = Twister::seed(seed);
    let data: Vec<f64> = (0..n * p).map(|_| t.next_real() * 2.0 - 1.0).collect();
    Matrix::new(n, p, data).unwrap()
}

/// A design with a handful of rows thrown far out: what an anomaly detector is pointed at.
fn planted(n: usize, p: usize, seed: u32) -> Matrix {
    let mut t = Twister::seed(seed);
    let mut data: Vec<f64> = (0..n * p).map(|_| t.next_real() * 2.0 - 1.0).collect();
    for row in 0..(n / 100).max(1) {
        for j in 0..p {
            data[row * p + j] += 9.0;
        }
    }
    Matrix::new(n, p, data).unwrap()
}

fn scalers(c: &mut Criterion) {
    let mut g = c.benchmark_group("scale");
    for &(name, n, p) in SHAPES {
        let x = design(n, p, 1);
        g.throughput(Throughput::Elements((n * p) as u64));
        g.bench_function(format!("standard_fit/{name}"), |b| {
            b.iter(|| black_box(scale::Standard::fit(black_box(&x))))
        });
        let fitted = scale::Standard::fit(&x);
        g.bench_function(format!("standard_apply/{name}"), |b| {
            b.iter(|| black_box(fitted.apply(black_box(&x)).unwrap()))
        });
        g.bench_function(format!("range_fit/{name}"), |b| {
            b.iter(|| black_box(scale::Range::fit(black_box(&x), scale::Span::UNIT)))
        });
        let ranged = scale::Range::fit(&x, scale::Span::UNIT);
        g.bench_function(format!("range_apply/{name}"), |b| {
            b.iter(|| black_box(ranged.apply(black_box(&x)).unwrap()))
        });
    }
    g.finish();
}

fn imputers(c: &mut Criterion) {
    let mut g = c.benchmark_group("impute");
    for &(name, n, p) in SHAPES {
        // One value in nineteen missing. NINETEEN, coprime with the 20 columns, so the gaps
        // walk across every column. A stride equal to `p` would put every gap in ONE column,
        // leaving that column with no observed value at all and the other nineteen complete
        // — scikit-learn drops such a column with a warning, so the two sides would no
        // longer be doing the same work. Measured: the first version of this benchmark had
        // exactly that bug.
        let x = design(n, p, 2);
        let cells: Vec<Option<f64>> = (0..n * p)
            .map(|k| {
                if k % 19 == 7 {
                    None
                } else {
                    Some(x.at(k / p, k % p))
                }
            })
            .collect();
        let gapped = impute::Partial::new(n, p, cells).unwrap();
        g.throughput(Throughput::Elements((n * p) as u64));
        g.bench_function(format!("mean_fit/{name}"), |b| {
            b.iter(|| {
                black_box(impute::Fill::fit(black_box(&gapped), impute::Statistic::Mean).unwrap())
            })
        });
        g.bench_function(format!("median_fit/{name}"), |b| {
            b.iter(|| {
                black_box(impute::Fill::fit(black_box(&gapped), impute::Statistic::Median).unwrap())
            })
        });
        let fill = impute::Fill::fit(&gapped, impute::Statistic::Mean).unwrap();
        g.bench_function(format!("apply/{name}"), |b| {
            b.iter(|| black_box(fill.apply(black_box(&gapped)).unwrap()))
        });
    }
    g.finish();
}

fn encoders(c: &mut Criterion) {
    let mut g = c.benchmark_group("encode");
    for &(name, n, _) in SHAPES {
        // Eight categorical columns of ten levels: a realistic width after one-hot (80).
        let levels = vec![10usize; 8];
        let alphabet: Vec<i64> = (0..10).collect();
        let label = encode::Label::fit(&alphabet).unwrap();
        let raw: Vec<i64> = {
            let mut t = Twister::seed(3);
            (0..n * 8).map(|_| t.below(9) as i64).collect()
        };
        let columns: Vec<Vec<Class>> = (0..8)
            .map(|j| {
                (0..n)
                    .map(|i| label.code(&raw[i * 8 + j]).unwrap())
                    .collect()
            })
            .collect();
        let codes = encode::Codes::columns(&columns).unwrap();
        let one_hot = encode::OneHot::of(&levels).unwrap();
        g.throughput(Throughput::Elements((n * 8) as u64));
        g.bench_function(format!("label_fit/{name}"), |b| {
            b.iter(|| black_box(encode::Label::fit(black_box(&raw)).unwrap()))
        });
        g.bench_function(format!("label_codes/{name}"), |b| {
            b.iter(|| black_box(label.codes(black_box(&raw)).unwrap()))
        });
        g.bench_function(format!("onehot_apply/{name}"), |b| {
            b.iter(|| black_box(one_hot.apply(black_box(&codes)).unwrap()))
        });
    }
    g.finish();
}

fn splitters(c: &mut Criterion) {
    let mut g = c.benchmark_group("split");
    let alphabet: Vec<i64> = vec![0, 1];
    let label = encode::Label::fit(&alphabet).unwrap();
    for &(name, n, _) in SHAPES {
        let mut t = Twister::seed(4);
        // 8% positive: the imbalance risk work actually has.
        let classes: Vec<Class> = (0..n)
            .map(|_| label.code(&i64::from(t.next_real() < 0.08)).unwrap())
            .collect();
        g.throughput(Throughput::Elements(n as u64));
        g.bench_function(format!("train_test/{name}"), |b| {
            b.iter(|| {
                black_box(split::train_test(
                    black_box(n),
                    0.25,
                    split::Order::Shuffled(0),
                ))
            })
        });
        g.bench_function(format!("kfold_5/{name}"), |b| {
            b.iter(|| black_box(split::folds(black_box(n), 5, split::Order::Shuffled(0)).unwrap()))
        });
        g.bench_function(format!("stratified_5/{name}"), |b| {
            b.iter(|| {
                black_box(
                    split::stratified(black_box(&classes), 5, split::Order::Shuffled(0)).unwrap(),
                )
            })
        });
    }
    g.finish();
}

fn metrics(c: &mut Criterion) {
    let mut g = c.benchmark_group("metric");
    for &(name, n, _) in SHAPES {
        let mut t = Twister::seed(5);
        let truth: Vec<bool> = (0..n).map(|_| t.next_real() < 0.07).collect();
        let score: Vec<f64> = truth
            .iter()
            .map(|&y| (if y { 0.35 } else { 0.0 }) + t.next_real())
            .collect();
        // log_loss needs a PROBABILITY, and `score` is not one — it reaches 1.35. Squashed
        // here exactly as `oracle/bench.py` squashes it, so both sides pay for the same
        // logarithms over the same values.
        let probability: Vec<f64> = score.iter().map(|s| 1.0 / (1.0 + (-s).exp())).collect();
        let curve = metric::Curve::of(&truth, &score).unwrap();
        g.throughput(Throughput::Elements(n as u64));
        g.bench_function(format!("curve_of/{name}"), |b| {
            b.iter(|| black_box(metric::Curve::of(black_box(&truth), black_box(&score)).unwrap()))
        });
        g.bench_function(format!("roc_auc/{name}"), |b| {
            b.iter(|| black_box(black_box(&curve).roc_auc()))
        });
        g.bench_function(format!("average_precision/{name}"), |b| {
            b.iter(|| black_box(black_box(&curve).average_precision()))
        });
        g.bench_function(format!("precision_recall/{name}"), |b| {
            b.iter(|| black_box(black_box(&curve).precision_recall()))
        });
        g.bench_function(format!("log_loss/{name}"), |b| {
            b.iter(|| {
                black_box(metric::log_loss(black_box(&truth), black_box(&probability)).unwrap())
            })
        });
    }
    g.finish();
}

fn isolation_forest(c: &mut Criterion) {
    let mut g = c.benchmark_group("isolation");
    // Sample size is what bounds a tree, so fit cost is nearly flat in n and predict cost is
    // linear in it. Both are measured so the shape of that claim is visible.
    g.sample_size(20);
    for &(name, n, p) in SHAPES {
        let x = planted(n, p, 6);
        let config = isolation::Config::default();
        g.throughput(Throughput::Elements(n as u64));
        g.bench_function(format!("fit_100_trees/{name}"), |b| {
            b.iter(|| black_box(config.fit(black_box(&x)).unwrap()))
        });
        let forest = config.fit(&x).unwrap();
        g.bench_function(format!("outlier/{name}"), |b| {
            b.iter(|| black_box(forest.outlier(black_box(&x)).unwrap()))
        });
    }
    g.finish();
}

fn local_outlier_factor(c: &mut Criterion) {
    let mut g = c.benchmark_group("neighbour");
    g.sample_size(10);
    for &(name, n, p) in NEAR_SHAPES {
        let x = planted(n, p, 7);
        g.throughput(Throughput::Elements(n as u64));
        g.bench_function(format!("fit_k20/{name}"), |b| {
            b.iter(|| {
                black_box(
                    neighbour::Local::fit(black_box(&x), neighbour::Neighbours::DEFAULT).unwrap(),
                )
            })
        });
        let fitted = neighbour::Local::fit(&x, neighbour::Neighbours::DEFAULT).unwrap();
        let query = design(1_000, p, 8);
        g.bench_function(format!("outlier_1k_queries/{name}"), |b| {
            b.iter(|| black_box(fitted.outlier(black_box(&query)).unwrap()))
        });
    }
    g.finish();
}

criterion_group!(
    benches,
    scalers,
    imputers,
    encoders,
    splitters,
    metrics,
    isolation_forest,
    local_outlier_factor
);
criterion_main!(benches);
