//! Card fraud on real labelled transactions, scored two ways off one design matrix.
//!
//! The dataset is the ULB / Worldline European card-fraud release: 284,807 transactions,
//! 492 of them fraudulent (0.1727%), features `V1..V28` the published PCA projection of the
//! raw transaction fields, plus `Amount`. It arrives already split into train / validation
//! / test parquet files, stratified.
//!
//! ```text
//! cargo run --release --example fraud -p hanzo-learn
//! ```
//!
//! Two arms off the SAME matrix, which is the comparison worth making: an isolation forest,
//! which never sees a label, against a logistic fit, which sees every one of them. Both are
//! summarised by [`metric::Curve`] — one sort, every number — and both are reported at the
//! operating point a review desk actually runs at, a fixed alarm budget.
//!
//! AVERAGE PRECISION IS THE HEADLINE, not the area under the ROC. At 0.17% prevalence a
//! detector that flags one in two hundred transactions is drowning in false alarms while its
//! ROC area still reads well, because the false-alarm rate divides by 284,315 negatives.
//! Average precision divides by what was flagged, so it cannot flatter.
//!
//! `Time` is dropped: it is seconds since the first transaction in the collection window, so
//! it is an index of the file rather than a property of the transaction, and it is the
//! feature that most readily manufactures a result that does not transfer. `original_index`
//! and `__index_level_0__` are dropped for the same reason — both are row numbers.

use std::fs::File;
use std::path::{Path, PathBuf};

use hanzo_learn::metric::Curve;
use hanzo_learn::{isolation, logistic, Fit, Matrix, Outlier, Result, Samples};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::Field;

/// The 29 features that describe a transaction. Every other column is a row number.
const FEATURE: [&str; 29] = [
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15",
    "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount",
];

const ROOT: &str = "/data/datasets/eu-cc-fraud/data";

/// Where a parquet column goes: into a feature slot, into the label, or nowhere.
enum Slot {
    Feature(usize),
    Label,
    Drop,
}

fn main() -> Result<()> {
    let split = ["train", "validation", "test"].map(|name| {
        let path = PathBuf::from(ROOT).join(format!("{name}-00000-of-00001.parquet"));
        let (x, y) = read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let positives = y.iter().filter(|&&t| t).count();
        println!(
            "{name:<10} n {:>6}  p {}  fraud {:>3}  ({:.4}%)",
            x.n(),
            x.p(),
            positives,
            100.0 * positives as f64 / x.n() as f64
        );
        (name, x, y)
    });
    println!();

    // UNSUPERVISED. Fit on the training design only — no label reaches the forest — then
    // score every split with the same fitted value.
    let forest = isolation::Config::new(100, 256, 0)?.fit(&split[0].1)?;
    println!(
        "isolation forest: {} trees of {} rows, seed {}",
        forest.trees(),
        forest.sample(),
        forest.config().seed()
    );
    for (name, x, y) in &split {
        report("isolation", name, y, &forest.outlier(x)?)?;
    }
    println!();

    // SUPERVISED baseline, for scale: the same matrix with the training labels attached.
    let labels: Vec<i64> = split[0].2.iter().map(|&t| i64::from(t)).collect();
    let model = logistic::Config::new().fit(&Samples::new(split[0].1.clone(), labels)?)?;
    println!(
        "logistic: C {}, intercept {}, {} Newton steps",
        model.config().c,
        model.config().intercept,
        model.iterations()
    );
    for (name, x, y) in &split {
        report("logistic", name, y, &model.probability(x)?)?;
    }

    Ok(())
}

/// One arm on one split: the two summaries, then what a fixed review capacity buys.
fn report(arm: &str, split: &str, truth: &[bool], score: &[f64]) -> Result<()> {
    let curve = Curve::of(truth, score)?;
    println!(
        "  {arm:<9} {split:<10} roc_auc {:.6}  average_precision {:.6}",
        curve.roc_auc(),
        curve.average_precision()
    );
    for budget in [0.005, 0.01] {
        let point = curve
            .at_alarm_rate(budget)
            .expect("a non-negative budget always has an operating point");
        let c = point.confusion;
        println!(
            "    alarm budget {:>4.1}%: flagged {:>5} of {:>6} ({:.3}%)  recall {:.4}  \
             precision {:.4}  f1 {:.4}  caught {:>3} of {:>3}",
            100.0 * budget,
            c.hit + c.alarm,
            c.total(),
            100.0 * (c.hit + c.alarm) as f64 / c.total() as f64,
            c.recall(),
            c.precision(),
            c.f1(),
            c.hit,
            c.hit + c.miss
        );
    }
    Ok(())
}

/// A split, as the design matrix and the event of interest.
///
/// The parquet record reader rather than arrow: the columns are read one row at a time and
/// laid straight into a row of the design, so nothing intermediate is materialised and the
/// example adds one dev dependency instead of two.
fn read(path: &Path) -> std::result::Result<(Matrix, Vec<bool>), Box<dyn std::error::Error>> {
    let reader = SerializedFileReader::new(File::open(path)?)?;
    // The column order is a property of the file, so where each column goes is settled once
    // against the schema rather than by name on every one of 284,807 rows.
    let plan: Vec<Slot> = reader
        .metadata()
        .file_metadata()
        .schema()
        .get_fields()
        .iter()
        .map(
            |f| match FEATURE.iter().position(|name| *name == f.name()) {
                Some(j) => Slot::Feature(j),
                None if f.name() == "Class" => Slot::Label,
                None => Slot::Drop,
            },
        )
        .collect();

    let mut rows = Vec::new();
    let mut truth = Vec::new();
    for row in reader.get_row_iter(None)? {
        let row = row?;
        let mut features = vec![0.0; FEATURE.len()];
        let mut label = None;
        for (slot, (_, field)) in plan.iter().zip(row.get_column_iter()) {
            match slot {
                Slot::Feature(j) => features[*j] = real(field)?,
                Slot::Label => label = Some(real(field)? != 0.0),
                Slot::Drop => {}
            }
        }
        rows.push(features);
        truth.push(label.ok_or("the file has no Class column")?);
    }
    Ok((Matrix::rows(&rows)?, truth))
}

/// A parquet value as a real.
///
/// `Class` and the index columns are `int64` in this file while the features are `double`,
/// so both widths are accepted here and anything else is refused rather than defaulted.
fn real(field: &Field) -> std::result::Result<f64, Box<dyn std::error::Error>> {
    match field {
        Field::Double(v) => Ok(*v),
        Field::Float(v) => Ok(f64::from(*v)),
        Field::Long(v) => Ok(*v as f64),
        Field::Int(v) => Ok(f64::from(*v)),
        other => Err(format!("{other:?} is not a number").into()),
    }
}
