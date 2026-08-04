//! Resampling plans: which rows a fit may see, and which are held back.
//!
//! # A plan names rows, it does not copy them
//!
//! Every function here returns [`Split`] values holding indices. Nothing is copied, so a
//! five-fold plan over a design of any size costs `8·n` bytes once and not five copies of
//! the data, and the plan can be written down, sent somewhere, or compared against the one
//! a Python baseline used. Gathering the rows is a separate step the caller takes when it
//! actually needs contiguous data.
//!
//! # There is no unseeded shuffle
//!
//! [`Order::Shuffled`] carries its seed. scikit-learn lets `shuffle=True` run with
//! `random_state=None`, which produces a plan nobody can reproduce — including the person
//! who ran it, ten seconds later. A resampling plan is the decision a whole evaluation
//! rests on, so an unreproducible one is not a value this module can return.
//!
//! # Bit-exact with scikit-learn
//!
//! The shuffled orders come from [`crate::twister`], which is `numpy`'s generator drawn
//! `numpy`'s way, and the fold arithmetic follows scikit-learn's. So the plans here are the
//! same plans, index for index, which is what makes a Rust model and a Python baseline
//! comparable rather than merely both evaluated. The fixtures assert exactly that.

use std::collections::BTreeMap;

use crate::{Class, Error, Result};

/// Which rows a fit may see, and which are held back. Indices ascend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Split {
    train: Vec<usize>,
    test: Vec<usize>,
}

impl Split {
    /// Rows the fit may see.
    pub fn train(&self) -> &[usize] {
        &self.train
    }

    /// Rows held back.
    pub fn test(&self) -> &[usize] {
        &self.test
    }
}

/// Whether rows keep their order, and if not, under which seed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// Rows keep the order they arrived in. Folds are contiguous blocks.
    ///
    /// Correct when the row order carries no information, and wrong when it does — a table
    /// sorted by label would give a fold with one class in it.
    Sequential,
    /// Rows are permuted under this seed before folds are cut.
    Shuffled(u32),
}

impl Order {
    fn arrange(&self, n: usize) -> Vec<usize> {
        match self {
            Self::Sequential => (0..n).collect(),
            Self::Shuffled(seed) => crate::twister::Twister::seed(*seed).permutation(n),
        }
    }
}

/// Hold back a proportion of the rows.
///
/// `proportion` is the share held back, rounded UP to a whole number of rows, which is
/// what `train_test_split` does with its `test_size`. Under [`Order::Sequential`] the held
/// back rows are the last ones, which is the right plan for a time series and the wrong one
/// for anything sorted.
///
/// Fails on a proportion outside `(0, 1)`, and when the rounding leaves either side empty:
/// a fit with nothing to fit on, or a held-back set with nothing in it, is a plan that
/// cannot answer the question it was made for.
pub fn train_test(n: usize, proportion: f64, order: Order) -> Result<Split> {
    if !(proportion > 0.0 && proportion < 1.0) {
        return Err(Error::Config(format!(
            "a held-back proportion lies strictly between 0 and 1, not {proportion}"
        )));
    }
    let held = (proportion * n as f64).ceil() as usize;
    if held == 0 || held >= n {
        return Err(Error::Config(format!(
            "holding back {proportion} of {n} rows leaves {} for the fit",
            n.saturating_sub(held)
        )));
    }
    match order {
        Order::Sequential => Ok(Split {
            train: (0..n - held).collect(),
            test: (n - held..n).collect(),
        }),
        Order::Shuffled(seed) => {
            let permutation = crate::twister::Twister::seed(seed).permutation(n);
            let mut test = permutation[..held].to_vec();
            let mut train = permutation[held..].to_vec();
            // scikit-learn returns these in permutation order. Ascending instead, so a
            // plan is a set of rows and not also a shuffle of them: the row order a fit
            // sees is then the caller's own, and two plans that hold back the same rows
            // compare equal.
            train.sort_unstable();
            test.sort_unstable();
            Ok(Split { train, test })
        }
    }
}

/// Cut the rows into `folds` parts, each part held back in turn.
///
/// Sizes differ by at most one, and the larger parts come first, which is `KFold`'s
/// arithmetic. Indices ascend within both sides of every split, shuffled or not — the
/// shuffle decides which rows share a fold, never the order they are visited in.
pub fn folds(n: usize, folds: usize, order: Order) -> Result<Vec<Split>> {
    if folds < 2 {
        return Err(Error::Config(format!(
            "cross validation needs at least 2 folds, not {folds}"
        )));
    }
    if folds > n {
        return Err(Error::Config(format!("{folds} folds do not fit {n} rows")));
    }
    let arranged = order.arrange(n);
    let mut plans = Vec::with_capacity(folds);
    let base = n / folds;
    let extra = n % folds;
    let mut start = 0usize;
    for f in 0..folds {
        let size = base + usize::from(f < extra);
        let mut held = vec![false; n];
        for &row in &arranged[start..start + size] {
            held[row] = true;
        }
        plans.push(Split {
            train: (0..n).filter(|&i| !held[i]).collect(),
            test: (0..n).filter(|&i| held[i]).collect(),
        });
        start += size;
    }
    Ok(plans)
}

/// Cut the rows into `folds` parts that each hold the class proportions of the whole.
///
/// This is the plan a rare-event problem needs: a plain [`folds`] over data with a one per
/// cent positive rate will, often enough to matter, produce a fold with no positives at
/// all, and every threshold-aware metric on that fold is then undefined.
///
/// Fails when any class has fewer members than there are folds. scikit-learn warns and
/// carries on, producing folds with no members of that class; the metric computed on such a
/// fold is not a number anyone should average, so it is refused here with the class named.
pub fn stratified(labels: &[Class], folds: usize, order: Order) -> Result<Vec<Split>> {
    let n = labels.len();
    if folds < 2 {
        return Err(Error::Config(format!(
            "cross validation needs at least 2 folds, not {folds}"
        )));
    }
    if n == 0 {
        return Err(Error::Shape("no labels to stratify by".to_string()));
    }

    // Classes are numbered by ORDER OF FIRST APPEARANCE, which is what scikit-learn does
    // internally. That makes the plan depend only on the PATTERN of the labels and not on
    // what they are called, so a vocabulary numbered differently yields the same folds.
    let mut rank: BTreeMap<Class, usize> = BTreeMap::new();
    let mut encoded = Vec::with_capacity(n);
    for &label in labels {
        let next = rank.len();
        let r = *rank.entry(label).or_insert(next);
        encoded.push(r);
    }
    let classes = rank.len();
    let mut count = vec![0usize; classes];
    for &c in &encoded {
        count[c] += 1;
    }
    if let Some(c) = count.iter().position(|&k| k < folds) {
        let named = rank
            .iter()
            .find(|(_, &r)| r == c)
            .map(|(class, _)| class.index())
            .unwrap_or(c);
        return Err(Error::Classes(format!(
            "class {named} has {} members, fewer than the {folds} folds, so some fold \
             would hold none of it",
            count[c]
        )));
    }

    // How many members of each class each fold holds back: deal the sorted labels round
    // robin, which is scikit-learn's allocation and keeps every fold's class proportions
    // within one member of the whole.
    let mut sorted = encoded.clone();
    sorted.sort_unstable();
    let mut allocation = vec![vec![0usize; classes]; folds];
    for (position, &c) in sorted.iter().enumerate() {
        allocation[position % folds][c] += 1;
    }

    // For each class, the fold each of its members goes to, in row order.
    let mut assigned = vec![0usize; n];
    for c in 0..classes {
        let mut fold_of_member: Vec<usize> = (0..folds)
            .flat_map(|f| std::iter::repeat_n(f, allocation[f][c]))
            .collect();
        if let Order::Shuffled(seed) = order {
            // Seeded per class exactly as scikit-learn does it: one generator walks the
            // classes in order, so the streams are consumed in the same sequence.
            crate::twister::Twister::seed(seed).shuffle(&mut fold_of_member);
        }
        let mut next = 0usize;
        for i in 0..n {
            if encoded[i] == c {
                assigned[i] = fold_of_member[next];
                next += 1;
            }
        }
    }

    Ok((0..folds)
        .map(|f| Split {
            train: (0..n).filter(|&i| assigned[i] != f).collect(),
            test: (0..n).filter(|&i| assigned[i] == f).collect(),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(pattern: &[usize]) -> Vec<Class> {
        pattern.iter().map(|&c| Class::at(c)).collect()
    }

    #[test]
    fn a_plan_partitions_the_rows_exactly_once() {
        for order in [
            Order::Sequential,
            Order::Shuffled(0),
            Order::Shuffled(12345),
        ] {
            for (n, k) in [(10usize, 2usize), (10, 3), (97, 5), (7, 7)] {
                let plans = folds(n, k, order).unwrap();
                assert_eq!(plans.len(), k);
                let mut seen = vec![0usize; n];
                for plan in &plans {
                    assert_eq!(plan.train().len() + plan.test().len(), n);
                    // Ascending, on both sides, always.
                    assert!(plan.train().windows(2).all(|w| w[0] < w[1]));
                    assert!(plan.test().windows(2).all(|w| w[0] < w[1]));
                    for &i in plan.test() {
                        seen[i] += 1;
                    }
                }
                // Every row is held back exactly once across the whole plan.
                assert!(seen.iter().all(|&c| c == 1), "n={n} k={k}");
                // Fold sizes differ by at most one.
                let sizes: Vec<usize> = plans.iter().map(|p| p.test().len()).collect();
                let (lo, hi) = (*sizes.iter().min().unwrap(), *sizes.iter().max().unwrap());
                assert!(hi - lo <= 1, "sizes {sizes:?}");
            }
        }
    }

    #[test]
    fn a_shuffled_plan_holds_back_different_rows_than_a_sequential_one() {
        let plain = folds(20, 4, Order::Sequential).unwrap();
        let mixed = folds(20, 4, Order::Shuffled(3)).unwrap();
        assert_eq!(plain[0].test(), &[0, 1, 2, 3, 4]);
        assert_ne!(plain[0].test(), mixed[0].test());
        // The same seed is the same plan.
        assert_eq!(mixed, folds(20, 4, Order::Shuffled(3)).unwrap());
    }

    #[test]
    fn every_fold_carries_the_class_proportions() {
        // Ten per cent positive, which is where a plain fold plan starts producing folds
        // with nothing in them.
        let pattern: Vec<usize> = (0..100).map(|i| usize::from(i % 10 == 0)).collect();
        let y = labels(&pattern);
        for order in [Order::Sequential, Order::Shuffled(11)] {
            for plan in stratified(&y, 5, order).unwrap() {
                let positives = plan.test().iter().filter(|&&i| pattern[i] == 1).count();
                assert_eq!(positives, 2, "each fold gets its share of the rare class");
                assert_eq!(plan.test().len(), 20);
            }
        }
    }

    #[test]
    fn a_class_too_rare_to_stratify_is_refused_with_its_name() {
        let y = labels(&[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]);
        let e = stratified(&y, 5, Order::Sequential).unwrap_err();
        assert!(format!("{e}").contains("class 1 has 2 members"), "{e}");
        assert!(stratified(&y, 2, Order::Sequential).is_ok());
    }

    #[test]
    fn a_stratified_plan_does_not_care_what_the_classes_are_called() {
        let a = labels(&[0, 1, 0, 1, 0, 1, 0, 1]);
        let b = labels(&[7, 3, 7, 3, 7, 3, 7, 3]);
        assert_eq!(
            stratified(&a, 2, Order::Sequential).unwrap(),
            stratified(&b, 2, Order::Sequential).unwrap()
        );
    }

    #[test]
    fn a_held_back_share_rounds_up_and_refuses_to_empty_either_side() {
        let s = train_test(10, 0.25, Order::Sequential).unwrap();
        assert_eq!(s.test(), &[7, 8, 9]);
        assert_eq!(s.train().len(), 7);
        assert!(train_test(10, 0.0, Order::Sequential).is_err());
        assert!(train_test(10, 1.0, Order::Sequential).is_err());
        assert!(train_test(2, 0.9, Order::Sequential).is_err());
        // Same seed, same rows held back.
        assert_eq!(
            train_test(50, 0.2, Order::Shuffled(1)).unwrap(),
            train_test(50, 0.2, Order::Shuffled(1)).unwrap()
        );
        assert_ne!(
            train_test(50, 0.2, Order::Shuffled(1)).unwrap(),
            train_test(50, 0.2, Order::Shuffled(2)).unwrap()
        );
    }

    #[test]
    fn a_fold_count_that_cannot_work_is_refused() {
        assert!(folds(10, 1, Order::Sequential).is_err());
        assert!(folds(10, 11, Order::Sequential).is_err());
        assert!(folds(10, 10, Order::Sequential).is_ok());
    }
}
