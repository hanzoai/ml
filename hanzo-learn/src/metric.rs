//! Scoring, for problems where one class is rare.
//!
//! # One value, every metric
//!
//! Ranked scores against binary truth support a whole family of numbers, and every member
//! of that family is a function of the same two cumulative counts at the same set of
//! thresholds. scikit-learn recomputes those counts inside `roc_auc_score`, inside
//! `average_precision_score`, and again inside `precision_recall_curve`: three sorts of
//! the same data to answer three questions about it.
//!
//! Here the counts are the value ([`Curve`]) and every metric is a QUERY on it. So the
//! sort happens once, the numbers cannot disagree with each other, and the answers a risk
//! team actually needs — the cost-minimal threshold, the threshold that reaches a required
//! recall, the confusion at a chosen cut — are the same kind of query rather than three
//! more functions with their own passes over the data.
//!
//! ```text
//!                        Curve::of(truth, score)          one sort
//!                                  |
//!    +-------------+---------------+---------------+-------------------+
//!    |             |               |               |                   |
//!  roc_auc   average_precision  at(t)         cheapest(a, m)      at_recall(r)
//!                                  |
//!                              Confusion  -->  recall, precision, cost
//! ```
//!
//! # Why the truth is `&[bool]`
//!
//! A binary metric over `&[i64]` has to define what it does with a `2`, and every
//! implementation answers that differently. `&[bool]` has no such case: "invalid label"
//! is not an error these functions can return because it is not an input they can be
//! given. Resolve a multi-class label down to the event of interest at the boundary, where
//! the caller knows which class is the positive one.
//!
//! # Scale
//!
//! [`Curve::of`] sorts, so it needs the scores resident: `8·n` for the scores, `8·n` for
//! the permutation, `n` for the truth, and the curve itself is `O(d)` in the number of
//! DISTINCT scores. That is about `1.7·10⁷` samples per GB, so a 128 GB box reaches
//! `10⁹` samples with room to spare — but it is a bound, and it is `O(n)` rather than
//! `O(1)`, which is the honest difference between this module and [`crate::scale`].
//! A distribution-free streaming quantile sketch would remove it and is not implemented.

use crate::{Error, Result};

/// The cumulative counts of a ranked binary classifier at every distinct score.
///
/// The one value from which every ranked metric in this module is a query.
#[derive(Debug, Clone, PartialEq)]
pub struct Curve {
    threshold: Vec<f64>,
    hit: Vec<u64>,
    alarm: Vec<u64>,
    positive: u64,
    negative: u64,
}

impl Curve {
    /// Accumulate the curve of `score` against `truth`.
    ///
    /// One stable sort, descending. Samples that share a score share one threshold point,
    /// so a tie cannot be broken in a way that changes the answer.
    ///
    /// Fails on lengths that disagree, on a score that is not finite, and on truth that
    /// holds only one class — a curve over one class has no false-positive rate, so every
    /// query on it would have to answer for a denominator of zero.
    pub fn of(truth: &[bool], score: &[f64]) -> Result<Self> {
        if truth.len() != score.len() {
            return Err(Error::Shape(format!(
                "{} labels against {} scores",
                truth.len(),
                score.len()
            )));
        }
        if truth.is_empty() {
            return Err(Error::Shape(
                "a curve needs at least one sample".to_string(),
            ));
        }
        if let Some(i) = score.iter().position(|s| !s.is_finite()) {
            return Err(Error::Shape(format!(
                "score at index {i} is not finite, so it cannot be ranked"
            )));
        }
        let positive = truth.iter().filter(|&&t| t).count() as u64;
        let negative = truth.len() as u64 - positive;
        if positive == 0 || negative == 0 {
            return Err(Error::Classes(format!(
                "a curve needs both classes: {positive} positive of {}",
                truth.len()
            )));
        }

        let mut order: Vec<u32> = (0..truth.len() as u32).collect();
        order.sort_by(|&a, &b| score[b as usize].total_cmp(&score[a as usize]));

        let mut threshold = Vec::new();
        let (mut hit, mut alarm) = (Vec::new(), Vec::new());
        let (mut h, mut a) = (0u64, 0u64);
        let mut i = 0usize;
        while i < order.len() {
            let s = score[order[i] as usize];
            while i < order.len() && score[order[i] as usize] == s {
                if truth[order[i] as usize] {
                    h += 1;
                } else {
                    a += 1;
                }
                i += 1;
            }
            threshold.push(s);
            hit.push(h);
            alarm.push(a);
        }
        Ok(Self {
            threshold,
            hit,
            alarm,
            positive,
            negative,
        })
    }

    /// How many samples were positive.
    pub fn positives(&self) -> u64 {
        self.positive
    }

    /// How many samples were negative.
    pub fn negatives(&self) -> u64 {
        self.negative
    }

    /// The distinct scores, descending. One threshold per point of the curve.
    pub fn thresholds(&self) -> &[f64] {
        &self.threshold
    }

    /// Area under the receiver operating characteristic.
    ///
    /// The trapezoid rule over the curve, including the origin, which is what
    /// `roc_auc_score` computes. Equal to the probability that a random positive outranks
    /// a random negative, with ties counting a half.
    pub fn roc_auc(&self) -> f64 {
        let (p, n) = (self.positive as f64, self.negative as f64);
        let mut area = 0.0;
        let (mut previous_alarm, mut previous_hit) = (0.0, 0.0);
        for k in 0..self.threshold.len() {
            let (x, y) = (self.alarm[k] as f64 / n, self.hit[k] as f64 / p);
            area += (x - previous_alarm) * (y + previous_hit) / 2.0;
            previous_alarm = x;
            previous_hit = y;
        }
        area
    }

    /// Average precision: the precision-recall curve summarised as a step function, which
    /// is what `average_precision_score` computes.
    ///
    /// Not the trapezoid area under the precision-recall curve. The difference is not
    /// rounding — interpolating between two operating points on a precision-recall curve
    /// claims an operating point that may not exist — and it is why this is a separate
    /// query rather than `roc_auc` with different axes.
    pub fn average_precision(&self) -> f64 {
        let p = self.positive as f64;
        let mut total = 0.0;
        let mut previous_recall = 0.0;
        for k in 0..self.threshold.len() {
            let flagged = (self.hit[k] + self.alarm[k]) as f64;
            let precision = if flagged == 0.0 {
                0.0
            } else {
                self.hit[k] as f64 / flagged
            };
            let recall = self.hit[k] as f64 / p;
            total += (recall - previous_recall) * precision;
            previous_recall = recall;
        }
        total
    }

    /// The precision-recall curve as `(precision, recall, threshold)`, beginning at the
    /// point where nothing is flagged.
    ///
    /// # One ordering for both curves, and it is not scikit-learn's
    ///
    /// Three arrays of the SAME length, ordered exactly like [`Curve::roc`]: index zero is
    /// the flag-nothing operating point — recall zero, precision one by convention,
    /// threshold `+inf` — and each later index lowers the threshold by one distinct score.
    /// So the index means the same thing in both curves, and a caller reading them together
    /// does not have to hold two conventions.
    ///
    /// scikit-learn returns this one REVERSED relative to its own `roc_curve`, with recall
    /// decreasing, the endpoint appended last, and a `thresholds` array one element shorter
    /// than the other two. That asymmetry is a plotting convenience in a library whose
    /// curves are usually handed straight to a chart; it is not a property of the curve, and
    /// reproducing it here would mean two orderings and an off-by-one for callers to trip
    /// over. `tests/sklearn.rs` maps between the two explicitly and asserts the mapped
    /// arrays agree exactly, so the divergence is a stated presentation choice rather than
    /// an unchecked one.
    ///
    /// EVERY distinct threshold is kept. An earlier version of this method truncated the
    /// arrays once recall reached one, which is what scikit-learn did up to 1.6; 1.9 keeps
    /// the whole curve, and truncating discards real operating points — the ones that raise
    /// precision at no cost in recall, which is exactly the region an operator tuning a
    /// threshold cares about.
    pub fn precision_recall(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let p = self.positive as f64;
        let mut precision = Vec::with_capacity(self.threshold.len() + 1);
        let mut recall = Vec::with_capacity(self.threshold.len() + 1);
        let mut threshold = Vec::with_capacity(self.threshold.len() + 1);
        precision.push(1.0);
        recall.push(0.0);
        threshold.push(f64::INFINITY);
        for k in 0..self.threshold.len() {
            let flagged = (self.hit[k] + self.alarm[k]) as f64;
            precision.push(if flagged == 0.0 {
                0.0
            } else {
                self.hit[k] as f64 / flagged
            });
            recall.push(self.hit[k] as f64 / p);
            threshold.push(self.threshold[k]);
        }
        (precision, recall, threshold)
    }

    /// The receiver operating characteristic as `(alarm rate, hit rate, threshold)`,
    /// beginning at the origin where nothing is flagged.
    ///
    /// Every distinct threshold is kept, which is `roc_curve(drop_intermediate=False)` —
    /// scikit-learn's default drops collinear interior points to make a lighter plot, which
    /// changes the arrays it returns but not the area under them.
    pub fn roc(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (p, n) = (self.positive as f64, self.negative as f64);
        let mut alarm = Vec::with_capacity(self.threshold.len() + 1);
        let mut hit = Vec::with_capacity(self.threshold.len() + 1);
        let mut threshold = Vec::with_capacity(self.threshold.len() + 1);
        alarm.push(0.0);
        hit.push(0.0);
        threshold.push(f64::INFINITY);
        for k in 0..self.threshold.len() {
            alarm.push(self.alarm[k] as f64 / n);
            hit.push(self.hit[k] as f64 / p);
            threshold.push(self.threshold[k]);
        }
        (alarm, hit, threshold)
    }

    /// The confusion at one cut: a sample is flagged when its score is at or above
    /// `threshold`.
    ///
    /// A threshold above every score flags nothing, which is a legitimate operating point
    /// and the one [`Curve::cheapest`] returns when flagging is never worth it.
    pub fn at(&self, threshold: f64) -> Confusion {
        // Thresholds descend, so the flagged set is the prefix whose threshold is at or
        // above the cut: find the last such point.
        let mut point = None;
        let (mut low, mut high) = (0usize, self.threshold.len());
        while low < high {
            let mid = (low + high) / 2;
            if self.threshold[mid] >= threshold {
                point = Some(mid);
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        match point {
            None => Confusion {
                hit: 0,
                miss: self.positive,
                alarm: 0,
                reject: self.negative,
            },
            Some(k) => Confusion {
                hit: self.hit[k],
                miss: self.positive - self.hit[k],
                alarm: self.alarm[k],
                reject: self.negative - self.alarm[k],
            },
        }
    }

    /// The operating point of least expected cost, given what one false alarm and one
    /// missed positive each cost.
    ///
    /// This is the query that makes a ranked score into a decision, and the reason the
    /// counts are a value: it is a walk over points already computed, not another pass
    /// over the data. Ties go to the higher threshold, which flags fewer samples.
    ///
    /// Fails on a negative cost, which would make flagging everything free.
    pub fn cheapest(&self, alarm: f64, miss: f64) -> Result<Operating> {
        if !(alarm >= 0.0) || !(miss >= 0.0) || !alarm.is_finite() || !miss.is_finite() {
            return Err(Error::Config(format!(
                "costs must be finite and not negative, not alarm {alarm} miss {miss}"
            )));
        }
        // Flagging nothing is an operating point too, and the cheapest one when misses are
        // free. Starting from it means the answer is never worse than doing nothing.
        let mut best = Operating {
            threshold: f64::INFINITY,
            confusion: self.at(f64::INFINITY),
        };
        let mut least = self.positive as f64 * miss;
        for k in 0..self.threshold.len() {
            let cost = self.alarm[k] as f64 * alarm + (self.positive - self.hit[k]) as f64 * miss;
            if cost < least {
                least = cost;
                best = Operating {
                    threshold: self.threshold[k],
                    confusion: Confusion {
                        hit: self.hit[k],
                        miss: self.positive - self.hit[k],
                        alarm: self.alarm[k],
                        reject: self.negative - self.alarm[k],
                    },
                };
            }
        }
        Ok(best)
    }

    /// The highest threshold whose recall is at least `wanted` — the most precise way to
    /// catch that share of the positives.
    ///
    /// Nothing if no threshold reaches it, which can only happen for `wanted` above one.
    pub fn at_recall(&self, wanted: f64) -> Option<Operating> {
        let p = self.positive as f64;
        for k in 0..self.threshold.len() {
            if self.hit[k] as f64 / p >= wanted {
                return Some(Operating {
                    threshold: self.threshold[k],
                    confusion: self.at(self.threshold[k]),
                });
            }
        }
        None
    }

    /// The lowest threshold whose false-alarm rate is at most `budget` — the most recall
    /// available for a fixed review capacity.
    ///
    /// Nothing if even the highest threshold exceeds the budget.
    pub fn at_alarm_rate(&self, budget: f64) -> Option<Operating> {
        let n = self.negative as f64;
        let mut found = None;
        for k in 0..self.threshold.len() {
            if self.alarm[k] as f64 / n <= budget {
                found = Some(self.threshold[k]);
            } else {
                break;
            }
        }
        match found {
            Some(t) => Some(Operating {
                threshold: t,
                confusion: self.at(t),
            }),
            None if budget >= 0.0 => Some(Operating {
                threshold: f64::INFINITY,
                confusion: self.at(f64::INFINITY),
            }),
            None => None,
        }
    }
}

/// A chosen cut, and what it does.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Operating {
    /// Flag a sample when its score is at or above this. Infinite means flag nothing.
    pub threshold: f64,
    /// What that cut produces.
    pub confusion: Confusion,
}

/// The two-by-two table of a binary decision, in detection-theory names.
///
/// `hit` and `miss` are the positives found and not found; `alarm` and `reject` are the
/// negatives flagged and not flagged. Those names are used in place of "true positive" and
/// the rest because the four of them are read together and the abbreviations are the
/// leading source of transposed confusion matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Confusion {
    /// Positives correctly flagged.
    pub hit: u64,
    /// Positives not flagged.
    pub miss: u64,
    /// Negatives wrongly flagged.
    pub alarm: u64,
    /// Negatives correctly not flagged.
    pub reject: u64,
}

impl Confusion {
    /// Count a set of decisions against the truth.
    pub fn of(truth: &[bool], flagged: &[bool]) -> Result<Self> {
        if truth.len() != flagged.len() {
            return Err(Error::Shape(format!(
                "{} labels against {} decisions",
                truth.len(),
                flagged.len()
            )));
        }
        let mut c = Confusion {
            hit: 0,
            miss: 0,
            alarm: 0,
            reject: 0,
        };
        for (&t, &f) in truth.iter().zip(flagged) {
            match (t, f) {
                (true, true) => c.hit += 1,
                (true, false) => c.miss += 1,
                (false, true) => c.alarm += 1,
                (false, false) => c.reject += 1,
            }
        }
        Ok(c)
    }

    /// The table as `confusion_matrix` returns it for labels `[false, true]`, row-major:
    /// `[reject, alarm, miss, hit]`. Truth is the row, the decision is the column.
    pub fn matrix(&self) -> [u64; 4] {
        [self.reject, self.alarm, self.miss, self.hit]
    }

    /// How many samples in all.
    pub fn total(&self) -> u64 {
        self.hit + self.miss + self.alarm + self.reject
    }

    /// The share of positives found. Zero when there are no positives.
    pub fn recall(&self) -> f64 {
        ratio(self.hit, self.hit + self.miss)
    }

    /// The share of flagged samples that were positive. Zero when nothing is flagged,
    /// which is what `precision_score` reports for that case.
    pub fn precision(&self) -> f64 {
        ratio(self.hit, self.hit + self.alarm)
    }

    /// The share of negatives correctly left alone.
    pub fn specificity(&self) -> f64 {
        ratio(self.reject, self.reject + self.alarm)
    }

    /// The share of negatives wrongly flagged — the review load a fixed capacity has to
    /// absorb.
    pub fn alarm_rate(&self) -> f64 {
        ratio(self.alarm, self.reject + self.alarm)
    }

    /// The harmonic mean of precision and recall. Zero when either is zero.
    pub fn f1(&self) -> f64 {
        let (p, r) = (self.precision(), self.recall());
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }

    /// The share of decisions that were right.
    ///
    /// Reported for completeness and named here as the wrong summary for rare events: a
    /// detector that flags nothing scores well on it whenever the positives are rare.
    pub fn accuracy(&self) -> f64 {
        ratio(self.hit + self.reject, self.total())
    }

    /// What this cut costs, given the price of one false alarm and one missed positive.
    pub fn cost(&self, alarm: f64, miss: f64) -> f64 {
        self.alarm as f64 * alarm + self.miss as f64 * miss
    }
}

fn ratio(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 / whole as f64
    }
}

/// The mean negative log likelihood of the truth under predicted probabilities.
///
/// Probabilities are pulled inside `[eps, 1 - eps]` before the logarithm, which is what
/// `log_loss` does: a confident wrong answer at exactly zero would otherwise make the whole
/// mean infinite, and one sample would erase every other.
///
/// Fails on lengths that disagree and on a value outside `[0, 1]` — a number that is not a
/// probability has no log likelihood, and clipping it silently would report a loss for a
/// model that is not producing probabilities at all.
pub fn log_loss(truth: &[bool], probability: &[f64]) -> Result<f64> {
    if truth.len() != probability.len() {
        return Err(Error::Shape(format!(
            "{} labels against {} probabilities",
            truth.len(),
            probability.len()
        )));
    }
    if truth.is_empty() {
        return Err(Error::Shape(
            "a log loss needs at least one sample".to_string(),
        ));
    }
    if let Some(i) = probability
        .iter()
        .position(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
    {
        return Err(Error::Shape(format!(
            "value {} at index {i} is not a probability",
            probability[i]
        )));
    }
    let eps = f64::EPSILON;
    let mut total = 0.0;
    for (&t, &p) in truth.iter().zip(probability) {
        let p = p.clamp(eps, 1.0 - eps);
        total -= if t { p.ln() } else { (1.0 - p).ln() };
    }
    Ok(total / truth.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A separable ranking: every positive outranks every negative, so the area is one and
    /// average precision is one, whatever the scores happen to be.
    #[test]
    fn a_perfect_ranking_scores_one_on_both_summaries() {
        let truth = [true, true, false, false];
        let score = [0.9, 0.8, 0.2, 0.1];
        let c = Curve::of(&truth, &score).unwrap();
        assert!((c.roc_auc() - 1.0).abs() < 1e-15);
        assert!((c.average_precision() - 1.0).abs() < 1e-15);
    }

    /// Every score identical: the ranking carries no information, so the area is a half.
    /// This is the case a tie-break would silently get wrong.
    #[test]
    fn one_score_for_everything_is_a_coin_flip() {
        let truth = [true, false, true, false];
        let score = [0.5; 4];
        let c = Curve::of(&truth, &score).unwrap();
        assert_eq!(c.thresholds().len(), 1);
        assert!((c.roc_auc() - 0.5).abs() < 1e-15);
    }

    #[test]
    fn a_curve_over_one_class_is_refused() {
        assert!(Curve::of(&[true, true], &[0.1, 0.2]).is_err());
        assert!(Curve::of(&[false, false], &[0.1, 0.2]).is_err());
        assert!(Curve::of(&[true, false], &[0.1]).is_err());
        assert!(Curve::of(&[true, false], &[0.1, f64::NAN]).is_err());
    }

    #[test]
    fn a_cut_agrees_with_counting_the_decisions_by_hand() {
        let truth = [true, false, true, true, false, false];
        let score = [0.9, 0.8, 0.7, 0.4, 0.3, 0.1];
        let c = Curve::of(&truth, &score).unwrap();
        for &t in &[1.5, 0.9, 0.75, 0.7, 0.4, 0.05] {
            let flagged: Vec<bool> = score.iter().map(|&s| s >= t).collect();
            assert_eq!(
                c.at(t),
                Confusion::of(&truth, &flagged).unwrap(),
                "threshold {t}"
            );
        }
    }

    #[test]
    fn the_cheapest_cut_is_the_cheapest_of_every_cut() {
        let truth = [true, false, true, true, false, false, false, true];
        let score = [0.9, 0.85, 0.7, 0.4, 0.35, 0.2, 0.15, 0.1];
        let c = Curve::of(&truth, &score).unwrap();
        for (alarm, miss) in [(1.0, 1.0), (1.0, 10.0), (10.0, 1.0), (1.0, 0.0), (0.0, 1.0)] {
            let best = c.cheapest(alarm, miss).unwrap();
            let least = best.confusion.cost(alarm, miss);
            // Nothing flagged, and every distinct cut, must all be at least as expensive.
            assert!(c.at(f64::INFINITY).cost(alarm, miss) >= least - 1e-12);
            for &t in c.thresholds() {
                assert!(
                    c.at(t).cost(alarm, miss) >= least - 1e-12,
                    "cut {t} beats the cheapest at alarm {alarm} miss {miss}"
                );
            }
        }
        // Free misses mean flagging nothing, which is an operating point and not an error.
        assert!(c.cheapest(1.0, 0.0).unwrap().threshold.is_infinite());
        assert!(c.cheapest(-1.0, 1.0).is_err());
    }

    #[test]
    fn a_required_recall_is_met_at_the_highest_threshold_that_meets_it() {
        let truth = [true, false, true, true, false, false];
        let score = [0.9, 0.8, 0.7, 0.4, 0.3, 0.1];
        let c = Curve::of(&truth, &score).unwrap();
        let o = c.at_recall(0.6).unwrap();
        assert!(o.confusion.recall() >= 0.6);
        // Nothing above it also meets the requirement.
        for &t in c.thresholds().iter().take_while(|&&t| t > o.threshold) {
            assert!(c.at(t).confusion_recall_below(0.6));
        }
        assert!(c.at_recall(1.5).is_none());
        assert!((c.at_recall(1.0).unwrap().confusion.recall() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn an_alarm_budget_is_respected() {
        let truth = [true, false, true, true, false, false];
        let score = [0.9, 0.8, 0.7, 0.4, 0.3, 0.1];
        let c = Curve::of(&truth, &score).unwrap();
        let o = c.at_alarm_rate(0.34).unwrap();
        assert!(o.confusion.alarm_rate() <= 0.34 + 1e-12);
        // A budget of nothing still returns a point: flag only what is free to flag.
        assert!(c.at_alarm_rate(0.0).unwrap().confusion.alarm == 0);
    }

    #[test]
    fn the_table_is_laid_out_the_way_confusion_matrix_lays_it_out() {
        let c = Confusion::of(&[true, true, false, false], &[true, false, true, false]).unwrap();
        assert_eq!(c.matrix(), [1, 1, 1, 1]);
        assert_eq!(c.hit, 1);
        assert_eq!(c.miss, 1);
        assert_eq!(c.alarm, 1);
        assert_eq!(c.reject, 1);
        assert!((c.f1() - 0.5).abs() < 1e-15);
        assert!((c.accuracy() - 0.5).abs() < 1e-15);
    }

    #[test]
    fn a_flagless_cut_reports_zero_precision_rather_than_dividing_by_zero() {
        let c = Confusion {
            hit: 0,
            miss: 3,
            alarm: 0,
            reject: 7,
        };
        assert_eq!(c.precision(), 0.0);
        assert_eq!(c.recall(), 0.0);
        assert_eq!(c.f1(), 0.0);
        assert!((c.accuracy() - 0.7).abs() < 1e-15);
    }

    #[test]
    fn a_log_loss_is_refused_for_something_that_is_not_a_probability() {
        assert!(log_loss(&[true], &[1.2]).is_err());
        assert!(log_loss(&[true], &[-0.1]).is_err());
        assert!(log_loss(&[true, false], &[0.5]).is_err());
        // A confident wrong answer is large and finite, not infinite.
        let l = log_loss(&[true], &[0.0]).unwrap();
        assert!(l > 30.0 && l.is_finite(), "{l}");
    }

    impl Confusion {
        fn confusion_recall_below(&self, wanted: f64) -> bool {
            self.recall() < wanted
        }
    }
}
