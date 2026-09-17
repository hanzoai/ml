//! Gradient boosted regression trees: the ensemble.
//!
//! ```text
//!     F_0(x) = mean(y)
//!     F_m(x) = F_{m-1}(x) + rate * t_m(x),   t_m fitted to  y - F_{m-1}
//! ```
//!
//! Under squared error the negative gradient of the loss IS the residual, so each round
//! fits a tree to what the ensemble so far got wrong, and the leaf value needs no line
//! search — the minimiser of squared error over a leaf is the mean of the residuals in
//! it, which is what the tree already computes. That is why this file is short and why
//! it is `squared_error` only: another loss needs a leaf-value step, and shipping one
//! loss that is exactly right beats two where the second is subtly not.
//!
//! Verified against scikit-learn's own arithmetic rather than assumed: its
//! `GradientBoostingRegressor` initialises at `mean(y)` (measured, exact), applies no
//! leaf update for this loss (measured: leaf values equal the residual means to 5.6e-17),
//! and predicts by accumulating `rate * leaf` in round order — walking its dumped trees
//! that way reproduces its own predictions to 0.0.
//!
//! # Why boosting and not a random forest
//!
//! A forest's answer depends on its bootstrap draw and its feature subsampling, so
//! matching scikit-learn's forest means reproducing scikit-learn's RNG — its exact
//! stream, consumed in its exact order. That is a port of an implementation detail, not
//! of an algorithm, and it would be the only thing the test proved. Boosting at
//! `subsample = 1.0` draws no randomness at all, so the ensemble is a function of the
//! data and the comparison is about the mathematics.
//!
//! There is a caveat and it is in [`crate::tree`]: at `min_leaf = 1` the TREES are not a
//! function of the data either, for a reason that has nothing to do with bootstrapping.
//! Read that module header before choosing hyperparameters you intend to reproduce.

use crate::address::{Address, Digest};
use crate::data::{mean, Matrix, Samples};
use crate::error::{Error, Result};
use crate::tree::{self, Tree};
use crate::{Fit, Predict};

/// How to fit a boosted ensemble.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Config {
    /// How many trees.
    pub rounds: usize,

    /// How much of each tree to believe. Shrinkage: smaller needs more rounds and
    /// generalises better.
    pub rate: f64,

    /// How to grow each tree.
    ///
    /// Composed rather than flattened. scikit-learn spreads `max_depth`,
    /// `min_samples_leaf` and the rest across the ensemble's own signature, which is why
    /// it needs `get_params(deep=True)` to find them again. Here the tree's
    /// hyperparameters are the tree's, and the ensemble holds one of them.
    pub tree: tree::Config,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            rounds: 100,
            rate: 0.1,
            tree: tree::Config::default(),
        }
    }
}

impl Config {
    /// The default: 100 rounds at 0.1, depth-3 trees.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reproducible by construction: a leaf bound that puts the tie described in
    /// [`crate::tree`] out of reach.
    ///
    /// Offered because the honest default is scikit-learn's default, and scikit-learn's
    /// default is not reproducible. Rather than quietly departing from it, both are
    /// reachable and the difference is named.
    pub fn reproducible() -> Self {
        Self {
            tree: tree::Config {
                min_leaf: 8,
                ..tree::Config::default()
            },
            ..Self::default()
        }
    }

    fn validate(&self) -> Result<()> {
        if self.rounds == 0 {
            return Err(Error::Config(
                "0 rounds is a model that only ever answers the mean; ask for that with \
                 a mean, not with an ensemble"
                    .into(),
            ));
        }
        if !(self.rate.is_finite() && self.rate > 0.0) {
            return Err(Error::Config(format!(
                "rate is {} — it must be finite and positive",
                self.rate
            )));
        }
        Ok(())
    }
}

/// A fitted boosted ensemble: a VALUE.
#[derive(Clone, PartialEq, Debug)]
pub struct Model {
    config: Config,
    base: f64,
    trees: Vec<Tree>,
    features: usize,
}

impl Model {
    /// What the ensemble answers before any tree: `mean(y)`.
    pub fn base(&self) -> f64 {
        self.base
    }

    /// The trees, in the order they were fitted. Order is part of the value: prediction
    /// accumulates in it, so two ensembles holding the same trees differently ordered
    /// answer differently in the last bits.
    pub fn trees(&self) -> &[Tree] {
        &self.trees
    }

    /// The hyperparameters this value was produced under.
    pub fn config(&self) -> &Config {
        &self.config
    }
}

impl crate::Model for Model {
    fn address(&self) -> Address {
        let mut d = Digest::new("hanzo.learn.boost")
            .size(self.config.rounds)
            .real(self.config.rate)
            .size(self.config.tree.depth)
            .size(self.config.tree.min_split)
            .size(self.config.tree.min_leaf)
            .size(self.features)
            .real(self.base)
            .size(self.trees.len());
        for t in &self.trees {
            d = t.digest(d);
        }
        d.finish()
    }

    fn features(&self) -> usize {
        self.features
    }
}

impl Predict for Model {
    /// A real number: this is a regressor.
    type Answer = f64;

    fn predict(&self, x: &Matrix) -> Result<Vec<f64>> {
        if x.p() != self.features {
            return Err(Error::Shape(format!(
                "fitted on {} features, asked about {}",
                self.features,
                x.p()
            )));
        }
        Ok((0..x.n())
            .map(|i| {
                let row = x.row(i);
                // Accumulated in ROUND ORDER, one `rate * leaf` at a time — the order
                // scikit-learn accumulates in. Summing the leaves first and scaling once
                // is algebraically identical and differs in the last bits.
                self.trees
                    .iter()
                    .fold(self.base, |acc, t| acc + self.config.rate * t.value(row))
            })
            .collect())
    }
}

impl Fit<f64> for Config {
    type Model = Model;

    fn fit(&self, data: &Samples<f64>) -> Result<Model> {
        self.validate()?;
        let (x, y) = (data.x(), data.y());
        let base = mean(y);
        let mut current = vec![base; data.n()];
        let mut residual = vec![0.0; data.n()];
        let mut trees = Vec::with_capacity(self.rounds);

        for _ in 0..self.rounds {
            for (r, (y, f)) in residual.iter_mut().zip(y.iter().zip(&current)) {
                *r = y - f;
            }
            let t = tree::grow(x, &residual, &self.tree)?;
            for (f, i) in current.iter_mut().zip(0..data.n()) {
                *f += self.rate * t.value(x.row(i));
            }
            trees.push(t);
        }

        Ok(Model {
            config: *self,
            base,
            trees,
            features: data.p(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Model as _;

    fn samples(n: usize) -> Samples<f64> {
        let rows: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                vec![t, (t * 7.0).sin()]
            })
            .collect();
        let y: Vec<f64> = rows.iter().map(|r| r[0] * 2.0 + r[1]).collect();
        Samples::new(Matrix::rows(&rows).unwrap(), y).unwrap()
    }

    #[test]
    fn one_round_at_full_rate_is_exactly_the_base_plus_one_tree() {
        let data = samples(40);
        let c = Config {
            rounds: 1,
            rate: 1.0,
            tree: tree::Config::default(),
        };
        let m = c.fit(&data).unwrap();
        assert_eq!(m.trees().len(), 1);
        let got = m.predict(data.x()).unwrap();
        for (i, g) in got.iter().enumerate() {
            let want = m.base() + m.trees()[0].value(data.x().row(i));
            assert!((g - want).abs() < 1e-15, "row {i}: {g} vs {want}");
        }
    }

    #[test]
    fn the_base_is_the_mean_of_the_targets() {
        let data = samples(30);
        let m = Config::new().fit(&data).unwrap();
        let want: f64 = data.y().iter().sum::<f64>() / data.n() as f64;
        assert_eq!(m.base(), want);
    }

    #[test]
    fn more_rounds_fit_the_training_data_more_closely() {
        // The one property that says boosting is boosting rather than a bag of trees.
        let data = samples(60);
        let error = |rounds| {
            let m = Config {
                rounds,
                ..Config::new()
            }
            .fit(&data)
            .unwrap();
            let p = m.predict(data.x()).unwrap();
            p.iter()
                .zip(data.y())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
        };
        let (few, many) = (error(5), error(80));
        assert!(many < few, "80 rounds ({many}) should beat 5 ({few})");
    }

    #[test]
    fn a_fit_is_a_function_of_its_data_so_the_name_is_stable() {
        let data = samples(50);
        let a = Config::new().fit(&data).unwrap();
        let b = Config::new().fit(&data).unwrap();
        assert_eq!(a.address(), b.address());
    }

    #[test]
    fn the_rate_and_the_tree_shape_are_both_part_of_the_name() {
        let data = samples(50);
        let base = Config::new().fit(&data).unwrap().address();
        let rate = Config {
            rate: 0.2,
            ..Config::new()
        }
        .fit(&data)
        .unwrap()
        .address();
        let deep = Config {
            tree: tree::Config {
                depth: 4,
                ..tree::Config::default()
            },
            ..Config::new()
        }
        .fit(&data)
        .unwrap()
        .address();
        assert_ne!(base, rate);
        assert_ne!(base, deep);
        assert_ne!(rate, deep);
    }

    #[test]
    fn zero_rounds_and_a_nonpositive_rate_are_both_refused() {
        let data = samples(20);
        assert!(matches!(
            Config {
                rounds: 0,
                ..Config::new()
            }
            .fit(&data),
            Err(Error::Config(_))
        ));
        assert!(matches!(
            Config {
                rate: 0.0,
                ..Config::new()
            }
            .fit(&data),
            Err(Error::Config(_))
        ));
    }

    #[test]
    fn predicting_with_the_wrong_feature_count_is_refused() {
        let m = Config::new().fit(&samples(20)).unwrap();
        assert!(m
            .predict(&Matrix::new(1, 3, vec![0.0; 3]).unwrap())
            .is_err());
    }
}
