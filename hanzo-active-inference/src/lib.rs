//! Finite-state active inference with exact Bayesian filtering and open-loop policies.
//!
//! A[a][o][s] = P(o|s,a); B[a][next][current] = P(next|current,a).
//! G = KL(Q(o)||C(o)) + E_Q(s) H(A[:,s]) = -E log C(o) - I(s;o).
//! Q(pi) is proportional to E(pi) exp(-precision G(pi)). No execution is provided.
use serde::{Deserialize, Serialize};

/// A refused invalid model or impossible observation.
#[derive(Debug, Clone, PartialEq)]
pub struct Error(pub String);
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for Error {}
/// A checked mathematical operation.
pub type Result<T> = std::result::Result<T, Error>;
fn refuse(message: &str) -> Error {
    Error(message.into())
}
fn distribution(p: &[f64]) -> Result<()> {
    if p.is_empty()
        || p.iter().any(|x| !x.is_finite() || *x < 0.)
        || (p.iter().sum::<f64>() - 1.).abs() > 1e-9
    {
        return Err(refuse(
            "expected a finite normalized probability distribution",
        ));
    }
    Ok(())
}
fn entropy(p: &[f64]) -> f64 {
    -p.iter()
        .filter(|x| **x > 0.)
        .map(|x| x * x.ln())
        .sum::<f64>()
}

/// Validated categorical generative model; fields cannot change behind validation.
#[derive(Debug, Clone)]
pub struct Model {
    a: Vec<Vec<Vec<f64>>>,
    b: Vec<Vec<Vec<f64>>>,
    c: Vec<f64>,
    states: usize,
}
/// Risk/ambiguity and pragmatic/epistemic decompositions, all in nats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Terms {
    /// Divergence between predicted and preferred outcomes.
    pub risk: f64,
    /// Expected observation likelihood entropy.
    pub ambiguity: f64,
    /// Expected negative log preference.
    pub extrinsic: f64,
    /// Mutual information between future states and outcomes.
    pub information_gain: f64,
    /// Expected free energy: risk + ambiguity = extrinsic - information gain.
    pub expected_free_energy: f64,
}
/// Exact filtered posterior and its variational free energy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Posterior {
    /// Q(s|o,a).
    pub belief: Vec<f64>,
    /// F at its exact minimum, -log P(o|a).
    pub free_energy: f64,
}
/// Posterior over explicitly enumerated, allowed policies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policies {
    /// Q(pi), with disallowed policies exactly zero.
    pub probabilities: Vec<f64>,
    /// Accumulated expected free energy; disallowed policies still have mathematical terms.
    pub terms: Vec<Terms>,
    /// First maximum on ties.
    pub selected: usize,
}
impl Model {
    /// Validate all stochastic columns, strict-positive preferences and bounded dimensions.
    pub fn new(a: Vec<Vec<Vec<f64>>>, b: Vec<Vec<Vec<f64>>>, c: Vec<f64>) -> Result<Self> {
        if b.is_empty() || b.len() > 64 || a.len() != b.len() || c.is_empty() || c.len() > 256 {
            return Err(refuse("invalid action/outcome dimensions"));
        }
        let states = b[0].len();
        if states == 0 || states > 256 {
            return Err(refuse("state budget is 1..256"));
        }
        distribution(&c)?;
        if c.iter().any(|x| *x <= 0.) {
            return Err(refuse(
                "preferences must be strictly positive; hard safety uses policy masks",
            ));
        }
        for action in 0..b.len() {
            if b[action].len() != states
                || b[action].iter().any(|r| r.len() != states)
                || a[action].len() != c.len()
                || a[action].iter().any(|r| r.len() != states)
            {
                return Err(refuse("inconsistent transition or likelihood dimensions"));
            }
            for state in 0..states {
                distribution(&b[action].iter().map(|r| r[state]).collect::<Vec<_>>())?;
                distribution(&a[action].iter().map(|r| r[state]).collect::<Vec<_>>())?;
            }
        }
        Ok(Self { a, b, c, states })
    }
    fn check(&self, q: &[f64], action: usize) -> Result<()> {
        distribution(q)?;
        if q.len() != self.states || action >= self.b.len() {
            return Err(refuse("belief shape or action outside model"));
        }
        Ok(())
    }
    /// Predict one local factor. Work is O(states squared), independent of global graph size.
    pub fn predict(&self, q: &[f64], action: usize) -> Result<Vec<f64>> {
        self.check(q, action)?;
        Ok(self.b[action]
            .iter()
            .map(|row| row.iter().zip(q).map(|(b, q)| b * q).sum())
            .collect())
    }
    /// F(Q) = sum Q(s) log[Q(s)/(prior(s) A(o|s,a))].
    /// Zero-probability posterior terms contribute zero; positive mass outside support is infinite.
    pub fn variational_free_energy(
        &self,
        q: &[f64],
        prior: &[f64],
        action: usize,
        outcome: usize,
    ) -> Result<f64> {
        self.check(q, action)?;
        self.check(prior, action)?;
        if outcome >= self.c.len() {
            return Err(refuse("outcome outside model"));
        }
        let mut total = 0.;
        for s in 0..self.states {
            if q[s] > 0. {
                let joint = prior[s] * self.a[action][outcome][s];
                if joint == 0. {
                    return Ok(f64::INFINITY);
                }
                total += q[s] * (q[s] / joint).ln();
            }
        }
        Ok(total)
    }
    /// Infer current state from a current prior and observation (no hidden extra transition).
    pub fn observe(&self, prior: &[f64], action: usize, outcome: usize) -> Result<Posterior> {
        self.check(prior, action)?;
        if outcome >= self.c.len() {
            return Err(refuse("outcome outside model"));
        }
        let mut belief: Vec<_> = prior
            .iter()
            .zip(&self.a[action][outcome])
            .map(|(q, a)| q * a)
            .collect();
        let evidence: f64 = belief.iter().sum();
        if evidence <= 0. {
            return Err(refuse("observation has zero evidence"));
        }
        for q in &mut belief {
            *q /= evidence;
        }
        Ok(Posterior {
            belief,
            free_energy: -evidence.ln(),
        })
    }
    /// Score predicted states using both algebraically equivalent EFE decompositions.
    pub fn terms(&self, predicted: &[f64], action: usize) -> Result<Terms> {
        self.check(predicted, action)?;
        let outcomes: Vec<f64> = self.a[action]
            .iter()
            .map(|r| r.iter().zip(predicted).map(|(a, q)| a * q).sum())
            .collect();
        let ambiguity = (0..self.states)
            .map(|s| {
                predicted[s] * entropy(&self.a[action].iter().map(|r| r[s]).collect::<Vec<_>>())
            })
            .sum::<f64>();
        let risk = outcomes
            .iter()
            .zip(&self.c)
            .filter(|(q, _)| **q > 0.)
            .map(|(q, c)| q * (q.ln() - c.ln()))
            .sum::<f64>();
        let extrinsic = -outcomes
            .iter()
            .zip(&self.c)
            .map(|(q, c)| q * c.ln())
            .sum::<f64>();
        let information_gain = (entropy(&outcomes) - ambiguity).max(0.);
        Ok(Terms {
            risk,
            ambiguity,
            extrinsic,
            information_gain,
            expected_free_energy: risk + ambiguity,
        })
    }
    /// Finite-horizon open-loop policy posterior with a habit prior and hard admissibility mask.
    /// This is not a contingent policy tree or parameter-information-gain planner.
    pub fn policies(
        &self,
        q: &[f64],
        policies: &[Vec<usize>],
        habits: &[f64],
        allowed: &[bool],
        precision: f64,
    ) -> Result<Policies> {
        if policies.is_empty()
            || policies.len() > 256
            || policies.len() != habits.len()
            || policies.len() != allowed.len()
            || !precision.is_finite()
            || precision < 0.
        {
            return Err(refuse("invalid policies or precision"));
        }
        distribution(habits)?;
        let mut terms = Vec::new();
        let mut logits = Vec::new();
        for (i, policy) in policies.iter().enumerate() {
            if policy.is_empty() || policy.len() > 16 {
                return Err(refuse("policy horizon budget is 1..16"));
            }
            let mut state = q.to_vec();
            let mut total = Terms {
                risk: 0.,
                ambiguity: 0.,
                extrinsic: 0.,
                information_gain: 0.,
                expected_free_energy: 0.,
            };
            for action in policy {
                state = self.predict(&state, *action)?;
                let t = self.terms(&state, *action)?;
                total.risk += t.risk;
                total.ambiguity += t.ambiguity;
                total.extrinsic += t.extrinsic;
                total.information_gain += t.information_gain;
                total.expected_free_energy += t.expected_free_energy;
            }
            logits.push(if allowed[i] && habits[i] > 0. {
                habits[i].ln() - precision * total.expected_free_energy
            } else {
                f64::NEG_INFINITY
            });
            terms.push(total);
        }
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !max.is_finite() {
            return Err(refuse("no admissible policy with nonzero prior"));
        }
        let mut probabilities: Vec<_> = logits.iter().map(|x| (x - max).exp()).collect();
        let norm: f64 = probabilities.iter().sum();
        for p in &mut probabilities {
            *p /= norm;
        }
        let selected = (0..probabilities.len()).fold(0, |best, i| {
            if probabilities[i] > probabilities[best] {
                i
            } else {
                best
            }
        });
        Ok(Policies {
            probabilities,
            terms,
            selected,
        })
    }
}

/// Positive Dirichlet pseudo-counts for learning the likelihood and transition tables.
/// Preferences are operator policy and are never learned from outcome frequency.
#[derive(Clone, Debug)]
pub struct Counts {
    /// Likelihood counts, action/outcome/state.
    pub a: Vec<Vec<Vec<f64>>>,
    /// Transition counts, action/next/current.
    pub b: Vec<Vec<Vec<f64>>>,
}
impl Counts {
    /// Normalize positive columns and validate the resulting model.
    pub fn model(&self, preferences: Vec<f64>) -> Result<Model> {
        fn normalize(t: &[Vec<Vec<f64>>]) -> Result<Vec<Vec<Vec<f64>>>> {
            let mut out = t.to_vec();
            for matrix in &mut out {
                if matrix.is_empty() || matrix[0].is_empty() {
                    return Err(refuse("empty count matrix"));
                }
                let cols = matrix[0].len();
                if matrix
                    .iter()
                    .any(|r| r.len() != cols || r.iter().any(|x| !x.is_finite() || *x <= 0.))
                {
                    return Err(refuse("counts must be finite, positive and rectangular"));
                }
                for col in 0..cols {
                    let sum: f64 = matrix.iter().map(|r| r[col]).sum();
                    if !sum.is_finite() {
                        return Err(refuse("count overflow"));
                    }
                    for row in matrix.iter_mut() {
                        row[col] /= sum;
                    }
                }
            }
            Ok(out)
        }
        Model::new(normalize(&self.a)?, normalize(&self.b)?, preferences)
    }
    /// A Bayesian expected-count update using the exact joint over adjacent states.
    /// Returns a new value; a bad observation cannot partially mutate the existing counts.
    pub fn learned(
        &self,
        previous: &[f64],
        action: usize,
        outcome: usize,
        rate: f64,
        preferences: Vec<f64>,
    ) -> Result<Self> {
        if !rate.is_finite() || rate <= 0. {
            return Err(refuse("learning rate must be finite and positive"));
        }
        let model = self.model(preferences.clone())?;
        let predicted = model.predict(previous, action)?;
        let posterior = model.observe(&predicted, action, outcome)?;
        let evidence = (-posterior.free_energy).exp();
        let mut next = self.clone();
        for s in 0..model.states {
            next.a[action][outcome][s] += rate * posterior.belief[s];
            for (old, p) in previous.iter().enumerate() {
                let joint = p * model.b[action][s][old] * model.a[action][outcome][s] / evidence;
                next.b[action][s][old] += rate * joint;
            }
        }
        next.model(preferences)?;
        Ok(next)
    }
}
