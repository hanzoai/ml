//! A run's rounds replayed in one process, from its log.

use super::coordinator::{outer, Part};
use super::net::{self, Params};
use super::{Model, Plan, Round};
use crate::adam::{Adam, Hyper};
use anyhow::{Context, Result};
use std::collections::HashMap;

/// θ_global after the rounds of `log` (a [`super::Report`]'s), replayed in one process from
/// `theta`: each round's parts, in the log's order, trained by that worker's own model and
/// [`Adam`] from the round's θ_global on the batches it took; the deltas exchanged as the
/// cluster exchanges them (bf16 with error feedback, token-weighted, a part without tokens
/// sending nothing) and stepped by the same outer step (`lr`, `momentum`). `make` builds a
/// worker's model the first time its name appears. A run of rounds that each hold every member
/// from the plan's start ends at the run's θ_global bit for bit.
pub fn replay<M: Model>(
    log: &[Vec<Part>],
    plan: &Plan,
    adam: Hyper,
    lr: f64,
    momentum: f64,
    mut theta: Vec<f32>,
    mut make: impl FnMut(&str) -> Result<M>,
) -> Result<Vec<f32>> {
    let n = theta.len();
    let (mut m, mut carry, mut sum) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    let mut members: HashMap<String, (M, Params, Adam, Vec<f32>)> = HashMap::new();
    for (round, parts) in log.iter().enumerate() {
        let mut weight = 0f64;
        for (name, batches, tokens) in parts {
            if *tokens == 0 {
                continue;
            }
            if !members.contains_key(name) {
                let model = make(name)?;
                let params = Params::of(model.vars(), |p| model.rate(p).is_some());
                anyhow::ensure!(
                    params.len == n,
                    "{name}: θ of {} values, the log's {n}",
                    params.len
                );
                let a = Adam::new(model.vars(), |p| model.rate(p), adam)?;
                members.insert(name.clone(), (model, params, a, vec![0f32; n]));
            }
            let (model, params, a, own) = members.get_mut(name).context("member")?;
            params.write(model.vars(), &theta)?;
            let at = Round {
                round: round as u64,
                params,
                theta: &theta,
            };
            for &b in batches {
                let batch = plan.batch(b);
                a.schedule(batch.rate);
                model.step(a, &batch, &at)?;
            }
            model.end();
            let q = net::delta(params.read(model.vars())?, &theta, own);
            net::accumulate(&mut sum, &q, *tokens as f32);
            weight += *tokens as f64;
        }
        if weight > 0.0 {
            outer(
                &mut theta, &mut sum, weight, &mut m, &mut carry, lr, momentum,
            );
        }
    }
    Ok(theta)
}
