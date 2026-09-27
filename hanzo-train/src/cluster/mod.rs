//! One model trained across heterogeneous machines: local SGD with an outer Nesterov step
//! (DiLoCo).
//!
//! A coordinator owns the plan (every batch of every epoch, drawn up front), θ_global in F32,
//! the outer optimizer and the checkpoints. Workers join it at any time, each with its own copy
//! of the model on its own device (Metal, CUDA, ROCm or the CPU), and take batches one at a
//! time, so a faster device takes more. Each round a worker runs [`Adam`] from θ_global on the
//! batches it takes until the round's time is up, then sends θ_local − θ_global in bf16 with
//! error feedback and the tokens it trained. The coordinator sums the round's deltas in the
//! order the workers joined (never the order their reports arrive), averages them by tokens,
//! takes the outer step and broadcasts the change to θ_global in bf16. A batch counts once its
//! delta is merged: a worker that drops before reporting has its batches requeued. A round waits
//! for no one past its deadline plus a grace, and a worker whose link fails joins again on its
//! own. A checkpoint holds θ_global, the outer optimizer's state and every worker's inner
//! optimizer state, so a run resumed from it continues bit for bit as the uninterrupted run
//! would.
//!
//! The model is the caller's, behind [`Model`]: its parameters as a [`VarMap`] of F32 masters,
//! a learning rate per trained parameter, and one step over a batch the caller planned
//! ([`Plan`]: rows are indices into the caller's data). What a coordinator measures and keeps
//! at round ends and checkpoints is the caller's too, behind [`Keep`].
//!
//! [`replay`] reruns a run's rounds in one process from its log: with the same parts, the same
//! exchange and the same outer step, it lands on the run's θ_global bit for bit.

pub mod coordinator;
pub mod net;
pub mod replay;
pub mod worker;

pub use crate::adam::{Adam, Hyper};
pub use coordinator::{
    lead, start, Closed, Coordinator, Inner, Keep, Mark, Part, Report, Run, Start, Work,
};
pub use net::{Msg, Params, Setup};
pub use replay::replay;
pub use worker::{join, Summary, Worker};

use anyhow::Result;
use hanzo_ml::Device;
use hanzo_nn::VarMap;

/// A batch as a worker trains it: its place in the plan, the epoch it belongs to, its rows
/// (indices into the caller's data) and the learning-rate factor at its place in the schedule.
#[derive(Debug, Clone, PartialEq)]
pub struct Batch {
    pub id: usize,
    pub epoch: usize,
    pub rows: Vec<usize>,
    pub rate: f64,
}

/// θ_global as round `round` began, flattened in `params`' order.
pub struct Round<'a> {
    pub round: u64,
    pub params: &'a Params,
    pub theta: &'a [f32],
}

impl Round<'_> {
    /// θ as a new F32 map on `dev`: the trained parameters alone.
    pub fn map(&self, dev: &Device) -> Result<VarMap> {
        self.params.map(self.theta, dev)
    }
}

/// What a step reports: the tokens it trained, which weigh its worker's delta, and numbers the
/// model adds up over a round, in its own order ([`Model::line`], [`Keep::round`]).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Step {
    pub tokens: u64,
    pub sums: Vec<f64>,
}

/// A model the cluster trains.
pub trait Model {
    /// Every parameter as an F32 master: the trained ones, which have a [`Model::rate`], and
    /// the frozen ones, which stay out of θ_global and which every process loads alike.
    fn vars(&self) -> &VarMap;

    /// Parameter `name`'s base learning rate; `None` holds it fixed.
    fn rate(&self, name: &str) -> Option<f64>;

    /// One optimizer step on `batch`, in `round`: forward, backward, [`Adam::step`]. The
    /// optimizer's schedule is already at `batch.rate`.
    fn step(&mut self, adam: &mut Adam, batch: &Batch, round: &Round) -> Result<Step>;

    /// The round's steps are done: drop what only the round needed.
    fn end(&mut self) {}

    /// What the worker's line adds for a round of `steps` steps in `secs` seconds whose steps
    /// reported `sums`.
    fn line(&self, _sums: &[f64], _steps: u64, _secs: f64) -> String {
        String::new()
    }

    /// The device and precision the coordinator's log names this worker by.
    fn describe(&self) -> String {
        device(self.vars()).map_or("none".into(), |d| format!("{:?}", d.location()))
    }
}

/// The device of `vars`' first parameter by name.
pub fn device(vars: &VarMap) -> Option<Device> {
    let data = vars.data().lock().expect("varmap lock");
    let first = data.keys().min()?;
    Some(data[first].device().clone())
}

/// Every epoch's batches in training order, drawn up front so the schedule knows its length.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Plan {
    /// Each batch's rows: indices into the caller's data.
    pub batches: Vec<Vec<usize>>,
    /// Batches done at the end of each epoch.
    pub ends: Vec<usize>,
    /// Warmup batches.
    pub warm: usize,
    /// Batches in the final decay.
    pub cool: usize,
}

impl Plan {
    /// The epoch batch `b` belongs to.
    pub fn epoch(&self, b: usize) -> usize {
        self.ends.partition_point(|&e| e <= b)
    }

    pub fn total(&self) -> usize {
        self.batches.len()
    }

    /// The learning-rate factor of batch `b`.
    pub fn rate(&self, b: usize) -> f64 {
        rate(b, self.warm, self.cool, self.total())
    }

    /// Batch `b` as a worker trains it.
    pub fn batch(&self, b: usize) -> Batch {
        Batch {
            id: b,
            epoch: self.epoch(b),
            rows: self.batches[b].clone(),
            rate: self.rate(b),
        }
    }
}

/// The warmup and final-decay steps of a plan of `total` steps: `warmup` and `cool` of them,
/// at least one each.
pub fn span(total: usize, warmup: f64, cool: f64) -> (usize, usize) {
    let warm = ((total as f64 * warmup) as usize).max(1);
    let cool = ((total as f64 * cool) as usize).clamp(1, total.saturating_sub(warm).max(1));
    (warm, cool)
}

/// Learning-rate factor at `step` of `total`: linear warmup from zero over `warm` steps, the
/// full rate, then over the last `cool` steps a linear decay to a tenth (warmup-stable-decay).
pub fn rate(step: usize, warm: usize, cool: usize, total: usize) -> f64 {
    let stable = total.saturating_sub(cool).max(warm);
    if step < warm {
        (step + 1) as f64 / warm as f64
    } else if step < stable {
        1.0
    } else {
        1.0 - 0.9 * (step - stable) as f64 / (total - stable).max(1) as f64
    }
}

/// The first address `addr` (host:port) resolves to.
pub fn resolve(addr: &str) -> Result<std::net::SocketAddr> {
    std::net::ToSocketAddrs::to_socket_addrs(addr)?
        .next()
        .ok_or_else(|| anyhow::anyhow!("{addr}: no address"))
}

/// This machine's name: its host name up to the first dot.
pub fn host() -> String {
    std::process::Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|h| h.trim().split('.').next().map(str::to_string))
        .filter(|h| !h.is_empty())
        .unwrap_or_else(|| "worker".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_schedule_warms_holds_and_cools() {
        // cool = total − warm is a linear decay from the end of warmup
        for step in 0..100 {
            let linear = if step < 10 {
                (step + 1) as f64 / 10.0
            } else {
                1.0 - 0.9 * (step - 10) as f64 / 90.0
            };
            assert!((rate(step, 10, 90, 100) - linear).abs() < 1e-12);
        }
        assert_eq!(rate(50, 10, 20, 100), 1.0);
        assert_eq!(rate(79, 10, 20, 100), 1.0);
        assert!((rate(90, 10, 20, 100) - 0.55).abs() < 1e-12);
        assert_eq!(span(100, 0.1, 0.9), (10, 90));
        assert_eq!(span(100, 0.03, 1.0), (3, 97));
        assert_eq!(span(5, 0.0, 0.0), (1, 1));
    }

    #[test]
    fn a_plan_knows_each_batchs_epoch_and_rate() {
        let p = Plan {
            batches: vec![vec![0], vec![1, 2], vec![3], vec![4], vec![5]],
            ends: vec![2, 5],
            warm: 1,
            cool: 2,
        };
        assert_eq!(
            (0..5).map(|b| p.epoch(b)).collect::<Vec<_>>(),
            [0, 0, 1, 1, 1]
        );
        let b = p.batch(1);
        assert_eq!((b.id, b.epoch, b.rows, b.rate), (1, 0, vec![1, 2], 1.0));
        assert_eq!(p.rate(4), rate(4, 1, 2, 5));
    }
}
