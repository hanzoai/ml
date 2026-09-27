//! The coordinator of a training run: local SGD with an outer Nesterov step (DiLoCo).
//!
//! It owns the plan, θ_global in F32, the outer optimizer and checkpoints; it runs inside a
//! worker process ([`lead`]), and other machines join it. Workers take batches one at a time,
//! so a faster device takes more. Each round a worker trains from θ_global until the round's
//! time is up, then sends θ_local − θ_global in bf16 with the tokens it processed. Once every
//! worker in the round has reported, the token-weighted mean delta Δ, summed in the order the
//! workers joined (never the order their reports arrive), is the negative pseudo-gradient
//! g = −Δ, and
//!
//!   m ← μ·m + g,   θ_global ← θ_global − η·(g + μ·m).
//!
//! The change to θ_global goes out in bf16; θ_global moves by exactly what went out and the
//! rounding joins the next change, so every worker's copy of θ_global is bit-identical to this
//! one. A batch counts once its delta is merged: a worker that drops before reporting has its
//! batches requeued. Frozen parameters are outside θ_global: every process loads them itself.
//!
//! A round waits for no one past its deadline plus the run's `grace` (for a worker that joined
//! late, past its joining plus the grace): a member that has not reported by then is dropped,
//! its link shut and its batches requeued, and the round closes with the reports that came. A
//! round none of whose members reported stays open for the next to join. A worker that joins
//! under the name of a member still linked replaces it, as a machine does that lost its link and
//! came back. Optimizer states stay kept by name, so a dropped worker gets its own back when it
//! joins again.
//!
//! Each round's line (and the log a checkpoint keeps) carries what the caller's
//! [`Keep::round`] measures: the workers' own numbers, anything measured at the new θ_global.
//!
//! A checkpoint (every `every` merged batches, each epoch, the end) is taken at a round
//! boundary: [`Keep::validate`], then [`Keep::save`] (the caller's weights), then
//! `state.safetensors`, the outer momentum, its rounding carry, the merged batches and the
//! round; and `inner/{worker}.safetensors`, each worker's AdamW moments, step count and
//! rounding carry, which a worker of that name gets back when it joins. A run resumed from it
//! continues as the uninterrupted run would.

use super::net::{self, Msg, Params, Setup};
use super::{host, worker, Model, Plan, Round};
use crate::adam::Hyper;
use anyhow::{bail, ensure, Context, Result};
use hanzo_ml::{DType, Device, Tensor};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::io::BufReader;
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Elements per parallel chunk of the outer step.
const CHUNK: usize = 1 << 16;

/// A run as its coordinator serves it.
pub struct Run {
    /// The build: a worker of another is turned away.
    pub code: String,
    /// What else a worker needs to load the model ([`Setup::body`]).
    pub body: Value,
    /// θ's layout: the trained parameters.
    pub params: Params,
    pub plan: Plan,
    /// Batches the run takes: the plan's first `limit`; the schedule still spans the plan.
    pub limit: usize,
    /// The workers' inner optimizer.
    pub adam: Hyper,
    /// Outer learning rate: the step along the token-weighted mean of the workers' deltas.
    pub outer: f64,
    /// Outer Nesterov momentum.
    pub momentum: f64,
    /// Seconds of local training per round; a round ends at the first step boundary past it,
    /// so 0 is one step.
    pub round: f64,
    /// Seconds past its round's deadline (or past its joining, when later) a worker's report is
    /// waited for. Then the round closes with the reports that came, and the worker is dropped:
    /// its batches requeued, its optimizer state kept for when it joins again.
    pub grace: f64,
    /// A checkpoint every this many merged batches (and at the end of each epoch, and the end).
    pub every: usize,
    /// The checkpoint directory; `None` validates at checkpoints and saves nothing.
    pub out: Option<PathBuf>,
}

/// A worker's optimizer state: its step count, and its first moments, second moments and
/// rounding carry as one F32 little-endian buffer.
#[derive(Clone)]
pub struct Inner {
    pub t: u64,
    pub data: Arc<Vec<u8>>,
}

/// Where a run starts: θ_global, the outer optimizer's state, the batches already merged and
/// the workers' optimizer states.
pub struct Start {
    pub theta: Vec<f32>,
    pub momentum: Vec<f32>,
    pub carry: Vec<f32>,
    pub merged: Vec<bool>,
    pub round: u64,
    pub inner: HashMap<String, Inner>,
}

impl Start {
    /// A new run from `theta` over `limit` batches.
    pub fn new(theta: Vec<f32>, limit: usize) -> Start {
        let n = theta.len();
        Start {
            theta,
            momentum: vec![0f32; n],
            carry: vec![0f32; n],
            merged: vec![false; limit],
            round: 0,
            inner: HashMap::new(),
        }
    }

    /// The run checkpointed in `dir` over `limit` batches: θ_global `theta`, which the caller
    /// read from its weights, saved at the start of `round` ([`Mark::round`]), and the state
    /// beside them.
    pub fn resume(
        dir: &Path,
        params: &Params,
        theta: Vec<f32>,
        round: u64,
        limit: usize,
    ) -> Result<Start> {
        ensure!(theta.len() == params.len, "weights for another model");
        let state = hanzo_ml::safetensors::load(dir.join("state.safetensors"), &Device::Cpu)?;
        let get = |k: &str| state.get(k).with_context(|| format!("state lacks {k}"));
        let momentum = get("momentum")?.to_vec1::<f32>()?;
        let carry = get("carry")?.to_vec1::<f32>()?;
        let mut merged: Vec<bool> = get("merged")?
            .to_vec1::<u8>()?
            .into_iter()
            .map(|b| b != 0)
            .collect();
        let at = get("round")?.to_vec1::<i64>()?[0] as u64;
        ensure!(
            momentum.len() == params.len && carry.len() == params.len,
            "state for another model"
        );
        ensure!(at == round, "weights and state from different rounds");
        ensure!(
            merged.len() <= limit,
            "state covers {} batches, the run {limit}",
            merged.len()
        );
        merged.resize(limit, false);
        let mut inner = HashMap::new();
        if let Ok(files) = std::fs::read_dir(dir.join("inner")) {
            for f in files.flatten() {
                let path = f.path();
                let Some(name) = path.file_stem().and_then(|s| s.to_str()) else {
                    continue;
                };
                let ts = hanzo_ml::safetensors::load(&path, &Device::Cpu)?;
                let flat = ts.get("state").context("inner state")?.to_vec1::<f32>()?;
                ensure!(
                    flat.len() == 3 * params.len,
                    "{}: another model",
                    path.display()
                );
                let t = ts.get("t").context("inner step")?.to_vec1::<i64>()?[0] as u64;
                inner.insert(
                    name.to_string(),
                    Inner {
                        t,
                        data: Arc::new(net::pack(&flat)),
                    },
                );
            }
        }
        eprintln!(
            "resume: round {round}, {} of {limit} batches merged, optimizer states of {:?}",
            merged.iter().filter(|m| **m).count(),
            inner.keys().collect::<Vec<_>>()
        );
        Ok(Start {
            theta,
            momentum,
            carry,
            merged,
            round,
            inner,
        })
    }
}

/// One worker's report for a round: its name, the tokens and steps it trained, the seconds
/// from the round's start to its last step's end, and what its steps' sums add up to.
#[derive(Debug, Clone, PartialEq)]
pub struct Work {
    pub name: String,
    pub tokens: u64,
    pub steps: u64,
    pub secs: f64,
    pub sums: Vec<f64>,
}

/// A round as it closed: its number, the batches merged after it of the run's `limit`, each
/// member's work in join order, the round's seconds, how late it closed past its deadline and
/// the outer step's seconds.
pub struct Closed<'a> {
    pub round: u64,
    pub merged: usize,
    pub limit: usize,
    pub works: &'a [Work],
    pub secs: f64,
    pub late: f64,
    pub outer: f64,
}

/// A checkpoint: the round θ_global starts, the batches merged of the plan's `total` and the
/// run's `limit`, the epochs done, the hours since the coordinator started, and each round's
/// log entry so far.
pub struct Mark<'a> {
    pub round: u64,
    pub merged: usize,
    pub total: usize,
    pub limit: usize,
    pub epochs: usize,
    pub hours: f64,
    pub log: &'a [Value],
}

/// What a coordinator's owner measures and keeps.
pub trait Keep: Send + 'static {
    /// Round `closed.round` closed and θ_global moved to `next`: what the round's line adds, and
    /// the fields its log entry adds (an object). By default each worker's steps and tokens per
    /// second.
    fn round(&mut self, closed: &Closed, next: &Round) -> Result<(String, Value)> {
        let _ = next;
        let mut line = String::new();
        let mut workers = Vec::new();
        for w in closed.works {
            let tps = w.tokens as f64 / w.secs.max(1e-9);
            line += &format!(" | {} {} steps {tps:.0} tok/s", w.name, w.steps);
            workers.push(
                json!({"worker": w.name, "steps": w.steps, "tokens": w.tokens,
                "tokens_per_s": tps, "sums": w.sums}),
            );
        }
        Ok((line, json!({ "workers": workers })))
    }

    /// Validation at a checkpoint's θ_global `at`. None by default.
    fn validate(&mut self, mark: &Mark, at: &Round) -> Result<Value> {
        let _ = (mark, at);
        Ok(Value::Null)
    }

    /// Save the checkpoint's weights, and whatever else it keeps, into `dir`; the runtime's
    /// state goes beside them. A resume reads θ_global back from them and needs `mark.round`.
    /// By default `model.safetensors`, θ_global's parameters as F32.
    fn save(&mut self, dir: &Path, mark: &Mark, at: &Round, val: &Value) -> Result<()> {
        let _ = (mark, val);
        std::fs::create_dir_all(dir)?;
        at.map(&Device::Cpu)?.save(dir.join("model.safetensors"))?;
        Ok(())
    }
}

struct Member {
    name: String,
    /// Its link, shut to drop it.
    link: TcpStream,
    joined: Instant,
    /// Batches taken this round.
    held: Vec<usize>,
    work: Option<Work>,
    /// This round's θ_local − θ_global in bf16, once reported with tokens.
    delta: Option<Vec<u8>>,
    /// Dropped after reporting: its delta and batches merge with the round.
    gone: bool,
}

struct State {
    /// Requeued batches, handed out before `next`.
    queue: VecDeque<usize>,
    /// The first batch of the plan never handed out.
    next: usize,
    merged: Vec<bool>,
    count: usize,
    /// The first unmerged batch: every epoch ending at or before it is done.
    frontier: usize,
    round: u64,
    opened: Instant,
    deadline: Instant,
    /// Every member has reported; the outer step is running.
    closing: bool,
    done: bool,
    members: BTreeMap<u64, Member>,
    ids: u64,
    /// The change to θ_global that opened `round`, bf16 (empty: none).
    update: Arc<Vec<u8>>,
    /// `round` starts a checkpoint: its members send their optimizer state.
    keep: bool,
    /// Members whose optimizer state the checkpoint still waits for.
    owed: BTreeSet<u64>,
    requeued: usize,
}

struct Shared {
    setup: Setup,
    plan: Plan,
    limit: usize,
    params: Params,
    outer: f64,
    momentum: f64,
    round: f64,
    grace: f64,
    every: usize,
    out: Option<PathBuf>,
    state: Mutex<State>,
    turn: Condvar,
    theta: RwLock<Vec<f32>>,
    /// The latest optimizer state of each worker, by name.
    inner: Mutex<HashMap<String, Inner>>,
}

/// One worker's part of a round: its name, the batches it trained, their tokens.
pub type Part = (String, Vec<usize>, u64);

/// What a finished run leaves.
pub struct Report {
    pub theta: Vec<f32>,
    pub rounds: u64,
    pub merged: usize,
    pub requeued: usize,
    /// Each round's parts, in round order.
    pub log: Vec<Vec<Part>>,
}

pub struct Coordinator {
    addr: SocketAddr,
    handle: JoinHandle<Result<Report>>,
}

impl State {
    /// Member `id` is out of the run, its link shut so its session ends. When it has reported,
    /// its delta and batches still merge with its round; else its batches go back to the queue.
    fn out(&mut self, id: u64) {
        self.owed.remove(&id);
        let Some(m) = self.members.get_mut(&id) else {
            return;
        };
        let _ = m.link.shutdown(Shutdown::Both);
        if m.work.is_some() {
            m.gone = true;
            return;
        }
        let m = self.members.remove(&id).expect("member");
        if !m.held.is_empty() {
            eprintln!(
                "cluster: {} left; requeued {} batches",
                m.name,
                m.held.len()
            );
        }
        self.requeued += m.held.len();
        for b in m.held.into_iter().rev() {
            self.queue.push_front(b);
        }
    }

    /// Drop every member that has not reported by the round's deadline (or its joining, when
    /// later) plus `grace`; the next time another would be due, if any.
    fn late(&mut self, grace: Duration) -> Option<Instant> {
        let now = Instant::now();
        let deadline = self.deadline;
        let due = |m: &Member| m.joined.max(deadline) + grace;
        let late: Vec<u64> = self
            .members
            .iter()
            .filter(|(_, m)| m.work.is_none() && due(m) <= now)
            .map(|(&id, _)| id)
            .collect();
        for id in late {
            eprintln!(
                "cluster: {} did not report round {} within {:.0}s of its deadline; dropped",
                self.members[&id].name,
                self.round,
                grace.as_secs_f64()
            );
            self.out(id);
        }
        self.members
            .values()
            .filter(|m| m.work.is_none())
            .map(due)
            .min()
    }
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().expect("state lock")
    }

    fn round(&self) -> Duration {
        Duration::from_secs_f64(self.round.max(0.0))
    }

    fn grace(&self) -> Duration {
        Duration::from_secs_f64(self.grace.max(0.0))
    }

    /// Wait on the state until notified or, given one, until `due`.
    fn wait<'a>(&self, s: MutexGuard<'a, State>, due: Option<Instant>) -> MutexGuard<'a, State> {
        match due {
            Some(t) => {
                let left = t.saturating_duration_since(Instant::now());
                self.turn.wait_timeout(s, left).expect("state lock").0
            }
            None => self.turn.wait(s).expect("state lock"),
        }
    }

    /// The run ended in an error: members are told it is done, as are workers that join.
    fn end(&self) {
        let mut s = self.lock();
        s.done = true;
        s.closing = false;
        s.keep = false;
        s.owed.clear();
        for m in s.members.values() {
            let _ = m.link.shutdown(Shutdown::Both);
        }
        drop(s);
        self.turn.notify_all();
    }

    /// The next batch for member `id`: a requeued one first, then the plan's next.
    fn take(&self, id: u64) -> Option<usize> {
        let mut s = self.lock();
        if s.done || !s.members.contains_key(&id) {
            return None;
        }
        let b = match s.queue.pop_front() {
            Some(b) => b,
            None if s.next < self.limit => {
                s.next += 1;
                s.next - 1
            }
            None => return None,
        };
        s.members.get_mut(&id)?.held.push(b);
        Some(b)
    }

    /// Member `id`'s session ended: unreported batches go back to the queue, and a checkpoint
    /// stops waiting for its state.
    fn leave(&self, id: u64) {
        self.lock().out(id);
        self.turn.notify_all();
    }

    /// Take member `id`'s optimizer state off the wire, for the checkpoint under way.
    fn kept(&self, id: u64, name: &str, r: &mut impl std::io::Read) -> Result<()> {
        let t = match net::recv(r)? {
            Msg::State { t } => t,
            m => bail!("expected state, got {m:?}"),
        };
        let data = net::get(r, 12 * self.params.len)?.context("optimizer state")?;
        self.inner.lock().expect("inner lock").insert(
            name.to_string(),
            Inner {
                t,
                data: Arc::new(data),
            },
        );
        self.lock().owed.remove(&id);
        self.turn.notify_all();
        Ok(())
    }
}

/// Serve `run` on `listener` from `begin` until its plan is merged.
pub fn start(
    listener: TcpListener,
    run: Run,
    begin: Start,
    keep: impl Keep,
) -> Result<Coordinator> {
    let Run {
        code,
        body,
        params,
        plan,
        limit,
        adam,
        outer,
        momentum,
        round,
        grace,
        every,
        out,
    } = run;
    let n = params.len;
    ensure!(
        limit <= plan.total(),
        "a limit of {limit} over a plan of {}",
        plan.total()
    );
    ensure!(
        begin.theta.len() == n,
        "θ has {} values, the model {n}",
        begin.theta.len()
    );
    ensure!(
        begin.merged.len() == limit,
        "merged covers {} of {limit} batches",
        begin.merged.len()
    );
    let merged = begin.merged;
    let next = merged.iter().rposition(|&m| m).map_or(0, |i| i + 1);
    let count = merged.iter().filter(|m| **m).count();
    let now = Instant::now();
    let state = State {
        queue: (0..next).filter(|&i| !merged[i]).collect(),
        next,
        count,
        frontier: merged.iter().position(|m| !m).unwrap_or(limit),
        merged,
        round: begin.round,
        opened: now,
        deadline: now,
        closing: false,
        done: count >= limit,
        members: BTreeMap::new(),
        ids: 0,
        update: Arc::new(Vec::new()),
        keep: false,
        owed: BTreeSet::new(),
        requeued: 0,
    };
    let setup = Setup {
        code,
        params: params.digest(),
        total: plan.total(),
        warm: plan.warm,
        cool: plan.cool,
        adam,
        body,
    };
    let shared = Arc::new(Shared {
        setup,
        plan,
        limit,
        params,
        outer,
        momentum,
        round,
        grace,
        every: every.max(1),
        out,
        state: Mutex::new(state),
        turn: Condvar::new(),
        theta: RwLock::new(begin.theta),
        inner: Mutex::new(begin.inner),
    });
    let addr = listener.local_addr()?;
    let sh = shared.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            let sh = sh.clone();
            std::thread::spawn(move || {
                let peer = stream
                    .peer_addr()
                    .map_or_else(|_| "?".into(), |a| a.to_string());
                let mut me = None;
                if let Err(e) = session(&sh, stream, &peer, &mut me) {
                    eprintln!("cluster: {peer}: {e:#}");
                }
                if let Some(id) = me {
                    sh.leave(id);
                }
            });
        }
    });
    let (momentum, carry) = (begin.momentum, begin.carry);
    let handle = std::thread::spawn(move || {
        let report = rounds(&shared, keep, momentum, carry);
        if report.is_err() {
            shared.end();
        }
        report
    });
    Ok(Coordinator { addr, handle })
}

impl Coordinator {
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Wait for the plan to finish.
    pub fn join(self) -> Result<Report> {
        self.handle
            .join()
            .map_err(|_| anyhow::anyhow!("coordinator panicked"))?
    }
}

fn session(sh: &Shared, stream: TcpStream, peer: &str, me: &mut Option<u64>) -> Result<()> {
    net::link(&stream)?;
    let mut r = BufReader::with_capacity(1 << 20, stream.try_clone()?);
    let link = stream.try_clone()?;
    let mut w = stream;
    let code = match net::recv(&mut r)? {
        Msg::Hello { code } => code,
        m => bail!("expected hello, got {m:?}"),
    };
    if code != sh.setup.code {
        let reason = format!("build {code}, the coordinator runs {}", sh.setup.code);
        net::send(
            &mut w,
            &Msg::Refuse {
                reason: reason.clone(),
            },
        )?;
        bail!(reason);
    }
    net::send(&mut w, &Msg::Setup(sh.setup.clone()))?;
    let (name, device) = match net::recv(&mut r)? {
        Msg::Ready { name, device } => (name, device),
        Msg::Refuse { reason } => bail!("refused: {reason}"),
        m => bail!("expected ready, got {m:?}"),
    };
    let named = !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c));
    let joined = {
        let mut s = sh.lock();
        while s.closing {
            s = sh.turn.wait(s).expect("state lock");
        }
        if !named {
            None
        } else if s.done {
            Some(None)
        } else {
            // the same machine again: its earlier link is dead or dying
            let before: Vec<u64> = s
                .members
                .iter()
                .filter(|(_, m)| m.name == name && !m.gone)
                .map(|(&id, _)| id)
                .collect();
            for id in before {
                eprintln!("cluster: {name} joins again; its earlier link is dropped");
                s.out(id);
            }
            let now = Instant::now();
            if s.members.is_empty() {
                s.opened = now;
                s.deadline = now + sh.round();
            }
            let id = s.ids;
            s.ids += 1;
            s.members.insert(
                id,
                Member {
                    name: name.clone(),
                    link,
                    joined: now,
                    held: Vec::new(),
                    work: None,
                    delta: None,
                    gone: false,
                },
            );
            // the round now waits for it, until it is due
            sh.turn.notify_all();
            Some(Some((
                id,
                s.round,
                s.deadline.saturating_duration_since(now),
            )))
        }
    };
    let (id, mut round, left) = match joined {
        None => {
            let reason = format!("a worker needs a name of [A-Za-z0-9._-], not {name:?}");
            net::send(
                &mut w,
                &Msg::Refuse {
                    reason: reason.clone(),
                },
            )?;
            bail!(reason);
        }
        Some(None) => return net::send(&mut w, &Msg::Done { keep: false }),
        Some(Some(j)) => j,
    };
    *me = Some(id);
    let inner = sh.inner.lock().expect("inner lock").get(&name).cloned();
    eprintln!(
        "cluster: {name} ({peer}) joins round {round} as worker {id} ({device}){}",
        inner.as_ref().map_or(String::new(), |i| format!(
            ", optimizer state at step {}",
            i.t
        ))
    );
    net::send(
        &mut w,
        &Msg::Welcome {
            id,
            round,
            left: left.as_secs_f64(),
            inner: inner.as_ref().map(|i| i.t),
        },
    )?;
    // a member that has not reported holds its round open until it is dropped, its link shut
    // first: θ_global cannot move under a joiner that could still use it
    let theta = net::pack(&sh.theta.read().expect("theta lock"));
    net::put(&mut w, &theta)?;
    drop(theta);
    if let Some(i) = inner {
        net::put(&mut w, &i.data)?;
    }
    loop {
        match net::recv(&mut r)? {
            Msg::Take => match sh.take(id) {
                Some(b) => net::send(
                    &mut w,
                    &Msg::Batch {
                        id: b,
                        epoch: sh.plan.epoch(b),
                        rows: sh.plan.batches[b].clone(),
                    },
                )?,
                None => net::send(&mut w, &Msg::Empty)?,
            },
            Msg::Delta {
                round: at,
                tokens,
                steps,
                secs,
                sums,
            } => {
                ensure!(at == round, "delta for round {at} in round {round}");
                let delta = net::get(&mut r, 2 * sh.params.len)?.filter(|_| tokens > 0);
                let (done, next, left, update, keep) = {
                    let mut s = sh.lock();
                    if let Some(m) = s.members.get_mut(&id) {
                        m.work = Some(Work {
                            name: name.clone(),
                            tokens,
                            steps,
                            secs,
                            sums,
                        });
                        m.delta = delta;
                    }
                    sh.turn.notify_all();
                    while s.round == round && !s.done {
                        s = sh.turn.wait(s).expect("state lock");
                    }
                    let left = s.deadline.saturating_duration_since(Instant::now());
                    (s.done, s.round, left, s.update.clone(), s.keep)
                };
                if done {
                    net::send(&mut w, &Msg::Done { keep })?;
                    if keep {
                        sh.kept(id, &name, &mut r)?;
                    }
                    return Ok(());
                }
                round = next;
                net::send(
                    &mut w,
                    &Msg::Update {
                        round,
                        left: left.as_secs_f64(),
                        keep,
                    },
                )?;
                net::put(&mut w, &update)?;
                if keep {
                    sh.kept(id, &name, &mut r)?;
                }
            }
            m => bail!("unexpected {m:?}"),
        }
    }
}

/// The outer step over the round's sum Σ tokens·delta of total `weight`, which it leaves at
/// zero: the step joins the carry, the carry goes out quantized and keeps what rounding
/// dropped, and θ moves by exactly what went out, which is returned.
pub(crate) fn outer(
    theta: &mut [f32],
    sum: &mut [f32],
    weight: f64,
    m: &mut [f32],
    carry: &mut [f32],
    lr: f64,
    mu: f64,
) -> Vec<u8> {
    let (w, lr, mu) = (weight as f32, lr as f32, mu as f32);
    sum.par_chunks_mut(CHUNK)
        .zip(m.par_chunks_mut(CHUNK))
        .zip(carry.par_chunks_mut(CHUNK))
        .for_each(|((s, m), c)| {
            for i in 0..s.len() {
                let g = -s[i] / w;
                m[i] = mu * m[i] + g;
                c[i] += -lr * (g + mu * m[i]);
                s[i] = 0.0;
            }
        });
    let q = net::quantize(carry);
    net::accumulate(theta, &q, 1.0);
    q
}

/// The outer optimizer's state and every worker's beside a checkpoint's weights.
fn store(
    dir: &Path,
    m: &[f32],
    carry: &[f32],
    merged: &[bool],
    round: u64,
    inner: &HashMap<String, Inner>,
) -> Result<()> {
    let cpu = Device::Cpu;
    let flags: Vec<u8> = merged.iter().map(|&b| b as u8).collect();
    let ts = HashMap::from([
        (
            "momentum".to_string(),
            Tensor::from_slice(m, m.len(), &cpu)?,
        ),
        (
            "carry".to_string(),
            Tensor::from_slice(carry, carry.len(), &cpu)?,
        ),
        (
            "merged".to_string(),
            Tensor::from_vec(flags, merged.len(), &cpu)?,
        ),
        ("round".to_string(), Tensor::new(&[round as i64], &cpu)?),
    ]);
    hanzo_ml::safetensors::save(&ts, dir.join("state.safetensors"))?;
    std::fs::create_dir_all(dir.join("inner"))?;
    for (name, i) in inner {
        let state = Tensor::from_raw_buffer(&i.data, DType::F32, &[i.data.len() / 4], &cpu)?;
        let ts = HashMap::from([
            ("state".to_string(), state),
            ("t".to_string(), Tensor::new(&[i.t as i64], &cpu)?),
        ]);
        hanzo_ml::safetensors::save(&ts, dir.join("inner").join(format!("{name}.safetensors")))?;
    }
    Ok(())
}

/// `base`'s fields with `more`'s: both objects.
fn merge(mut base: Value, more: Value) -> Value {
    if let (Some(b), Value::Object(m)) = (base.as_object_mut(), more) {
        b.extend(m);
    }
    base
}

/// Close rounds as their members report or are dropped, until the plan is merged.
fn rounds(
    sh: &Shared,
    mut keep: impl Keep,
    mut m: Vec<f32>,
    mut carry: Vec<f32>,
) -> Result<Report> {
    let every = sh.every;
    let grace = sh.grace();
    let started = Instant::now();
    let (first, mut epochs) = {
        let s = sh.lock();
        let ended = sh.plan.ends.iter().filter(|&&e| e <= s.frontier).count();
        (s.count, ended)
    };
    let mut evaluated = first / every;
    let mut log = Vec::new();
    let mut parts = Vec::new();
    let mut sum = vec![0f32; sh.params.len];
    loop {
        let (works, deltas, batches, opened, deadline) = {
            let mut s = sh.lock();
            while !s.done && (s.members.is_empty() || s.members.values().any(|m| m.work.is_none()))
            {
                // none due: every member reported, or none is left and the round waits for one
                let due = s.late(grace);
                if due.is_some() || s.members.is_empty() {
                    s = sh.wait(s, due);
                }
            }
            s.closing = true;
            let mut works = Vec::new();
            let mut deltas = Vec::new();
            let mut batches = Vec::new();
            let mut part = Vec::new();
            // members by id, the order they joined
            for m in s.members.values_mut() {
                if let Some(w) = &m.work {
                    part.push((m.name.clone(), m.held.clone(), w.tokens));
                    if let Some(d) = m.delta.take() {
                        deltas.push((w.tokens, d));
                    }
                    works.push(w.clone());
                }
                batches.append(&mut m.held);
            }
            parts.push(part);
            (works, deltas, batches, s.opened, s.deadline)
        };
        let closed = Instant::now();
        let weight: f64 = deltas.iter().map(|(t, _)| *t as f64).sum();
        let update = if weight > 0.0 {
            for (t, d) in deltas {
                net::accumulate(&mut sum, &d, t as f32);
            }
            let mut theta = sh.theta.write().expect("theta lock");
            outer(
                &mut theta,
                &mut sum,
                weight,
                &mut m,
                &mut carry,
                sh.outer,
                sh.momentum,
            )
        } else {
            Vec::new()
        };
        let stepped = Instant::now();
        let (round, count, done, merged, checkpoint) = {
            let mut s = sh.lock();
            for &b in &batches {
                if !s.merged[b] {
                    s.merged[b] = true;
                    s.count += 1;
                }
            }
            while s.frontier < sh.limit && s.merged[s.frontier] {
                s.frontier += 1;
            }
            let (count, frontier) = (s.count, s.frontier);
            let ended = sh.plan.ends.iter().filter(|&&e| e <= frontier).count();
            let done = count >= sh.limit;
            let checkpoint = ended > epochs || count / every > evaluated || done;
            epochs = ended;
            s.members.retain(|_, m| !m.gone);
            for m in s.members.values_mut() {
                m.work = None;
            }
            let now = Instant::now();
            s.round += 1;
            s.update = Arc::new(update);
            s.opened = now;
            s.deadline = now + sh.round();
            s.done = done;
            s.keep = checkpoint;
            s.owed = if checkpoint {
                s.members.keys().copied().collect()
            } else {
                BTreeSet::new()
            };
            s.closing = false;
            (s.round, count, done, s.merged.clone(), checkpoint)
        };
        sh.turn.notify_all();

        let f = sh.plan.rate(count);
        let per = started.elapsed().as_secs_f64() / (count - first).max(1) as f64;
        let (secs, late, step) = (
            (closed - opened).as_secs_f64(),
            closed.saturating_duration_since(deadline).as_secs_f64(),
            (stepped - closed).as_secs_f64(),
        );
        let theta = sh.theta.read().expect("theta lock");
        let at = Round {
            round,
            params: &sh.params,
            theta: &theta,
        };
        let (extra, fields) = keep.round(
            &Closed {
                round: round - 1,
                merged: count,
                limit: sh.limit,
                works: &works,
                secs,
                late,
                outer: step,
            },
            &at,
        )?;
        eprintln!(
            "round {}: {count}/{} lr×{f:.3} eta {:.0}m{extra} | {secs:.1}s, closed {late:.1}s \
             past the deadline, outer step {step:.1}s",
            round - 1,
            sh.limit,
            per * (sh.limit - count) as f64 / 60.0
        );
        log.push(merge(
            json!({"round": round - 1, "batches": count, "secs": secs, "late": late,
                "outer": step}),
            fields,
        ));
        if !checkpoint {
            continue;
        }

        evaluated = count / every;
        let mark = Mark {
            round,
            merged: count,
            total: sh.plan.total(),
            limit: sh.limit,
            epochs,
            hours: started.elapsed().as_secs_f64() / 3600.0,
            log: &log,
        };
        let val = keep.validate(&mark, &at)?;
        {
            // a member owes its state before its first batch of the round, so one that does not
            // report the round in time is dropped here as well
            let mut s = sh.lock();
            while !s.owed.is_empty() {
                let due = s.late(grace);
                if s.owed.is_empty() {
                    break;
                }
                s = sh.wait(s, due);
            }
        }
        if let Some(dir) = &sh.out {
            keep.save(dir, &mark, &at, &val)
                .context("save checkpoint")?;
            let inner = sh.inner.lock().expect("inner lock").clone();
            store(dir, &m, &carry, &merged, round, &inner).context("save state")?;
            eprintln!(
                "saved {} at {count} batches, {epochs} epochs",
                dir.display()
            );
        }
        if done {
            let s = sh.lock();
            return Ok(Report {
                theta: theta.clone(),
                rounds: s.round,
                merged: s.count,
                requeued: s.requeued,
                log: parts,
            });
        }
    }
}

/// Coordinate `run` from `begin` on `listen` and work in it: `model`, loaded as the run's
/// setup describes, becomes this machine's worker, named by its host.
pub fn lead<M: Model>(
    listen: &str,
    run: Run,
    begin: Start,
    keep: impl Keep,
    model: M,
) -> Result<Report> {
    let code = run.code.clone();
    let listener = TcpListener::bind(listen).with_context(|| format!("listen on {listen}"))?;
    let c = start(listener, run, begin, keep)?;
    let mut local = c.addr();
    if local.ip().is_unspecified() {
        local.set_ip([127, 0, 0, 1].into());
    }
    eprintln!("cluster: coordinating on {}", c.addr());
    if let Some(w) = worker::join(local, &host(), &code, |_| Ok(model))? {
        w.run()?;
    }
    c.join()
}
