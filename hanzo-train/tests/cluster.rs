//! The cluster runtime on the CPU with a tiny model (a frozen projection under a two-layer
//! classifier): one worker and three against their rounds replayed in one process, bit for bit,
//! at an outer step with momentum; micro-batches against the batch in one pass; two workers
//! against training alone; a dropped worker's batches requeued; a worker that cannot load
//! refused with its reason; a worker gone silent mid-round, or while it sends its delta,
//! dropped at the deadline; a worker whose link is cut joining again; frozen parameters; a run
//! killed (worker and coordinator) and resumed against the run uninterrupted. The transport's
//! round trip is tested in `net`.

use anyhow::{Context, Result};
use hanzo_ml::{DType, Device, Tensor, Var, D};
use hanzo_nn::VarMap;
use hanzo_train::adam::accumulate;
use hanzo_train::cluster::net;
use hanzo_train::cluster::{
    join as enroll, replay, span, start, Adam, Batch, Coordinator, Hyper, Keep, Model, Msg, Params,
    Plan, Report, Round, Run, Start, Step, Summary, Worker,
};
use std::collections::HashMap;
use std::io::BufReader;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier};
use std::thread::JoinHandle;
use std::time::Duration;

const CODE: &str = "tiny";
/// Inputs, hidden units, classes.
const X: usize = 8;
const H: usize = 16;
const K: usize = 4;

/// A fixed stream: splitmix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [−1, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Examples whose class is the argmax of a fixed linear teacher.
struct Data {
    x: Vec<f32>,
    y: Vec<u32>,
}

impl Data {
    fn new(n: usize) -> Arc<Data> {
        let mut rng = Rng(7);
        let teacher: Vec<f32> = (0..K * X).map(|_| rng.unit()).collect();
        let x: Vec<f32> = (0..n * X).map(|_| rng.unit()).collect();
        let y = (0..n)
            .map(|i| {
                let s =
                    |k: usize| -> f32 { (0..X).map(|j| teacher[k * X + j] * x[i * X + j]).sum() };
                (0..K).max_by(|&a, &b| s(a).total_cmp(&s(b))).unwrap() as u32
            })
            .collect();
        Arc::new(Data { x, y })
    }

    fn len(&self) -> usize {
        self.y.len()
    }

    fn gather(&self, rows: &[usize]) -> Result<(Tensor, Tensor)> {
        let x: Vec<f32> = rows
            .iter()
            .flat_map(|&i| self.x[i * X..(i + 1) * X].to_vec())
            .collect();
        let y: Vec<u32> = rows.iter().map(|&i| self.y[i]).collect();
        Ok((
            Tensor::from_vec(x, (rows.len(), X), &Device::Cpu)?,
            Tensor::from_vec(y, rows.len(), &Device::Cpu)?,
        ))
    }
}

/// `logits = tanh((x·pᵀ)·aᵀ + a₀)·bᵀ + b₀`, `p` frozen when `frozen`; every parameter from a
/// fixed stream, so every copy starts alike. A batch runs as micro-batches of at most `micro`
/// rows whose gradients add up to the batch's.
struct Tiny {
    vars: VarMap,
    data: Arc<Data>,
    frozen: bool,
    micro: Option<usize>,
    lr: f64,
}

impl Tiny {
    fn new(data: &Arc<Data>, frozen: bool, lr: f64) -> Result<Tiny> {
        let vars = VarMap::new();
        let mut rng = Rng(11);
        {
            let mut map = vars.data().lock().unwrap();
            for (name, shape) in [
                ("a.bias", vec![H]),
                ("a.weight", vec![H, X]),
                ("b.bias", vec![K]),
                ("b.weight", vec![K, H]),
                ("p.weight", vec![X, X]),
            ] {
                let fan = *shape.last().unwrap() as f32;
                let n: usize = shape.iter().product();
                let xs: Vec<f32> = (0..n).map(|_| rng.unit() / fan.sqrt()).collect();
                let t = Tensor::from_vec(xs, shape.as_slice(), &Device::Cpu)?;
                map.insert(name.to_string(), Var::from_tensor(&t)?);
            }
        }
        Ok(Tiny {
            vars,
            data: data.clone(),
            frozen,
            micro: None,
            lr,
        })
    }

    fn logits(&self, x: &Tensor) -> Result<Tensor> {
        let map = self.vars.data().lock().unwrap();
        let t = |n: &str| map[n].as_tensor().clone();
        let h = x.matmul(&t("p.weight").t()?)?;
        let h = h
            .matmul(&t("a.weight").t()?)?
            .broadcast_add(&t("a.bias"))?
            .tanh()?;
        Ok(h.matmul(&t("b.weight").t()?)?.broadcast_add(&t("b.bias"))?)
    }

    /// Mean log loss and accuracy over `rows`.
    fn score(&self, rows: &[usize]) -> Result<(f64, f64)> {
        let (x, y) = self.data.gather(rows)?;
        let logits = self.logits(&x)?;
        let nll = hanzo_nn::loss::cross_entropy(&logits, &y)?.to_scalar::<f32>()? as f64;
        let hit = logits
            .argmax(D::Minus1)?
            .eq(&y)?
            .to_dtype(DType::F32)?
            .mean_all()?;
        Ok((nll, hit.to_scalar::<f32>()? as f64))
    }
}

impl Model for Tiny {
    fn vars(&self) -> &VarMap {
        &self.vars
    }

    fn rate(&self, name: &str) -> Option<f64> {
        (!(self.frozen && name.starts_with("p."))).then_some(self.lr)
    }

    fn step(&mut self, adam: &mut Adam, batch: &Batch, _: &Round) -> Result<Step> {
        let n = batch.rows.len();
        let (mut grads, mut loss) = (None, 0.0);
        for part in batch.rows.chunks(self.micro.unwrap_or(n).max(1)) {
            let (x, y) = self.data.gather(part)?;
            let l = hanzo_nn::loss::cross_entropy(&self.logits(&x)?, &y)?;
            let w = part.len() as f64 / n as f64;
            let g = if w == 1.0 {
                l.backward()?
            } else {
                (&l * w)?.backward()?
            };
            accumulate(&mut grads, g, adam.vars())?;
            loss += w * l.to_scalar::<f32>()? as f64;
        }
        adam.step(&grads.context("an empty batch")?)?;
        Ok(Step {
            tokens: n as u64,
            sums: vec![loss],
        })
    }

    fn line(&self, sums: &[f64], steps: u64, _: f64) -> String {
        format!(" loss {:.4}", sums[0] / steps.max(1) as f64)
    }
}

const ADAM: Hyper = Hyper {
    beta1: 0.9,
    beta2: 0.98,
    eps: 1e-6,
    decay: 0.01,
    clip: 1.0,
};

/// `epochs` passes over `n` rows in batches of `rows`, each epoch in its own seeded order.
fn plan(n: usize, rows: usize, epochs: usize) -> Plan {
    let mut rng = Rng(3);
    let (mut batches, mut ends) = (Vec::new(), Vec::new());
    for _ in 0..epochs {
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            order.swap(i, rng.below(i + 1));
        }
        batches.extend(order.chunks(rows).map(<[usize]>::to_vec));
        ends.push(batches.len());
    }
    let (warm, cool) = span(batches.len(), 0.1, 0.9);
    Plan {
        batches,
        ends,
        warm,
        cool,
    }
}

/// How a test's run goes: outer lr and momentum, round seconds and grace, checkpoint interval
/// and directory, and the batches it takes.
#[derive(Clone)]
struct Cfg {
    outer: f64,
    momentum: f64,
    round: f64,
    grace: f64,
    every: usize,
    out: Option<PathBuf>,
    limit: Option<usize>,
}

impl Cfg {
    fn new(outer: f64, momentum: f64, round: f64) -> Cfg {
        Cfg {
            outer,
            momentum,
            round,
            // past any step here: only a worker that is gone misses it
            grace: 60.0,
            every: 1 << 30,
            out: None,
            limit: None,
        }
    }
}

/// The defaults: each worker's steps and tokens, no validation, θ saved as F32.
struct Plain;

impl Keep for Plain {}

/// A coordinator on a loopback port for `plan` from `begin`, `init`'s parameters trained.
fn coordinate(plan: &Plan, init: &Tiny, begin: Start, cfg: &Cfg) -> Result<Coordinator> {
    let run = Run {
        code: CODE.into(),
        body: serde_json::json!({}),
        params: Params::of(init.vars(), |n| init.rate(n).is_some()),
        plan: plan.clone(),
        limit: cfg.limit.unwrap_or(plan.total()),
        adam: ADAM,
        outer: cfg.outer,
        momentum: cfg.momentum,
        round: cfg.round,
        grace: cfg.grace,
        every: cfg.every,
        out: cfg.out.clone(),
    };
    start(TcpListener::bind("127.0.0.1:0")?, run, begin, Plain)
}

/// θ of `model`: its trained parameters.
fn theta(model: &Tiny) -> Result<Vec<f32>> {
    Params::of(model.vars(), |n| model.rate(n).is_some()).read(model.vars())
}

/// `name` joining the coordinator at `addr` with the model `make` builds.
fn enter(
    addr: SocketAddr,
    name: &str,
    make: impl Fn() -> Result<Tiny>,
) -> Result<Option<Worker<Tiny>>> {
    enroll(addr, name, CODE, |_| make())
}

/// Workers joining `c`, each under its name. None takes a batch before all are members, and a
/// round closes only once every member has reported, so every round holds them all while
/// batches last.
fn crew(
    c: &Coordinator,
    names: &[&str],
    make: impl Fn() -> Result<Tiny> + Send + Clone + 'static,
) -> Vec<JoinHandle<Result<Summary>>> {
    let gate = Arc::new(Barrier::new(names.len()));
    names
        .iter()
        .map(|name| {
            let (addr, name, gate, make) = (c.addr(), name.to_string(), gate.clone(), make.clone());
            std::thread::spawn(move || {
                let joined = enter(addr, &name, make);
                gate.wait();
                joined?.map_or(Ok(Summary::default()), Worker::run)
            })
        })
        .collect()
}

/// The plan trained by one process, one AdamW step per batch.
fn alone(model: &mut Tiny, plan: &Plan) -> Result<Vec<f32>> {
    let mut adam = Adam::new(model.vars(), |n| model.rate(n), ADAM)?;
    let params = Params::of(model.vars(), |n| model.rate(n).is_some());
    let theta = params.read(model.vars())?;
    for b in 0..plan.total() {
        let batch = plan.batch(b);
        adam.schedule(batch.rate);
        let at = Round {
            round: b as u64,
            params: &params,
            theta: &theta,
        };
        model.step(&mut adam, &batch, &at)?;
    }
    params.read(model.vars())
}

fn max(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// The share of `a − b` over `bound` in magnitude.
fn over(a: &[f32], b: &[f32], bound: f32) -> f64 {
    a.iter()
        .zip(b)
        .filter(|(x, y)| (*x - *y).abs() > bound)
        .count() as f64
        / a.len() as f64
}

/// One worker, one step a round, outer lr 0.7 and momentum 0.9: θ_global is its rounds
/// replayed in one process, bit for bit; and at outer lr 1 without momentum, training in one
/// process up to what the bf16 exchange moves.
#[test]
fn one_worker_matches_its_rounds_replayed() -> Result<()> {
    let data = Data::new(96);
    let p = plan(data.len(), 8, 2);
    let make = {
        let data = data.clone();
        move || Tiny::new(&data, false, 3e-3)
    };
    let init = make()?;
    let start0 = theta(&init)?;
    for (lr, mu) in [(0.7, 0.9), (1.0, 0.0)] {
        let c = coordinate(
            &p,
            &init,
            Start::new(start0.clone(), p.total()),
            &Cfg::new(lr, mu, 0.0),
        )?;
        let w = crew(&c, &["a"], make.clone());
        let report = c.join()?;
        let summary = w.into_iter().next().unwrap().join().expect("worker")?;
        assert_eq!(report.merged, p.total());
        assert_eq!(summary.ids, (0..p.total()).collect::<Vec<_>>());
        assert_eq!(report.rounds as usize, p.total());
        let same = replay(&report.log, &p, ADAM, lr, mu, start0.clone(), |_| make())?;
        let moved = max(&start0, &same);
        eprintln!(
            "outer {lr}, μ {mu}: max |θ_cluster − θ_replay| = {:e}; moved {moved:e}",
            max(&report.theta, &same)
        );
        assert!(moved > 1e-3, "training moved nothing");
        assert_eq!(max(&report.theta, &same), 0.0);
        if mu == 0.0 {
            let plain = alone(&mut make()?, &p)?;
            let (far, most) = (
                max(&report.theta, &plain),
                over(&report.theta, &plain, 1e-4),
            );
            eprintln!(
                "against training alone: max {far:e}, {:.3}% over 1e-4",
                100.0 * most
            );
            assert!(
                far <= 2.0 * 3e-3 * p.total() as f32 && most < 0.05,
                "{far} {most}"
            );
        }
    }
    Ok(())
}

/// Three workers, members before any takes a batch, one step a round, outer lr 0.7, momentum
/// 0.9: every round holds all three, and the same rounds (who took which batch) replayed in
/// one process with the same exchange give the same θ bit for bit, whatever order the reports
/// arrived in.
#[test]
fn three_workers_match_their_rounds_replayed() -> Result<()> {
    let data = Data::new(192);
    let p = plan(data.len(), 8, 1);
    let make = {
        let data = data.clone();
        move || Tiny::new(&data, false, 3e-3)
    };
    let init = make()?;
    let start0 = theta(&init)?;
    let c = coordinate(
        &p,
        &init,
        Start::new(start0.clone(), p.total()),
        &Cfg::new(0.7, 0.9, 0.0),
    )?;
    let ws = crew(&c, &["a", "b", "c"], make.clone());
    let report = c.join()?;
    for w in ws {
        w.join().expect("worker")?;
    }
    let sizes: Vec<usize> = report.log.iter().map(Vec::len).collect();
    let same = replay(&report.log, &p, ADAM, 0.7, 0.9, start0.clone(), |_| make())?;
    eprintln!(
        "rounds of {sizes:?} workers: max |θ_cluster − θ_replay| = {:e}",
        max(&report.theta, &same)
    );
    assert_eq!(sizes, vec![3; p.total() / 3]);
    assert!(max(&start0, &same) > 1e-3, "training moved nothing");
    assert_eq!(max(&report.theta, &same), 0.0);
    Ok(())
}

/// A batch run as micro-batches of a quarter of its rows steps as the batch in one pass.
#[test]
fn micro_batches_add_up_to_the_batch() -> Result<()> {
    let data = Data::new(96);
    let p = plan(data.len(), 16, 1);
    let mut one = Tiny::new(&data, false, 3e-3)?;
    let mut four = Tiny::new(&data, false, 3e-3)?;
    four.micro = Some(4);
    let steps = |m: &mut Tiny| -> Result<Vec<f32>> {
        let mut adam = Adam::new(m.vars(), |n| m.rate(n), ADAM)?;
        let params = Params::of(m.vars(), |n| m.rate(n).is_some());
        let start0 = params.read(m.vars())?;
        let at = Round {
            round: 0,
            params: &params,
            theta: &start0,
        };
        let mut losses = Vec::new();
        for b in 0..3 {
            let batch = p.batch(b);
            adam.schedule(batch.rate);
            losses.extend(m.step(&mut adam, &batch, &at)?.sums);
        }
        losses.extend(params.read(m.vars())?.iter().map(|&x| x as f64));
        Ok(losses.into_iter().map(|x| x as f32).collect())
    };
    let (a, b) = (steps(&mut one)?, steps(&mut four)?);
    let diff = max(&a, &b);
    eprintln!("3 steps of 16 rows as 4 × 4: max |Δ| over losses and θ = {diff:e}");
    assert!(diff <= 1e-6, "{diff}");
    assert!(
        max(&a[3..], &theta(&Tiny::new(&data, false, 3e-3)?)?) > 1e-4,
        "nothing moved"
    );
    Ok(())
}

/// Rounds of 10 ms (a dozen steps each), momentum 0.5: two workers, each taking half the plan's
/// batches and so half its sequential steps, end near the accuracy and loss one process
/// reaches over the whole plan (measured: accuracy 0.965–0.968 against 0.982, log loss 0.18
/// against 0.14, from 1.39).
#[test]
fn two_workers_reach_the_loss_of_training_alone() -> Result<()> {
    let data = Data::new(1024);
    let p = plan(data.len(), 16, 4);
    let make = {
        let data = data.clone();
        move || Tiny::new(&data, false, 1e-2)
    };
    let init = make()?;
    let rows: Vec<usize> = (0..data.len()).collect();
    let first = init.score(&rows)?;
    let mut one = make()?;
    alone(&mut one, &p)?;
    let single = one.score(&rows)?;
    let c = coordinate(
        &p,
        &init,
        Start::new(theta(&init)?, p.total()),
        &Cfg::new(0.7, 0.5, 0.01),
    )?;
    let ws = crew(&c, &["a", "b"], make.clone());
    let report = c.join()?;
    let done: Vec<Summary> = ws
        .into_iter()
        .map(|w| w.join().expect("worker"))
        .collect::<Result<_>>()?;
    let two = make()?;
    Params::of(two.vars(), |_| true).write(two.vars(), &report.theta)?;
    let pair = two.score(&rows)?;
    eprintln!(
        "nll/accuracy: start {first:?}, alone {single:?}, two workers {pair:?} ({} rounds, {} + {} batches)",
        report.rounds,
        done[0].ids.len(),
        done[1].ids.len()
    );
    assert_eq!(done[0].ids.len() + done[1].ids.len(), p.total());
    assert!(
        single.0 < 0.5 * first.0,
        "training alone did not learn: {first:?} → {single:?}"
    );
    assert!(
        pair.1 >= single.1 - 0.03 && pair.0 <= 1.5 * single.0,
        "alone {single:?}, cluster {pair:?}"
    );
    Ok(())
}

/// A worker that joins as `name`, takes `n` batches and dies without reporting.
fn die(c: &Coordinator, name: &str, n: usize, len: usize) -> Result<Vec<usize>> {
    Ok(hang(c, name, n, len)?.2)
}

/// A worker that joins as `name`, takes `n` batches and falls silent with its link open, as a
/// machine does whose network goes: its link, and the batches it took.
fn hang(
    c: &Coordinator,
    name: &str,
    n: usize,
    len: usize,
) -> Result<(BufReader<TcpStream>, TcpStream, Vec<usize>)> {
    let stream = TcpStream::connect(c.addr())?;
    let (mut r, mut w) = (BufReader::new(stream.try_clone()?), stream);
    net::send(&mut w, &Msg::Hello { code: CODE.into() })?;
    assert!(matches!(net::recv(&mut r)?, Msg::Setup(_)));
    net::send(
        &mut w,
        &Msg::Ready {
            name: name.into(),
            device: "Cpu".into(),
        },
    )?;
    let inner = match net::recv(&mut r)? {
        Msg::Welcome { inner, .. } => inner,
        m => panic!("expected welcome, got {m:?}"),
    };
    net::get(&mut r, 4 * len)?;
    if inner.is_some() {
        net::get(&mut r, 12 * len)?;
    }
    let mut taken = Vec::new();
    for _ in 0..n {
        net::send(&mut w, &Msg::Take)?;
        match net::recv(&mut r)? {
            Msg::Batch { id, .. } => taken.push(id),
            m => panic!("expected a batch, got {m:?}"),
        }
    }
    Ok((r, w, taken))
}

/// The run `c` coordinates, which must finish within `secs`: a stalled round fails the test
/// rather than holding it.
fn finish(c: Coordinator, secs: u64) -> Result<Report> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || tx.send(c.join()));
    rx.recv_timeout(Duration::from_secs(secs))
        .map_err(|_| anyhow::anyhow!("the run did not finish in {secs} s"))?
}

/// A tiny model over `rows` rows in batches of 8, its init, and a coordinator for it whose
/// grace is `grace` seconds.
fn silent(
    rows: usize,
    grace: f64,
) -> Result<(
    Plan,
    Params,
    Coordinator,
    impl Fn() -> Result<Tiny> + Send + Clone + 'static,
)> {
    let data = Data::new(rows);
    let p = plan(data.len(), 8, 1);
    let make = move || Tiny::new(&data, false, 1e-3);
    let init = make()?;
    let mut cfg = Cfg::new(0.7, 0.9, 0.0);
    cfg.grace = grace;
    let c = coordinate(&p, &init, Start::new(theta(&init)?, p.total()), &cfg)?;
    Ok((p, Params::of(init.vars(), |_| true), c, make))
}

/// A worker whose network goes mid-round (it holds its link open and says nothing) is dropped
/// at the round's deadline plus the grace: the round closes with the deltas that arrived, its
/// batches are requeued and the run finishes.
#[test]
fn a_silent_worker_is_dropped_at_the_deadline() -> Result<()> {
    let (p, params, c, make) = silent(96, 5.0)?;
    let (mut r, _w, taken) = hang(&c, "x", 2, params.len)?;
    assert_eq!(taken, [0, 1]);
    let a = crew(&c, &["a"], make).remove(0);
    let report = finish(c, 60)?;
    let summary = a.join().expect("worker")?;
    assert!(net::recv(&mut r).is_err(), "x is still linked");
    assert_eq!((report.requeued, report.merged), (2, p.total()));
    let mut ids = summary.ids.clone();
    ids.sort();
    assert_eq!(ids, (0..p.total()).collect::<Vec<_>>());
    Ok(())
}

/// A worker whose network goes while it sends its delta (half the frame arrives, then nothing)
/// is dropped at the deadline plus the grace; its batch is requeued and the run finishes.
#[test]
fn a_worker_lost_while_sending_its_delta_is_dropped() -> Result<()> {
    use std::io::Write;
    let (p, params, c, make) = silent(96, 5.0)?;
    let (mut r, mut w, taken) = hang(&c, "y", 1, params.len)?;
    assert_eq!(taken, [0]);
    net::send(
        &mut w,
        &Msg::Delta {
            round: 0,
            tokens: 100,
            steps: 1,
            secs: 0.1,
            sums: vec![1.0],
        },
    )?;
    w.write_all(&((2 * params.len) as u64).to_le_bytes())?;
    w.write_all(&vec![0u8; params.len])?;
    w.flush()?;
    let a = crew(&c, &["a"], make).remove(0);
    let report = finish(c, 60)?;
    a.join().expect("worker")?;
    assert!(net::recv(&mut r).is_err(), "y is still linked");
    assert_eq!((report.requeued, report.merged), (1, p.total()));
    Ok(())
}

/// Forward whatever connects to the returned address to `to`, both ways. The first connection
/// is cut, both sides shut, once `cut` bytes have gone from `to` to it.
fn proxy(to: SocketAddr, cut: usize) -> Result<SocketAddr> {
    fn pump(mut from: TcpStream, mut to: TcpStream, limit: usize) {
        use std::io::{Read, Write};
        std::thread::spawn(move || {
            let (mut buf, mut sent) = (vec![0u8; 1 << 16], 0);
            while sent < limit {
                match from.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) if to.write_all(&buf[..n]).is_ok() => sent += n,
                    Ok(_) => break,
                }
            }
            let _ = from.shutdown(std::net::Shutdown::Both);
            let _ = to.shutdown(std::net::Shutdown::Both);
        });
    }
    let l = TcpListener::bind("127.0.0.1:0")?;
    let addr = l.local_addr()?;
    std::thread::spawn(move || {
        for (i, client) in l.incoming().enumerate() {
            let (Ok(client), Ok(server)) = (client, TcpStream::connect(to)) else {
                continue;
            };
            let limit = if i == 0 { cut } else { usize::MAX };
            pump(
                client.try_clone().unwrap(),
                server.try_clone().unwrap(),
                usize::MAX,
            );
            pump(server, client, limit);
        }
    });
    Ok(addr)
}

/// A worker whose link is cut mid-run (here while an update comes in) joins again on its own,
/// and the run finishes.
#[test]
fn a_worker_whose_link_is_cut_joins_again() -> Result<()> {
    let (p, params, c, make) = silent(192, 60.0)?;
    let via = proxy(c.addr(), 7 * params.len)?;
    let a = std::thread::spawn(move || -> Result<Summary> {
        let w = enter(via, "a", make)?;
        w.context("the run is under way")?.run()
    });
    let report = finish(c, 60)?;
    let summary = a.join().expect("worker")?;
    assert_eq!(report.merged, p.total());
    assert_eq!(summary.links, 2);
    assert!(summary.ids.len() >= p.total());
    Ok(())
}

/// Another build is turned away, a worker that cannot load refuses with its reason, and a
/// worker that drops before reporting has its batches requeued.
#[test]
fn a_dropped_workers_batches_are_requeued() -> Result<()> {
    let data = Data::new(96);
    let p = plan(data.len(), 8, 1);
    let make = {
        let data = data.clone();
        move || Tiny::new(&data, false, 1e-3)
    };
    let init = make()?;
    let params = Params::of(init.vars(), |_| true);
    let c = coordinate(
        &p,
        &init,
        Start::new(theta(&init)?, p.total()),
        &Cfg::new(0.7, 0.9, 0.0),
    )?;

    let stream = TcpStream::connect(c.addr())?;
    let (mut r, mut w) = (BufReader::new(stream.try_clone()?), stream);
    net::send(
        &mut w,
        &Msg::Hello {
            code: "other".into(),
        },
    )?;
    assert!(matches!(net::recv(&mut r)?, Msg::Refuse { .. }));

    let err = enroll(c.addr(), "x", CODE, |_| -> Result<Tiny> {
        anyhow::bail!("no data here")
    });
    assert!(err.is_err_and(|e| e.to_string().contains("no data here")));

    assert_eq!(die(&c, "x", 3, params.len)?, [0, 1, 2]);
    let summary = crew(&c, &["a"], make).remove(0).join().expect("worker")?;
    let report = c.join()?;
    assert_eq!(report.requeued, 3);
    assert_eq!(report.merged, p.total());
    let mut ids = summary.ids.clone();
    ids.sort();
    assert_eq!(
        ids,
        (0..p.total()).collect::<Vec<_>>(),
        "trained {:?}",
        summary.ids
    );
    Ok(())
}

#[test]
fn frozen_parameters_stay_put_and_stay_off_the_wire() -> Result<()> {
    let data = Data::new(96);
    let p = plan(data.len(), 8, 1);
    let init = Tiny::new(&data, true, 1e-3)?;
    let (all, trained) = (
        Params::of(init.vars(), |_| true),
        Params::of(init.vars(), |n| init.rate(n).is_some()),
    );
    assert_eq!(trained.len, all.len - X * X);
    let c = coordinate(
        &p,
        &init,
        Start::new(theta(&init)?, p.total()),
        &Cfg::new(0.7, 0.9, 0.0),
    )?;
    // the worker's parameters, watched from here
    let worker = Tiny::new(&data, true, 1e-3)?;
    let (addr, vars) = (c.addr(), worker.vars.clone());
    let w = std::thread::spawn(move || {
        let share = move || {
            Ok(Tiny {
                vars: vars.clone(),
                data: data.clone(),
                frozen: true,
                micro: None,
                lr: 1e-3,
            })
        };
        enter(addr, "a", share)?.context("the run was done")?.run()
    });
    w.join().expect("worker")?;
    let report = c.join()?;
    assert_eq!(report.theta.len(), trained.len);
    let (before, after) = (all.read(init.vars())?, all.read(worker.vars())?);
    let mut moved = false;
    for (i, name) in all.names.iter().enumerate() {
        let same = before[all.span(i)] == after[all.span(i)];
        if name.starts_with("p.") {
            assert!(same, "{name} moved");
        } else {
            moved |= !same;
        }
    }
    assert!(moved, "nothing trained");
    Ok(())
}

/// A scratch directory.
fn scratch(name: &str) -> Result<PathBuf> {
    let dir = std::env::temp_dir().join(format!("hanzo-train-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn tensors(path: &Path) -> Result<HashMap<String, Vec<f32>>> {
    Ok(hanzo_ml::safetensors::load(path, &Device::Cpu)?
        .into_iter()
        .map(|(k, t)| {
            let v = t.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
            (k, v.to_vec1::<f32>().unwrap())
        })
        .collect())
}

/// Checkpoints every round. The interrupted run stops its coordinator after five of twelve
/// batches and resumes from the checkpoint; mid-round a worker's network goes silent (it is
/// dropped at the deadline) and then the worker dies outright, and it joins again: weights,
/// outer momentum and carry, the worker's moments, step count and carry, the merged batches and
/// the round all end bit-identical to the run uninterrupted.
#[test]
fn a_run_killed_and_resumed_matches_the_run_uninterrupted() -> Result<()> {
    let data = Data::new(96);
    let p = plan(data.len(), 8, 1);
    assert_eq!(p.total(), 12);
    let make = {
        let data = data.clone();
        move || Tiny::new(&data, false, 1e-3)
    };
    let init = make()?;
    let params = Params::of(init.vars(), |_| true);
    let start0 = theta(&init)?;
    let mut cfg = Cfg::new(0.7, 0.9, 0.0);
    cfg.every = 1;
    cfg.grace = 5.0;

    let whole = scratch("whole")?;
    cfg.out = Some(whole.clone());
    let c = coordinate(&p, &init, Start::new(start0.clone(), 12), &cfg)?;
    crew(&c, &["a"], make.clone())
        .remove(0)
        .join()
        .expect("worker")?;
    let a = c.join()?;

    let cut = scratch("cut")?;
    cfg.out = Some(cut.clone());
    let mut first = cfg.clone();
    first.limit = Some(5);
    let c = coordinate(&p, &init, Start::new(start0.clone(), 5), &first)?;
    crew(&c, &["a"], make.clone())
        .remove(0)
        .join()
        .expect("worker")?;
    let stopped = c.join()?;
    assert_eq!((stopped.merged, stopped.rounds), (5, 5));
    let weights = tensors(&cut.join("model.safetensors"))?;
    let back: Vec<f32> = params
        .names
        .iter()
        .flat_map(|n| weights[n].clone())
        .collect();
    let begin = Start::resume(&cut, &params, back, stopped.rounds, 12)?;
    assert_eq!((begin.round, begin.inner["a"].t), (5, 5));
    assert!(Start::resume(&cut, &params, start0.clone(), 4, 12).is_err());
    let c = coordinate(&p, &init, begin, &cfg)?;
    let (mut r, _w, taken) = hang(&c, "x", 1, params.len)?;
    assert_eq!(taken, [5]);
    assert!(net::recv(&mut r).is_err(), "x is still linked");
    assert_eq!(die(&c, "a", 1, params.len)?, [5]);
    let summary = crew(&c, &["a"], make).remove(0).join().expect("worker")?;
    let b = c.join()?;

    assert_eq!(b.requeued, 2);
    assert_eq!(summary.ids, (5..12).collect::<Vec<_>>());
    assert_eq!((a.merged, a.rounds), (b.merged, b.rounds));
    assert_eq!(max(&a.theta, &b.theta), 0.0, "weights");
    assert!(max(&a.theta, &start0) > 1e-4, "training moved nothing");
    for f in [
        "state.safetensors",
        "inner/a.safetensors",
        "model.safetensors",
    ] {
        let (x, y) = (tensors(&whole.join(f))?, tensors(&cut.join(f))?);
        assert_eq!(
            x.keys().collect::<std::collections::BTreeSet<_>>(),
            y.keys().collect()
        );
        for (k, v) in &x {
            assert_eq!(max(v, &y[k]), 0.0, "{f}: {k}");
        }
    }
    std::fs::remove_dir_all(&whole)?;
    std::fs::remove_dir_all(&cut)?;
    Ok(())
}
