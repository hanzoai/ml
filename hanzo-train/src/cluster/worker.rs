//! A worker: inner AdamW on the batches it takes, one delta per round.
//!
//! It keeps θ_global in F32 on the CPU, updated by the coordinator's bf16 changes. Each round
//! it trains θ_local on its device, at the learning rate of each batch's position in the plan,
//! until the round's time is up; sends θ_local − θ_global plus what bf16 rounding dropped last
//! time; then resets θ_local to the new θ_global. Its AdamW state carries across rounds, goes
//! to the coordinator at checkpoints, and comes back when a worker of its name joins a run.
//!
//! A worker outlives its link. When the link fails (the network drops, the coordinator drops it
//! from a round it did not report in time, the coordinator restarts), it connects again with
//! backoff and joins as a new process would, taking θ_global and the optimizer state the
//! coordinator keeps under its name (fresh without one); only its loaded model carries over. It
//! stops when the run is done, on an error the network did not cause, or once the coordinator
//! has refused it for `GONE` or been out of reach for `ABSENT`.
//! [`join`] makes it a member of the run; [`Worker::run`] works its rounds.

use super::net::{self, Msg, Params, Setup};
use super::{device, Batch, Model, Round};
use crate::adam::Adam;
use crate::gpu;
use anyhow::{bail, Context, Result};
use hanzo_ml::Device;
use std::io::{BufReader, ErrorKind, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::{Duration, Instant};

/// Longest a connection attempt waits.
const CONNECT: Duration = Duration::from_secs(30);

/// Longest wait between attempts; the first waits a second, each after twice the last.
const BACKOFF: Duration = Duration::from_secs(60);

/// How long the coordinator's port may refuse a worker: time for a coordinator to restart and
/// load, past which nothing serves the run any more.
const GONE: Duration = Duration::from_secs(1800);

/// How long a worker tries to reach its coordinator at all.
const ABSENT: Duration = Duration::from_secs(12 * 3600);

/// What a worker did.
#[derive(Debug, Default)]
pub struct Summary {
    /// Batches trained, in order, those of a round lost with its link among them.
    pub ids: Vec<usize>,
    pub tokens: u64,
    pub rounds: u64,
    /// Times it joined the run.
    pub links: u64,
}

/// What a worker keeps across links: the run's setup, the model it loaded for it, θ's layout
/// and the device it trains on.
struct Held<M> {
    setup: Setup,
    model: M,
    params: Params,
    dev: Device,
}

/// One link to the coordinator: the worker's id in the run, the round it is in and the seconds
/// left in it, its optimizer, its copy of θ_global and its rounding carry.
struct Link {
    r: BufReader<TcpStream>,
    w: TcpStream,
    id: u64,
    round: u64,
    left: f64,
    adam: Adam,
    theta: Vec<f32>,
    carry: Vec<f32>,
}

/// A worker in a run: where its coordinator is, its name and build, its model and its link.
pub struct Worker<M: Model> {
    addr: SocketAddr,
    name: String,
    code: String,
    held: Held<M>,
    link: Link,
}

/// Send the optimizer state: step count, then moments and carry as one F32 frame.
fn keep(w: &mut impl Write, adam: &Adam, carry: &[f32]) -> Result<()> {
    let mut flat = adam.moments()?;
    flat.extend_from_slice(carry);
    net::send(w, &Msg::State { t: adam.t })?;
    net::put(w, &net::pack(&flat))
}

/// Refuse the coordinator on `w` for `reason`, as far as the link still carries it.
fn refuse<T>(w: &mut TcpStream, reason: String) -> Result<T> {
    let _ = net::send(
        w,
        &Msg::Refuse {
            reason: reason.clone(),
        },
    );
    bail!(reason)
}

/// Whether `e` is the link's failure, not the run's: the worker joins again. EHOSTDOWN, which
/// macOS answers for a LAN host that is asleep or gone, has no kind of its own.
fn transient(e: &anyhow::Error) -> bool {
    e.chain().any(|c| {
        c.downcast_ref::<std::io::Error>().is_some_and(|e| {
            matches!(
                e.kind(),
                ErrorKind::ConnectionRefused
                    | ErrorKind::ConnectionReset
                    | ErrorKind::ConnectionAborted
                    | ErrorKind::NotConnected
                    | ErrorKind::BrokenPipe
                    | ErrorKind::TimedOut
                    | ErrorKind::WouldBlock
                    | ErrorKind::UnexpectedEof
                    | ErrorKind::Interrupted
                    | ErrorKind::HostUnreachable
                    | ErrorKind::NetworkUnreachable
                    | ErrorKind::NetworkDown
                    | ErrorKind::AddrNotAvailable
            ) || e.raw_os_error() == Some(libc::EHOSTDOWN)
        })
    })
}

fn refused(e: &anyhow::Error) -> bool {
    e.chain().any(|c| {
        c.downcast_ref::<std::io::Error>()
            .is_some_and(|e| e.kind() == ErrorKind::ConnectionRefused)
    })
}

/// One attempt to join the run at `addr` under `name`, running build `code`. The first builds
/// the model with `load` from the coordinator's setup (a failure there refuses the coordinator
/// with its reason, and is final); its parameters must hash as the coordinator's do. Later ones
/// keep it and need the same setup. `None`: the run is done.
fn attempt<M: Model, L: FnOnce(&Setup) -> Result<M>>(
    addr: SocketAddr,
    name: &str,
    code: &str,
    held: &mut Option<Held<M>>,
    load: &mut Option<L>,
) -> Result<Option<Link>> {
    let stream = TcpStream::connect_timeout(&addr, CONNECT)?;
    net::link(&stream)?;
    let mut r = BufReader::with_capacity(1 << 20, stream.try_clone()?);
    let mut w = stream;
    net::send(&mut w, &Msg::Hello { code: code.into() })?;
    let setup = match net::recv(&mut r)? {
        Msg::Setup(s) => s,
        Msg::Refuse { reason } => bail!("refused: {reason}"),
        m => bail!("expected setup, got {m:?}"),
    };
    let h = match held {
        Some(h) if h.setup == setup => h,
        Some(_) => return refuse(&mut w, "the coordinator runs another setup now".into()),
        None => {
            if setup.code != code {
                return refuse(
                    &mut w,
                    format!("build {code}, the coordinator runs {}", setup.code),
                );
            }
            let load = load.take().context("the model loads once")?;
            let model = match load(&setup) {
                Ok(m) => m,
                Err(e) => return refuse(&mut w, format!("load: {e:#}")),
            };
            let params = Params::of(model.vars(), |n| model.rate(n).is_some());
            if params.digest() != setup.params {
                return refuse(
                    &mut w,
                    "the model's parameters differ from the coordinator's".into(),
                );
            }
            let dev = device(model.vars()).context("a model without parameters")?;
            held.insert(Held {
                setup,
                model,
                params,
                dev,
            })
        }
    };
    net::send(
        &mut w,
        &Msg::Ready {
            name: name.into(),
            device: h.model.describe(),
        },
    )?;
    let (id, round, left, inner) = match net::recv(&mut r)? {
        Msg::Welcome {
            id,
            round,
            left,
            inner,
        } => (id, round, left, inner),
        Msg::Done { .. } => return Ok(None),
        Msg::Refuse { reason } => bail!("refused: {reason}"),
        m => bail!("expected welcome, got {m:?}"),
    };
    let n = h.params.len;
    let theta = net::unpack(&net::get(&mut r, 4 * n)?.context("θ_global")?);
    let state = match inner {
        Some(_) => Some(net::unpack(
            &net::get(&mut r, 12 * n)?.context("optimizer state")?,
        )),
        None => None,
    };
    h.params.write(h.model.vars(), &theta)?;
    let mut adam = Adam::new(h.model.vars(), |n| h.model.rate(n), h.setup.adam)?;
    let mut carry = vec![0f32; n];
    if let (Some(t), Some(state)) = (inner, state) {
        adam.load(&state[..2 * n], t)?;
        carry.copy_from_slice(&state[2 * n..]);
    }
    eprintln!(
        "worker {id} ({name}): joined round {round}{}",
        inner.map_or(String::new(), |t| format!(", optimizer state at step {t}"))
    );
    Ok(Some(Link {
        r,
        w,
        id,
        round,
        left,
        adam,
        theta,
        carry,
    }))
}

/// Join the run at `addr` under `name`, trying again with backoff while the link fails. `None`:
/// the run is done.
fn enter<M: Model, L: FnOnce(&Setup) -> Result<M>>(
    addr: SocketAddr,
    name: &str,
    code: &str,
    held: &mut Option<Held<M>>,
    load: &mut Option<L>,
) -> Result<Option<Link>> {
    let (since, mut shut, mut wait) = (Instant::now(), None, Duration::from_secs(1));
    loop {
        let e = match attempt(addr, name, code, held, load) {
            Err(e) if transient(&e) => e,
            r => return r,
        };
        let now = Instant::now();
        shut = if refused(&e) {
            shut.or(Some(now))
        } else {
            None
        };
        if now - since >= ABSENT || shut.is_some_and(|t| now - t >= GONE) {
            return Err(e.context(format!(
                "{addr}: out of reach for {:.0} min",
                (now - since).as_secs_f64() / 60.0
            )));
        }
        eprintln!("worker ({name}): {addr}: {e:#}; again in {wait:?}");
        std::thread::sleep(wait);
        wait = (wait * 2).min(BACKOFF);
    }
}

/// Join the run the coordinator at `addr` serves, under `name`, running build `code`; `load`
/// builds the model its setup describes. Once joined, the worker is a member of its round.
/// `None`: the run was done.
pub fn join<M: Model>(
    addr: SocketAddr,
    name: &str,
    code: &str,
    load: impl FnOnce(&Setup) -> Result<M>,
) -> Result<Option<Worker<M>>> {
    let mut held = None;
    let Some(link) = enter(addr, name, code, &mut held, &mut Some(load))? else {
        return Ok(None);
    };
    Ok(Some(Worker {
        addr,
        name: name.into(),
        code: code.into(),
        held: held.expect("a link holds a model"),
        link,
    }))
}

impl<M: Model> Worker<M> {
    /// Work rounds until the plan is done, joining again whenever the link fails.
    pub fn run(self) -> Result<Summary> {
        let Worker {
            addr,
            name,
            code,
            held,
            link,
        } = self;
        let mut held = Some(held);
        let mut link = Some(link);
        let mut summary = Summary {
            links: 1,
            ..Summary::default()
        };
        let meter = gpu::Meter::start();
        loop {
            let h = held.as_mut().expect("held");
            let e = match rounds(h, link.take().expect("linked"), &mut summary, &meter) {
                Ok(()) => return Ok(summary),
                Err(e) if transient(&e) => e,
                Err(e) => return Err(e),
            };
            // the round is lost with the link: what it alone needed goes with it
            h.model.end();
            eprintln!("worker ({name}): lost the coordinator: {e:#}; joining again");
            let none: &mut Option<fn(&Setup) -> Result<M>> = &mut None;
            match enter(addr, &name, &code, &mut held, none)? {
                Some(l) => {
                    link = Some(l);
                    summary.links += 1;
                }
                None => return Ok(summary),
            }
        }
    }
}

/// Work rounds on `link` until the plan is done.
fn rounds<M: Model>(
    h: &mut Held<M>,
    link: Link,
    summary: &mut Summary,
    meter: &gpu::Meter,
) -> Result<()> {
    let Link {
        mut r,
        mut w,
        id,
        mut round,
        mut left,
        mut adam,
        mut theta,
        mut carry,
    } = link;
    let Held {
        setup,
        model,
        params,
        dev,
    } = h;
    let n = params.len;
    loop {
        let begun = Instant::now();
        let deadline = begun + Duration::from_secs_f64(left);
        let (mut tokens, mut steps, mut sums) = (0u64, 0u64, Vec::<f64>::new());
        loop {
            net::send(&mut w, &Msg::Take)?;
            let (b, epoch, rows) = match net::recv(&mut r)? {
                Msg::Batch { id, epoch, rows } => (id, epoch, rows),
                Msg::Empty => break,
                m => bail!("expected a batch, got {m:?}"),
            };
            let batch = Batch {
                id: b,
                epoch,
                rows,
                rate: super::rate(b, setup.warm, setup.cool, setup.total),
            };
            adam.schedule(batch.rate);
            let at = Round {
                round,
                params,
                theta: &theta,
            };
            let s = model.step(&mut adam, &batch, &at)?;
            tokens += s.tokens;
            steps += 1;
            if sums.len() < s.sums.len() {
                sums.resize(s.sums.len(), 0.0);
            }
            for (a, x) in sums.iter_mut().zip(&s.sums) {
                *a += x;
            }
            summary.ids.push(b);
            if Instant::now() >= deadline {
                break;
            }
        }
        model.end();
        dev.synchronize()?;
        let secs = begun.elapsed().as_secs_f64();
        let t = Instant::now();
        let delta = if tokens > 0 {
            net::delta(params.read(model.vars())?, &theta, &mut carry)
        } else {
            Vec::new()
        };
        net::send(
            &mut w,
            &Msg::Delta {
                round,
                tokens,
                steps,
                secs,
                sums: sums.clone(),
            },
        )?;
        net::put(&mut w, &delta)?;
        let sent = t.elapsed().as_secs_f64();
        summary.tokens += tokens;
        summary.rounds += 1;
        let t = Instant::now();
        let next = net::recv(&mut r)?;
        let waited = t.elapsed().as_secs_f64();
        let line = format!(
            "worker {id} round {round}: {steps} steps {:.0} tok/s{} gpu {} free {} | delta \
             sent {sent:.1}s, waited {waited:.1}s",
            tokens as f64 / secs.max(1e-9),
            model.line(&sums, steps, secs),
            meter
                .utilization()
                .map_or("n/a".into(), |u| format!("{u:.0}%")),
            gpu::free().map_or("n/a".into(), |f| format!("{f}%")),
        );
        match next {
            Msg::Update {
                round: at,
                left: l,
                keep: k,
            } => {
                let t = Instant::now();
                if let Some(u) = net::get(&mut r, 2 * n)? {
                    net::accumulate(&mut theta, &u, 1.0);
                }
                let got = t.elapsed().as_secs_f64();
                params.write(model.vars(), &theta)?;
                if k {
                    keep(&mut w, &adam, &carry)?;
                }
                eprintln!(
                    "{line}, update {got:.1}s{}",
                    if k { ", state kept" } else { "" }
                );
                (round, left) = (at, l);
            }
            Msg::Done { keep: k } => {
                if k {
                    keep(&mut w, &adam, &carry)?;
                }
                eprintln!("{line}; done");
                return Ok(());
            }
            m => bail!("expected an update, got {m:?}"),
        }
    }
}
