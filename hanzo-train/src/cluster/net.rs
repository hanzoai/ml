//! The cluster's transport over TCP. A frame is a little-endian u64 length and that many bytes.
//! A message is one JSON frame; a tensor frame follows some messages and holds every trained
//! parameter, flattened in sorted name order ([`Params`]), as raw little-endian F32 or bf16. An
//! empty tensor frame stands for zeros.
//!
//! Every link is bounded ([`link`]): keepalive finds a peer that vanished (its machine asleep,
//! its cable out) in about a minute and a half, a write that makes no progress fails after
//! [`WRITE`] and a read after [`READ`]. A peer that is there but silent is the coordinator's to
//! drop, at its round's deadline.

use crate::adam::Hyper;
use anyhow::{ensure, Result};
use hanzo_ml::{DType, Device, Tensor, Var};
use hanzo_nn::VarMap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

/// Longest a read waits: past a joining worker's load and a coordinator's validation, which
/// are the longest silences a live peer keeps.
pub const READ: Duration = Duration::from_secs(3600);

/// Longest a write waits without progress: the other side reads each frame as it comes.
pub const WRITE: Duration = Duration::from_secs(120);

/// Keepalive: the first probe after this much silence, then one each [`PROBE`], and the link
/// fails after [`PROBES`] go unanswered.
const IDLE: Duration = Duration::from_secs(30);
const PROBE: Duration = Duration::from_secs(10);
const PROBES: u32 = 6;

/// Largest message frame; a tensor frame is checked against its exact size instead.
const MESSAGE: u64 = 1 << 26;

/// Bytes per socket call: macOS refuses a send or receive over `i32::MAX` bytes (EINVAL).
const SLICE: usize = 1 << 28;

/// Elements per parallel chunk in the tensor conversions.
const CHUNK: usize = 1 << 16;

/// What a worker loads, and what its build and parameters must hash to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Setup {
    /// The build: a worker of another is turned away.
    pub code: String,
    /// [`Params::digest`] of the coordinator's model.
    pub params: String,
    /// Batches in the whole plan, in its warmup and in its final decay: the inner schedule
    /// ([`super::rate`]).
    pub total: usize,
    pub warm: usize,
    pub cool: usize,
    /// The inner optimizer's constants.
    pub adam: Hyper,
    /// What else a worker needs to load the model, as the run's owner states it.
    pub body: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "msg", rename_all = "lowercase")]
pub enum Msg {
    /// Worker, first: the build it runs.
    Hello { code: String },
    /// Coordinator: what to load.
    Setup(Setup),
    /// Either side, then it closes: why the other is turned away.
    Refuse { reason: String },
    /// Worker: loaded, and every hash matched. `name` is the machine's, unique in the run: its
    /// optimizer state is kept under it. `device` says what it computes on.
    Ready { name: String, device: String },
    /// Coordinator, then θ_global as F32: the worker's id, the round it joins and the seconds
    /// left in it. With `inner`, the worker's kept optimizer state follows: its step count here,
    /// then one F32 frame of first moments, second moments and rounding carry.
    Welcome {
        id: u64,
        round: u64,
        left: f64,
        inner: Option<u64>,
    },
    /// Worker: the next batch.
    Take,
    /// Coordinator: batch `id` of the plan (its position sets the learning rate), the epoch it
    /// belongs to and its rows.
    Batch {
        id: usize,
        epoch: usize,
        rows: Vec<usize>,
    },
    /// Coordinator: nothing to hand out this round.
    Empty,
    /// Worker, then θ_local − θ_global as bf16: this round's work. `sums` adds up what each
    /// step reported ([`super::Step`]); `secs` is the wall time from the round's start to its
    /// last step's end.
    Delta {
        round: u64,
        tokens: u64,
        steps: u64,
        secs: f64,
        sums: Vec<f64>,
    },
    /// Coordinator, then the change to θ_global as bf16: the next round and its seconds. With
    /// `keep`, the round starts a checkpoint: the worker answers with [`Msg::State`].
    Update { round: u64, left: f64, keep: bool },
    /// Coordinator: the plan is done; with `keep`, the worker answers with [`Msg::State`].
    Done { keep: bool },
    /// Worker, then one F32 frame of first moments, second moments and rounding carry: its
    /// optimizer state and step count `t`, for a checkpoint.
    State { t: u64 },
}

/// Ready `s` for the cluster: no Nagle delay, keepalive, and a bound on every read and write.
pub fn link(s: &TcpStream) -> Result<()> {
    s.set_nodelay(true)?;
    s.set_read_timeout(Some(READ))?;
    s.set_write_timeout(Some(WRITE))?;
    let alive = socket2::TcpKeepalive::new()
        .with_time(IDLE)
        .with_interval(PROBE)
        .with_retries(PROBES);
    socket2::SockRef::from(s).set_tcp_keepalive(&alive)?;
    Ok(())
}

fn length(r: &mut impl Read) -> Result<u64> {
    let mut n = [0u8; 8];
    r.read_exact(&mut n)?;
    Ok(u64::from_le_bytes(n))
}

pub fn send(w: &mut impl Write, m: &Msg) -> Result<()> {
    let body = serde_json::to_vec(m)?;
    let mut b = Vec::with_capacity(8 + body.len());
    b.extend((body.len() as u64).to_le_bytes());
    b.extend(body);
    w.write_all(&b)?;
    w.flush()?;
    Ok(())
}

pub fn recv(r: &mut impl Read) -> Result<Msg> {
    let n = length(r)?;
    ensure!(n <= MESSAGE, "message frame of {n} bytes");
    let mut b = vec![0; n as usize];
    r.read_exact(&mut b)?;
    Ok(serde_json::from_slice(&b)?)
}

/// Write a tensor frame (empty: zeros).
pub fn put(w: &mut impl Write, bytes: &[u8]) -> Result<()> {
    w.write_all(&(bytes.len() as u64).to_le_bytes())?;
    for c in bytes.chunks(SLICE) {
        w.write_all(c)?;
    }
    w.flush()?;
    Ok(())
}

/// Read a tensor frame of exactly `size` bytes; `None` for an empty one.
pub fn get(r: &mut impl Read, size: usize) -> Result<Option<Vec<u8>>> {
    let n = length(r)?;
    if n == 0 {
        return Ok(None);
    }
    ensure!(
        n == size as u64,
        "tensor frame of {n} bytes, expected {size}"
    );
    let mut b = vec![0; size];
    for c in b.chunks_mut(SLICE) {
        r.read_exact(c)?;
    }
    Ok(Some(b))
}

/// Round to the nearest bf16, ties to even.
pub fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

pub fn widen(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// `e` as bf16 bytes; `e` keeps what rounding dropped, to be added to the next one.
pub fn quantize(e: &mut [f32]) -> Vec<u8> {
    let mut out = vec![0u8; 2 * e.len()];
    out.par_chunks_mut(2 * CHUNK)
        .zip(e.par_chunks_mut(CHUNK))
        .for_each(|(o, e)| {
            for (o, x) in o.as_chunks_mut::<2>().0.iter_mut().zip(e) {
                let q = bf16(*x);
                *o = q.to_le_bytes();
                *x -= widen(q);
            }
        });
    out
}

/// A worker's delta from θ_local `e`: `e − θ_global` plus what rounding dropped last time
/// (`carry`), as bf16 bytes; `carry` keeps what rounding drops now.
pub fn delta(mut e: Vec<f32>, theta: &[f32], carry: &mut Vec<f32>) -> Vec<u8> {
    e.par_iter_mut()
        .zip(theta)
        .zip(&*carry)
        .for_each(|((e, g), c)| *e = *e - g + c);
    let q = quantize(&mut e);
    *carry = e;
    q
}

/// `acc += w · bytes`, the bytes read as bf16.
pub fn accumulate(acc: &mut [f32], bytes: &[u8], w: f32) {
    acc.par_chunks_mut(CHUNK)
        .zip(bytes.par_chunks(2 * CHUNK))
        .for_each(|(a, b)| {
            for (a, h) in a.iter_mut().zip(b.as_chunks::<2>().0) {
                *a += w * widen(u16::from_le_bytes(*h));
            }
        });
}

pub fn pack(xs: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; 4 * xs.len()];
    out.par_chunks_mut(4 * CHUNK)
        .zip(xs.par_chunks(CHUNK))
        .for_each(|(o, x)| {
            for (o, x) in o.as_chunks_mut::<4>().0.iter_mut().zip(x) {
                *o = x.to_le_bytes();
            }
        });
    out
}

pub fn unpack(bytes: &[u8]) -> Vec<f32> {
    bytes
        .par_chunks(4 * CHUNK)
        .flat_map_iter(|b| b.as_chunks::<4>().0.iter().map(|w| f32::from_le_bytes(*w)))
        .collect()
}

/// A model's trained parameters in sorted name order, flattened into one vector: θ.
pub struct Params {
    pub names: Vec<String>,
    pub shapes: Vec<Vec<usize>>,
    pub offsets: Vec<usize>,
    /// Elements in all.
    pub len: usize,
}

impl Params {
    /// The parameters of `vars` that `trained` names.
    pub fn of(vars: &VarMap, trained: impl Fn(&str) -> bool) -> Params {
        let data = vars.data().lock().expect("varmap lock");
        let mut names: Vec<String> = data.keys().filter(|n| trained(n)).cloned().collect();
        names.sort();
        let shapes: Vec<Vec<usize>> = names.iter().map(|n| data[n].dims().to_vec()).collect();
        let mut offsets = Vec::with_capacity(names.len());
        let mut len = 0;
        for s in &shapes {
            offsets.push(len);
            len += s.iter().product::<usize>();
        }
        Params {
            names,
            shapes,
            offsets,
            len,
        }
    }

    /// SHA-256 of the names and shapes.
    pub fn digest(&self) -> String {
        let mut h = Sha256::new();
        for (n, s) in self.names.iter().zip(&self.shapes) {
            h.update(format!("{n}{s:?}\n"));
        }
        h.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Where parameter `i` sits in the flat vector.
    pub fn span(&self, i: usize) -> std::ops::Range<usize> {
        let n: usize = self.shapes[i].iter().product();
        self.offsets[i]..self.offsets[i] + n
    }

    /// Every parameter of `vars`, flattened, as F32 on the CPU.
    pub fn read(&self, vars: &VarMap) -> Result<Vec<f32>> {
        let data = vars.data().lock().expect("varmap lock");
        let mut out = vec![0f32; self.len];
        for (i, n) in self.names.iter().enumerate() {
            let t = data[n].as_tensor().flatten_all()?.to_dtype(DType::F32)?;
            out[self.span(i)].copy_from_slice(&t.to_vec1::<f32>()?);
        }
        Ok(out)
    }

    /// Set every parameter of `vars` from `flat`.
    pub fn write(&self, vars: &VarMap, flat: &[f32]) -> Result<()> {
        let data = vars.data().lock().expect("varmap lock");
        for (i, n) in self.names.iter().enumerate() {
            let v = &data[n];
            let t = Tensor::from_slice(&flat[self.span(i)], self.shapes[i].as_slice(), v.device())?;
            v.set(&t.to_dtype(v.dtype())?)?;
        }
        Ok(())
    }

    /// A new F32 map on `dev` holding `flat`.
    pub fn map(&self, flat: &[f32], dev: &Device) -> Result<VarMap> {
        let out = VarMap::new();
        {
            let mut data = out.data().lock().expect("varmap lock");
            for (i, n) in self.names.iter().enumerate() {
                let t = Tensor::from_slice(&flat[self.span(i)], self.shapes[i].as_slice(), dev)?;
                data.insert(n.clone(), Var::from_tensor(&t)?);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{TcpListener, TcpStream};

    #[test]
    fn frames_round_trip() -> Result<()> {
        let l = TcpListener::bind("127.0.0.1:0")?;
        let mut a = TcpStream::connect(l.local_addr()?)?;
        let (mut b, _) = l.accept()?;
        let msgs = vec![
            Msg::Hello { code: "ab".into() },
            Msg::Setup(Setup {
                code: "ab".into(),
                params: "p".into(),
                total: 53322,
                warm: 1599,
                cool: 51723,
                adam: Hyper {
                    beta1: 0.9,
                    beta2: 0.98,
                    eps: 1e-6,
                    decay: 0.01,
                    clip: 1.0,
                },
                body: serde_json::json!({"stage": "a", "lr": 3e-5, "corpus": "c"}),
            }),
            Msg::Refuse {
                reason: "no".into(),
            },
            Msg::Ready {
                name: "evo".into(),
                device: "Cpu".into(),
            },
            Msg::Welcome {
                id: 3,
                round: 7,
                left: 12.5,
                inner: Some(41),
            },
            Msg::Take,
            Msg::Batch {
                id: 9,
                epoch: 1,
                rows: vec![4, 1, 7],
            },
            Msg::Empty,
            Msg::Delta {
                round: 7,
                tokens: 12345,
                steps: 4,
                secs: 1.25,
                sums: vec![0.5, 0.25, 0.125],
            },
            Msg::Update {
                round: 8,
                left: 300.0,
                keep: true,
            },
            Msg::State { t: 41 },
            Msg::Done { keep: false },
        ];
        let xs: Vec<f32> = vec![0.0, -0.0, 1.0, -3.5, 1e-40, 3.0e38, 0.1, -7.25e-5];
        let writer = {
            let msgs = msgs.clone();
            let xs = xs.clone();
            std::thread::spawn(move || -> Result<()> {
                for m in &msgs {
                    send(&mut a, m)?;
                }
                put(&mut a, &pack(&xs))?;
                put(&mut a, &quantize(&mut xs.clone()))?;
                put(&mut a, &[])?;
                Ok(())
            })
        };
        for m in &msgs {
            assert_eq!(&recv(&mut b)?, m);
        }
        let full = unpack(&get(&mut b, 4 * xs.len())?.expect("f32 frame"));
        assert_eq!(
            full.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            xs.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        let half = get(&mut b, 2 * xs.len())?.expect("bf16 frame");
        let mut back = vec![0f32; xs.len()];
        accumulate(&mut back, &half, 1.0);
        for (x, y) in xs.iter().zip(&back) {
            // relative for normal values; bf16 subnormals keep fewer bits
            assert!(
                (x - y).abs() <= x.abs() / 256.0 + f32::MIN_POSITIVE,
                "{x} → {y}"
            );
        }
        assert_eq!(get(&mut b, 2 * xs.len())?, None);
        writer.join().expect("writer")?;
        Ok(())
    }

    #[test]
    fn rounding_is_nearest_even_and_feeds_back() {
        assert_eq!(bf16(1.0), 0x3f80);
        // halfway between 0x3f80 and 0x3f81: ties to the even one
        assert_eq!(bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert_eq!(bf16(f32::from_bits(0x3f80_8001)), 0x3f81);
        assert_eq!(widen(bf16(-2.5)), -2.5);
        let xs: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.37).sin() * 1e-3).collect();
        let mut e = xs.clone();
        let q = quantize(&mut e);
        let mut back = vec![0f32; xs.len()];
        accumulate(&mut back, &q, 1.0);
        for i in 0..xs.len() {
            assert_eq!(back[i] + e[i], xs[i], "residual is exact at {i}");
        }
        // a delta carries what the last one's rounding dropped
        let (theta, local): (Vec<f32>, Vec<f32>) =
            (vec![1.0; 1000], xs.iter().map(|x| 1.0 + x).collect());
        let mut carry = e.clone();
        let q = delta(local.clone(), &theta, &mut carry);
        let mut sent = vec![0f32; xs.len()];
        accumulate(&mut sent, &q, 1.0);
        for i in 0..xs.len() {
            assert_eq!(sent[i] + carry[i], local[i] - theta[i] + e[i], "at {i}");
        }
    }
}
