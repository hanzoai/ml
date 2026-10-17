//! The gated-delta-net kernels on the device against plain f64 loops, at Qwen3.8-Flash-Next's
//! shapes (16 key heads, 48 value heads, head width 128, conv width 4): a prompt from an empty
//! state, then a continuation chunk and a single decode step on the state it left.
#![cfg(feature = "rocm")]

use hanzo_ml::rocm_backend::{gdn_conv_rocm, gdn_mixer_rocm};
use hanzo_ml::{DType, Device, Result, Tensor};

const HK: usize = 16;
const HV: usize = 48;
const D: usize = 128;
const KW: usize = 4;
const CH: usize = (2 * HK + HV) * D;
const EPS: f64 = 1e-6;

struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) % 20001) as f64 / 10000.0 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f64) -> Vec<f64> {
        (0..n).map(|_| self.unit() * scale).collect()
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// max |got - want| / max |want|
fn rel(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-30);
    got.iter()
        .zip(want)
        .map(|(g, w)| (*g as f64 - w).abs())
        .fold(0f64, f64::max)
        / scale
}

fn f32s(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

fn host(t: &Tensor) -> Result<Vec<f64>> {
    Ok(t.to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f64::from)
        .collect())
}

/// The conv: out[t] = silu(sum_j w[j] h[t+1+j]) over h = history ++ x; the history left behind
/// is the last KW entries of h. `x` is [seq, CH]; `hist` is [CH, KW].
fn conv_ref(x: &[f64], w: &[f64], hist: &[f64], seq: usize) -> (Vec<f64>, Vec<f64>) {
    let mut out = vec![0.0; seq * CH];
    let mut next = vec![0.0; CH * KW];
    for c in 0..CH {
        let mut h: Vec<f64> = hist[c * KW..(c + 1) * KW].to_vec();
        h.extend((0..seq).map(|t| x[t * CH + c]));
        for t in 0..seq {
            let a: f64 = (0..KW).map(|j| w[c * KW + j] * h[t + 1 + j]).sum();
            out[t * CH + c] = a * sigmoid(a);
        }
        next[c * KW..(c + 1) * KW].copy_from_slice(&h[seq..seq + KW]);
    }
    (out, next)
}

/// The mixer for one sequence: `qkv` [seq, CH]; `a`, `b` [seq, HV]; `z` [seq, HV*D]; `state`
/// [HV, D, D] advanced in place. Returns [seq, HV*D].
#[allow(clippy::too_many_arguments)]
fn mixer_ref(
    qkv: &[f64],
    a: &[f64],
    b: &[f64],
    z: &[f64],
    a_coef: &[f64],
    dt: &[f64],
    w: &[f64],
    state: &mut [f64],
    seq: usize,
    sigmoid_gate: bool,
) -> Vec<f64> {
    let mut out = vec![0.0; seq * HV * D];
    for t in 0..seq {
        let row = &qkv[t * CH..(t + 1) * CH];
        for h in 0..HV {
            let kh = h % HK;
            let unit = |v: &[f64]| {
                let r = 1.0 / (v.iter().map(|x| x * x).sum::<f64>() + 1e-6).sqrt();
                v.iter().map(|x| x * r).collect::<Vec<f64>>()
            };
            let q: Vec<f64> = unit(&row[kh * D..(kh + 1) * D])
                .into_iter()
                .map(|x| x / (D as f64).sqrt())
                .collect();
            let k = unit(&row[(HK + kh) * D..(HK + kh + 1) * D]);
            let v = &row[2 * HK * D + h * D..2 * HK * D + (h + 1) * D];
            let beta = sigmoid(b[t * HV + h]);
            let x = a[t * HV + h] + dt[h];
            let decay = (a_coef[h] * (1.0 + x.exp()).ln()).exp();
            let s = &mut state[h * D * D..(h + 1) * D * D];
            let mut y = vec![0.0; D];
            for col in 0..D {
                let mut kv = 0.0;
                for j in 0..D {
                    s[j * D + col] *= decay;
                    kv += s[j * D + col] * k[j];
                }
                let delta = (v[col] - kv) * beta;
                for j in 0..D {
                    s[j * D + col] += k[j] * delta;
                    y[col] += s[j * D + col] * q[j];
                }
            }
            let r = 1.0 / (y.iter().map(|x| x * x).sum::<f64>() / D as f64 + EPS).sqrt();
            for col in 0..D {
                let zv = z[t * HV * D + h * D + col];
                let gate = if sigmoid_gate { sigmoid(zv) } else { zv * sigmoid(zv) };
                out[t * HV * D + h * D + col] = y[col] * r * w[col] * gate;
            }
        }
    }
    out
}

#[test]
fn conv_matches_the_reference_from_a_start_and_on_a_carried_state() -> Result<()> {
    let gpu = Device::new_rocm(0)?;
    let mut rng = Lcg(0xc0de);
    let w = rng.vec(CH * KW, 0.5);
    let wg = Tensor::from_vec(f32s(&w), (CH, KW), &gpu)?;
    for state_dtype in [DType::F32, DType::F16] {
        // The device keeps the state in its own dtype, so the reference reads it back from there.
        let state = Tensor::from_vec(f32s(&rng.vec(CH * KW, 1.0)), (1, CH, KW), &gpu)?
            .to_dtype(state_dtype)?;
        let mut hist = vec![0.0; CH * KW];
        for (seq, fresh) in [(37usize, true), (5, false), (1, false)] {
            let x = rng.vec(seq * CH, 1.0);
            if !fresh {
                hist = host(&state)?;
            }
            let (want, next) = conv_ref(&x, &w, &hist, seq);
            let xg = Tensor::from_vec(f32s(&x), (1, seq, CH), &gpu)?;
            let got = gdn_conv_rocm(&xg, &wg, &state, fresh)?;
            let e = rel(&got.flatten_all()?.to_vec1::<f32>()?, &want);
            let s = rel(&host(&state)?.iter().map(|&v| v as f32).collect::<Vec<_>>(), &next);
            println!("conv {state_dtype:?} seq={seq} fresh={fresh}: out {e:.2e}, state {s:.2e}");
            assert!(e < 1e-5, "conv output off by {e}");
            // An f16 state rounds what it keeps.
            let bound = if state_dtype == DType::F16 { 1e-3 } else { 1e-6 };
            assert!(s < bound, "conv state off by {s}");
        }
    }
    Ok(())
}

#[test]
fn mixer_matches_the_reference_over_prompt_chunk_and_decode() -> Result<()> {
    let gpu = Device::new_rocm(0)?;
    let mut rng = Lcg(0x6d78);
    let a_coef: Vec<f64> = (0..HV).map(|_| -0.2 - rng.unit().abs()).collect();
    let dt = rng.vec(HV, 1.0);
    let w: Vec<f64> = (0..D).map(|_| 0.8 + 0.4 * rng.unit().abs()).collect();
    let (acg, dtg, wg) = (
        Tensor::from_vec(f32s(&a_coef), HV, &gpu)?,
        Tensor::from_vec(f32s(&dt), HV, &gpu)?,
        Tensor::from_vec(f32s(&w), D, &gpu)?,
    );
    for sigmoid_gate in [true, false] {
        let mut state = vec![0.0; HV * D * D];
        let sg = Tensor::zeros((1, HV, D, D), DType::F32, &gpu)?;
        for seq in [37usize, 5, 1] {
            let qkv = rng.vec(seq * CH, 1.0);
            let (a, b) = (rng.vec(seq * HV, 2.0), rng.vec(seq * HV, 2.0));
            let z = rng.vec(seq * HV * D, 2.0);
            let want = mixer_ref(
                &qkv,
                &a,
                &b,
                &z,
                &a_coef,
                &dt,
                &w,
                &mut state,
                seq,
                sigmoid_gate,
            );
            let got = gdn_mixer_rocm(
                &Tensor::from_vec(f32s(&qkv), (1, seq, CH), &gpu)?,
                &Tensor::from_vec(f32s(&a), (1, seq, HV), &gpu)?,
                &Tensor::from_vec(f32s(&b), (1, seq, HV), &gpu)?,
                &Tensor::from_vec(f32s(&z), (1, seq, HV * D), &gpu)?,
                &acg,
                &dtg,
                &wg,
                &sg,
                HK,
                EPS as f32,
                sigmoid_gate,
            )?;
            let e = rel(&got.flatten_all()?.to_vec1::<f32>()?, &want);
            let s = rel(&sg.flatten_all()?.to_vec1::<f32>()?, &state);
            println!("mixer sigmoid={sigmoid_gate} seq={seq}: out {e:.2e}, state {s:.2e}");
            assert!(e < 1e-4, "mixer output off by {e}");
            assert!(s < 1e-4, "mixer state off by {s}");
        }
    }
    Ok(())
}
