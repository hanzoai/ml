//! Fused F32 head_dim-512 online-softmax flash-decode attention.
//!
//! DeepSeek-V4 decode uses head_dim 512, above the 256 cap of the tensor-core flash path,
//! so the engine otherwise falls back to an eager 3-pass attention (QK^T, softmax, ×V) that
//! loses to ds4. This is the single-token decode fast path: one CUDA launch, all-F32 (decode
//! is memory-bound; F32 keeps bit-closeness to the ds4 reference), running-max/running-sum
//! online softmax over the KV cache with the attention sink folded in as an extra logit into
//! the denominator only. See `hanzo-kernels/src/fattn_ds4.cu` for the kernel and the exact
//! ds4 provenance.
//!
//! Shapes: `q` is `[n_head, q_len, 512]` with `q_len` in `1..=8` (`1` for plain single-token
//! decode; `2..=8` for speculative-verify, where the draft proposed a trailing block of tokens
//! and the KV cache already holds all `q_len` new positions, so `kv_len >= q_len`); `k`/`v` are
//! `[kv_len, 512]` (single KV head / MQA, the V4 case) or `[n_kv_head, kv_len, 512]` (GQA);
//! `sinks` is `[n_head]` (optional). `window` is the sliding-window size (0 = attend the whole
//! cache). Attention is causal per query row: row `s` is the token at absolute position
//! `qpos = kv_len - q_len + s` and attends kv rows `(qpos+1-window .. qpos]` clamped to `>= 0`.
//! `scale` multiplies the QK dot (e.g. 1/sqrt(512)).

use crate::{Device, Result, Tensor};

const HEAD_DIM: usize = 512;

/// CPU reference (also the off-device fallback). Standard attention in f32: `QK^T · scale`,
/// softmax with the sink folded into the denominator only (no value contribution), `×V`,
/// causal per query row within the trailing block, over the sliding window. Mathematically
/// identical to the kernel's online softmax.
fn fattn_decode_cpu_f32(
    q: &[f32],             // [n_head, q_len, 512]
    k: &[f32],             // [n_kv_head, kv_len, 512]
    v: &[f32],             // [n_kv_head, kv_len, 512]
    sinks: Option<&[f32]>, // [n_head]
    n_head: usize,
    n_kv_head: usize,
    kv_len: usize,
    q_len: usize,
    window: usize,
    scale: f32,
) -> Vec<f32> {
    let group = n_head / n_kv_head;
    let mut out = vec![0f32; n_head * q_len * HEAD_DIM];
    for h in 0..n_head {
        let kv_head = h / group;
        for s in 0..q_len {
            // Query row s is at absolute position qpos and attends kv rows [start, end):
            // end = qpos + 1 (causal), start = max(0, qpos+1-window) (its own window).
            let qpos = kv_len - q_len + s;
            let end = qpos + 1;
            let start = if window != 0 && end > window { end - window } else { 0 };
            let qbase = (h * q_len + s) * HEAD_DIM;
            let qh = &q[qbase..qbase + HEAD_DIM];

            let mut scores = Vec::with_capacity(end - start);
            let mut m = f32::NEG_INFINITY;
            for j in start..end {
                let base = (kv_head * kv_len + j) * HEAD_DIM;
                let krow = &k[base..base + HEAD_DIM];
                let mut dot = 0f32;
                for d in 0..HEAD_DIM {
                    dot += qh[d] * krow[d];
                }
                let sc = dot * scale;
                m = m.max(sc);
                scores.push(sc);
            }
            if let Some(sk) = sinks {
                m = m.max(sk[h]);
            }

            let mut denom = 0f32;
            let mut acc = vec![0f32; HEAD_DIM];
            for (idx, j) in (start..end).enumerate() {
                let e = (scores[idx] - m).exp();
                denom += e;
                let base = (kv_head * kv_len + j) * HEAD_DIM;
                let vrow = &v[base..base + HEAD_DIM];
                for d in 0..HEAD_DIM {
                    acc[d] += e * vrow[d];
                }
            }
            if let Some(sk) = sinks {
                denom += (sk[h] - m).exp();
            }
            let inv = if denom == 0.0 { 0.0 } else { 1.0 / denom };
            for d in 0..HEAD_DIM {
                out[qbase + d] = acc[d] * inv;
            }
        }
    }
    out
}

/// Derive `(n_head, n_kv_head, kv_len, q_len)` from the tensor shapes and validate the contract.
fn dims(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<(usize, usize, usize, usize)> {
    if k.dims().last().copied() != Some(HEAD_DIM) {
        crate::bail!("fattn_decode: k last dim must be {HEAD_DIM}, got {:?}", k.dims());
    }
    if k.dims() != v.dims() {
        crate::bail!("fattn_decode: k {:?} and v {:?} must have equal shape", k.dims(), v.dims());
    }
    // q is [n_head, q_len, 512] (canonical) or [n_head, 512] (q_len == 1).
    let (n_head, q_len) = match q.dims() {
        [n_head, hd] if *hd == HEAD_DIM => (*n_head, 1usize),
        [n_head, q_len, hd] if *hd == HEAD_DIM => (*n_head, *q_len),
        other => crate::bail!("fattn_decode: q must be [n_head, 512] or [n_head, q_len, 512], got {other:?}"),
    };
    let (n_kv_head, kv_len) = match k.dims() {
        [kv_len, _] => (1usize, *kv_len),
        [n_kv_head, kv_len, _] => (*n_kv_head, *kv_len),
        other => crate::bail!("fattn_decode: k rank must be 2 or 3, got {other:?}"),
    };
    if n_head == 0 || n_kv_head == 0 || n_head % n_kv_head != 0 {
        crate::bail!("fattn_decode: n_head {n_head} must be a nonzero multiple of n_kv_head {n_kv_head}");
    }
    if q_len == 0 || !(1..=8).contains(&q_len) {
        crate::bail!("fattn_decode: q_len {q_len} must be in 1..=8");
    }
    if kv_len < q_len {
        crate::bail!("fattn_decode: kv_len {kv_len} must be >= q_len {q_len}");
    }
    Ok((n_head, n_kv_head, kv_len, q_len))
}

/// Fused F32 head_dim-512 flash-decode attention. Returns the attention output shaped like
/// `q` (`[n_head, q_len, 512]`, `q_len` in `1..=8`). Runs the CUDA kernel on-device and the
/// CPU reference off-device.
pub fn fattn_decode_f32_hd512(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sinks: Option<&Tensor>,
    window: usize,
    scale: f32,
) -> Result<Tensor> {
    let (n_head, n_kv_head, kv_len, q_len) = dims(q, k, v)?;
    let out_shape = q.shape().clone();

    let q = q.to_dtype(crate::DType::F32)?.contiguous()?;
    let k = k.to_dtype(crate::DType::F32)?.contiguous()?;
    let v = v.to_dtype(crate::DType::F32)?.contiguous()?;
    let sinks = match sinks {
        Some(s) => {
            if s.elem_count() != n_head {
                crate::bail!("fattn_decode: sinks must have {n_head} elems, got {}", s.elem_count());
            }
            Some(s.to_dtype(crate::DType::F32)?.contiguous()?)
        }
        None => None,
    };

    match q.device() {
        Device::Cpu => {
            let qv = q.flatten_all()?.to_vec1::<f32>()?;
            let kv = k.flatten_all()?.to_vec1::<f32>()?;
            let vv = v.flatten_all()?.to_vec1::<f32>()?;
            let sv = match &sinks {
                Some(s) => Some(s.flatten_all()?.to_vec1::<f32>()?),
                None => None,
            };
            let out = fattn_decode_cpu_f32(
                &qv,
                &kv,
                &vv,
                sv.as_deref(),
                n_head,
                n_kv_head,
                kv_len,
                q_len,
                window,
                scale,
            );
            Tensor::from_vec(out, out_shape, &Device::Cpu)
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(dev) => cuda_impl(
            dev, &q, &k, &v, sinks.as_ref(), n_head, n_kv_head, kv_len, q_len, window, scale,
            out_shape,
        ),
        _ => crate::bail!("fattn_decode_f32_hd512: unsupported device"),
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn cuda_impl(
    dev: &crate::CudaDevice,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sinks: Option<&Tensor>,
    n_head: usize,
    n_kv_head: usize,
    kv_len: usize,
    q_len: usize,
    window: usize,
    scale: f32,
    out_shape: crate::Shape,
) -> Result<Tensor> {
    use crate::op::BackpropOp;
    use crate::Storage;
    use cudarc::driver::DevicePtr;
    use std::ffi::c_void;

    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *mut c_void;

    let (qg, _) = q.storage_and_layout();
    let (kg, _) = k.storage_and_layout();
    let (vg, _) = v.storage_and_layout();
    let q_ptr = match &*qg {
        Storage::Cuda(s) => s.as_cuda_slice::<f32>()?.device_ptr(&stream).0 as *const f32,
        _ => crate::bail!("fattn_decode: q not on cuda"),
    };
    let k_ptr = match &*kg {
        Storage::Cuda(s) => s.as_cuda_slice::<f32>()?.device_ptr(&stream).0 as *const f32,
        _ => crate::bail!("fattn_decode: k not on cuda"),
    };
    let v_ptr = match &*vg {
        Storage::Cuda(s) => s.as_cuda_slice::<f32>()?.device_ptr(&stream).0 as *const f32,
        _ => crate::bail!("fattn_decode: v not on cuda"),
    };

    let sink_guard = sinks.map(|s| s.storage_and_layout());
    let sinks_ptr = match &sink_guard {
        Some((g, _)) => match &**g {
            Storage::Cuda(s) => s.as_cuda_slice::<f32>()?.device_ptr(&stream).0 as *const f32,
            _ => crate::bail!("fattn_decode: sinks not on cuda"),
        },
        None => std::ptr::null(),
    };

    let out = unsafe { dev.alloc::<f32>(n_head * q_len * HEAD_DIM)? };
    let out_ptr = out.device_ptr(&stream).0 as *mut f32;

    unsafe {
        hanzo_kernels::ffi::hanzo_fattn_decode_f32_hd512(
            stream_ptr,
            q_ptr,
            k_ptr,
            v_ptr,
            sinks_ptr,
            out_ptr,
            n_head as i32,
            n_kv_head as i32,
            kv_len as i32,
            q_len as i32,
            window as i32,
            scale,
        );
    }

    let storage = crate::CudaStorage::wrap_cuda_slice(out, dev.clone());
    Ok(crate::tensor::from_storage(
        Storage::Cuda(storage),
        out_shape,
        BackpropOp::none(),
        false,
    ))
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::{Device, Tensor};
    use rand::prelude::*;

    // Run one case on CUDA and against the CPU reference; return max|Δ|.
    fn max_abs_diff(kv_len: usize, q_len: usize, n_kv_head: usize, with_sink: bool) -> Result<f32> {
        let n_head = 64usize;
        let window = 128usize;
        let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();

        let mut rng = StdRng::seed_from_u64(
            0x00d5_4f10 ^ (kv_len as u64) ^ ((n_kv_head as u64) << 20) ^ ((q_len as u64) << 40),
        );
        let q: Vec<f32> = (0..n_head * q_len * HEAD_DIM).map(|_| rng.random::<f32>() - 0.5).collect();
        let k: Vec<f32> = (0..n_kv_head * kv_len * HEAD_DIM).map(|_| rng.random::<f32>() - 0.5).collect();
        let v: Vec<f32> = (0..n_kv_head * kv_len * HEAD_DIM).map(|_| rng.random::<f32>() - 0.5).collect();
        let sinks: Option<Vec<f32>> = if with_sink {
            Some((0..n_head).map(|_| rng.random::<f32>() - 0.5).collect())
        } else {
            None
        };

        let dev = Device::new_cuda(0)?;
        let qt = Tensor::from_vec(q.clone(), (n_head, q_len, HEAD_DIM), &dev)?;
        let kt = if n_kv_head == 1 {
            Tensor::from_vec(k.clone(), (kv_len, HEAD_DIM), &dev)?
        } else {
            Tensor::from_vec(k.clone(), (n_kv_head, kv_len, HEAD_DIM), &dev)?
        };
        let vt = if n_kv_head == 1 {
            Tensor::from_vec(v.clone(), (kv_len, HEAD_DIM), &dev)?
        } else {
            Tensor::from_vec(v.clone(), (n_kv_head, kv_len, HEAD_DIM), &dev)?
        };
        let st = match &sinks {
            Some(s) => Some(Tensor::from_vec(s.clone(), (n_head,), &dev)?),
            None => None,
        };

        let out = fattn_decode_f32_hd512(&qt, &kt, &vt, st.as_ref(), window, scale)?;
        assert_eq!(out.dims(), &[n_head, q_len, HEAD_DIM]);
        let got = out.flatten_all()?.to_vec1::<f32>()?;
        let want = fattn_decode_cpu_f32(
            &q, &k, &v, sinks.as_deref(), n_head, n_kv_head, kv_len, q_len, window, scale,
        );

        assert_eq!(got.len(), want.len());
        let mut m = 0f32;
        for (a, b) in got.iter().zip(want.iter()) {
            m = m.max((a - b).abs());
        }
        Ok(m)
    }

    #[test]
    fn fattn_decode_matches_cpu_reference() -> Result<()> {
        // (kv_len, q_len, n_kv_head, with_sink); window is 128. q_len in {1,2,3,8} covers plain
        // decode plus speculative-verify widths; kv_len in {7,40,130} makes the per-row window
        // clip differently across the trailing block (130 straddles the window boundary so early
        // rows clip and late rows don't; 40 sits fully inside the window; 7 < window entirely).
        let cases = [
            (130usize, 1usize, 1usize, true), // MQA, plain decode, window active (byte-compat S=1)
            (130, 2, 1, true),                // MQA, verify width 2, window straddles the block
            (130, 3, 1, false),               // MQA, verify width 3, no sink
            (130, 8, 1, true),                // MQA, verify width 8, window straddles the block
            (40, 8, 1, true),                 // MQA, width 8 fully inside the window
            (40, 3, 8, false),                // GQA (8 kv heads), width 3, inside window
            (7, 1, 1, true),                  // MQA, kv_len < window, plain decode
            (7, 2, 1, true),                  // MQA, kv_len < window, verify width 2
            (7, 3, 8, true),                  // GQA, kv_len < window, verify width 3, sink
            (130, 8, 8, true),                // GQA, verify width 8, window straddles the block
        ];
        for (kv_len, q_len, n_kv_head, with_sink) in cases {
            let d = max_abs_diff(kv_len, q_len, n_kv_head, with_sink)?;
            println!(
                "fattn_decode kv_len={kv_len} q_len={q_len} n_kv_head={n_kv_head} sink={with_sink} max|Δ|={d:e}"
            );
            assert!(
                d < 1e-4,
                "kv_len={kv_len} q_len={q_len} n_kv_head={n_kv_head} sink={with_sink}: max|Δ|={d} >= 1e-4"
            );
        }
        Ok(())
    }
}
