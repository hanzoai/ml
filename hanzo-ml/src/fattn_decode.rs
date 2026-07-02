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
//! Shapes: `q` is `[n_head, 1, 512]`; `k`/`v` are `[kv_len, 512]` (single KV head / MQA,
//! the V4 case) or `[n_kv_head, kv_len, 512]` (GQA); `sinks` is `[n_head]` (optional).
//! `window` is the sliding-window size (0 = attend the whole cache); the query is the newest
//! token, attending the last `window` rows. `scale` multiplies the QK dot (e.g. 1/sqrt(512)).

use crate::{Device, Result, Tensor};

const HEAD_DIM: usize = 512;

/// CPU reference (also the off-device fallback). Standard attention in f32: `QK^T · scale`,
/// softmax with the sink folded into the denominator only (no value contribution), `×V`,
/// over the sliding window. Mathematically identical to the kernel's online softmax.
fn fattn_decode_cpu_f32(
    q: &[f32],             // [n_head, 512]
    k: &[f32],             // [n_kv_head, kv_len, 512]
    v: &[f32],             // [n_kv_head, kv_len, 512]
    sinks: Option<&[f32]>, // [n_head]
    n_head: usize,
    n_kv_head: usize,
    kv_len: usize,
    window: usize,
    scale: f32,
) -> Vec<f32> {
    let group = n_head / n_kv_head;
    let start = if window != 0 && kv_len > window {
        kv_len - window
    } else {
        0
    };
    let mut out = vec![0f32; n_head * HEAD_DIM];
    for h in 0..n_head {
        let kv_head = h / group;
        let qh = &q[h * HEAD_DIM..h * HEAD_DIM + HEAD_DIM];

        let mut scores = Vec::with_capacity(kv_len - start);
        let mut m = f32::NEG_INFINITY;
        for j in start..kv_len {
            let base = (kv_head * kv_len + j) * HEAD_DIM;
            let krow = &k[base..base + HEAD_DIM];
            let mut dot = 0f32;
            for d in 0..HEAD_DIM {
                dot += qh[d] * krow[d];
            }
            let s = dot * scale;
            m = m.max(s);
            scores.push(s);
        }
        if let Some(sk) = sinks {
            m = m.max(sk[h]);
        }

        let mut denom = 0f32;
        let mut acc = vec![0f32; HEAD_DIM];
        for (idx, j) in (start..kv_len).enumerate() {
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
            out[h * HEAD_DIM + d] = acc[d] * inv;
        }
    }
    out
}

/// Derive `(n_head, n_kv_head, kv_len)` from the tensor shapes and validate the contract.
fn dims(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<(usize, usize, usize)> {
    if q.dims().last().copied() != Some(HEAD_DIM) {
        crate::bail!("fattn_decode: q last dim must be {HEAD_DIM}, got {:?}", q.dims());
    }
    if k.dims().last().copied() != Some(HEAD_DIM) {
        crate::bail!("fattn_decode: k last dim must be {HEAD_DIM}, got {:?}", k.dims());
    }
    if k.dims() != v.dims() {
        crate::bail!("fattn_decode: k {:?} and v {:?} must have equal shape", k.dims(), v.dims());
    }
    let n_head = q.elem_count() / HEAD_DIM;
    let (n_kv_head, kv_len) = match k.dims() {
        [kv_len, _] => (1usize, *kv_len),
        [n_kv_head, kv_len, _] => (*n_kv_head, *kv_len),
        other => crate::bail!("fattn_decode: k rank must be 2 or 3, got {other:?}"),
    };
    if n_head == 0 || n_kv_head == 0 || n_head % n_kv_head != 0 {
        crate::bail!("fattn_decode: n_head {n_head} must be a nonzero multiple of n_kv_head {n_kv_head}");
    }
    Ok((n_head, n_kv_head, kv_len))
}

/// Fused F32 head_dim-512 flash-decode attention. Returns the attention output shaped like
/// `q` (`[n_head, 1, 512]`). Runs the CUDA kernel on-device and the CPU reference off-device.
pub fn fattn_decode_f32_hd512(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sinks: Option<&Tensor>,
    window: usize,
    scale: f32,
) -> Result<Tensor> {
    let (n_head, n_kv_head, kv_len) = dims(q, k, v)?;
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
                window,
                scale,
            );
            Tensor::from_vec(out, out_shape, &Device::Cpu)
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(dev) => cuda_impl(
            dev, &q, &k, &v, sinks.as_ref(), n_head, n_kv_head, kv_len, window, scale, out_shape,
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

    let out = unsafe { dev.alloc::<f32>(n_head * HEAD_DIM)? };
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
    fn max_abs_diff(kv_len: usize, n_kv_head: usize, with_sink: bool) -> Result<f32> {
        let n_head = 64usize;
        let window = 128usize;
        let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();

        let mut rng = StdRng::seed_from_u64(0x00d5_4f10 ^ (kv_len as u64) ^ ((n_kv_head as u64) << 20));
        let q: Vec<f32> = (0..n_head * HEAD_DIM).map(|_| rng.random::<f32>() - 0.5).collect();
        let k: Vec<f32> = (0..n_kv_head * kv_len * HEAD_DIM).map(|_| rng.random::<f32>() - 0.5).collect();
        let v: Vec<f32> = (0..n_kv_head * kv_len * HEAD_DIM).map(|_| rng.random::<f32>() - 0.5).collect();
        let sinks: Option<Vec<f32>> = if with_sink {
            Some((0..n_head).map(|_| rng.random::<f32>() - 0.5).collect())
        } else {
            None
        };

        let dev = Device::new_cuda(0)?;
        let qt = Tensor::from_vec(q.clone(), (n_head, 1, HEAD_DIM), &dev)?;
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
        let got = out.flatten_all()?.to_vec1::<f32>()?;
        let want = fattn_decode_cpu_f32(
            &q, &k, &v, sinks.as_deref(), n_head, n_kv_head, kv_len, window, scale,
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
        // (kv_len, n_kv_head, with_sink): window is 128.
        //   130 -> above the window (start=2, 128 rows attended); 7 -> below the window.
        let cases = [
            (130usize, 1usize, true),  // MQA, sink, sliding window active
            (7, 1, true),              // MQA, sink, kv_len < window
            (130, 1, false),           // MQA, no sink
            (7, 1, false),             // MQA, no sink, kv_len < window
            (130, 8, true),            // GQA (8 kv heads), sink
        ];
        for (kv_len, n_kv_head, with_sink) in cases {
            let d = max_abs_diff(kv_len, n_kv_head, with_sink)?;
            println!(
                "fattn_decode kv_len={kv_len} n_kv_head={n_kv_head} sink={with_sink} max|Δ|={d:e}"
            );
            assert!(
                d < 1e-4,
                "kv_len={kv_len} n_kv_head={n_kv_head} sink={with_sink}: max|Δ|={d} >= 1e-4"
            );
        }
        Ok(())
    }
}
