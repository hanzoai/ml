// Regression gate for the heterogeneous PP-ring Metal deadlock.
//
// Root cause: hanzo_ml::quantized::moe_combine's generic fallback did broadcast_mul(ys, scores)
// with no dtype reconciliation. scores come from moe_route in f32; on a ring rank running a GPU
// compute dtype without a native moe_combine kernel (Metal -> bf16), ys is bf16 and scores f32, so
// the mul panicked ("dtype mismatch in mul, lhs: BF16, rhs: F32"). The panic killed the PP worker
// thread while the daemon's main loop kept spinning -> the ring "deadlock" (100% CPU, GPU idle, no
// tokens). CUDA/ROCm ranks have a native kernel; CPU ranks run at f32 (== scores), so only a Metal
// rank hit it. See engine models::quantized_qwen3_moe::FusedMoe::forward -> moe_combine.
#![cfg(feature = "metal")]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

use hanzo_ml::backend::BackendStorage;
use hanzo_ml::quantized::moe_combine;
use hanzo_ml::{CpuStorage, DType, Device, Result, Storage, Tensor};

fn metal() -> Option<Device> {
    match Device::new_metal(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("no metal device, skipping: {e}");
            None
        }
    }
}

// f32 reference: out[t,n] = sum_k ys[t,k,n] * scores[t,k].
fn combine_ref(ys: &[f32], scores: &[f32], t: usize, topk: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; t * n];
    for ti in 0..t {
        for k in 0..topk {
            let s = scores[ti * topk + k];
            for ni in 0..n {
                out[ti * n + ni] += ys[(ti * topk + k) * n + ni] * s;
            }
        }
    }
    out
}

// moe_combine must accept ys in the model compute dtype (bf16/f16) with f32 scores, exactly as the
// Metal MoE decode path feeds it. Pre-fix this panicked with a dtype mismatch on Metal.
#[test]
fn metal_moe_combine_mixed_dtype_matches_cpu() {
    let Some(dev) = metal() else { return };
    let (t, topk, n) = (4usize, 8usize, 512usize);

    let ys_f32: Vec<f32> = (0..t * topk * n).map(|i| ((i % 23) as f32) * 0.1 - 1.0).collect();
    let scores_f32: Vec<f32> = (0..t * topk).map(|i| ((i % 7) as f32 + 1.0) / 28.0).collect();
    let want = combine_ref(&ys_f32, &scores_f32, t, topk, n);

    // scores are always f32 out of moe_route; ys carries the compute dtype.
    let scores = Tensor::from_vec(scores_f32, (t, topk), &dev).unwrap();
    for ys_dtype in [DType::F32, DType::F16, DType::BF16] {
        let ys = Tensor::from_vec(ys_f32.clone(), (t, topk, n), &dev)
            .unwrap()
            .to_dtype(ys_dtype)
            .unwrap();
        let got = moe_combine(&ys, &scores)
            .unwrap_or_else(|e| panic!("moe_combine failed for ys={ys_dtype:?}: {e}"));
        assert_eq!(got.dims(), &[t, n], "shape for {ys_dtype:?}");
        assert_eq!(got.dtype(), ys_dtype, "output dtype must follow ys");
        let got = got.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let tol = match ys_dtype {
            DType::F32 => 1e-5,
            DType::F16 => 3e-2,
            _ => 2e-1,
        };
        let maxerr = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(maxerr <= tol, "ys={ys_dtype:?} maxerr {maxerr} > {tol}");
        eprintln!("moe_combine ys={ys_dtype:?} OK maxerr={maxerr:.3e}");
    }
}

// The task's 2-rank micro-harness: rank0 CPU (main), rank1 Metal (spawned worker), a loopback TCP
// pair. Mirrors hanzo-quant/src/distributed send_right (tensor_to_f32 via to_cpu_storage) and
// recv_left (from_vec + to_dtype). Proves a Metal rank exchanges an activation over the socket and
// the bytes survive the round-trip.
fn cpu_to_f32(cpu: &CpuStorage) -> Vec<f32> {
    match cpu {
        CpuStorage::F32(x) => x.clone(),
        CpuStorage::F16(x) => x.iter().map(|v| v.to_f32()).collect(),
        CpuStorage::BF16(x) => x.iter().map(|v| v.to_f32()).collect(),
        _ => panic!("unsupported dtype"),
    }
}

fn tensor_to_f32(xs: &Tensor) -> Result<Vec<f32>> {
    let xs = xs.contiguous()?;
    let storage = xs.storage_and_layout().0;
    match &*storage {
        Storage::Cpu(s) => Ok(cpu_to_f32(s)),
        Storage::Metal(s) => Ok(cpu_to_f32(&s.to_cpu_storage()?)),
        _ => panic!("unexpected storage"),
    }
}

fn send(stream: &mut TcpStream, v: &[f32]) {
    let raw = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
    stream.write_all(raw).unwrap();
    stream.flush().unwrap();
}

fn recv(stream: &mut TcpStream, n: usize) -> Vec<f32> {
    let mut buf = vec![0f32; n];
    let raw = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, n * 4) };
    stream.read_exact(raw).unwrap();
    buf
}

#[test]
fn metal_rank_exchanges_activation_over_socket() {
    let Some(dev) = metal() else { return };
    let (rows, cols) = (3usize, 2048usize);
    let n = rows * cols;
    let payload: Vec<f32> = (0..n).map(|i| ((i % 17) as f32) * 0.5 - 4.0).collect();
    let expect = payload.clone();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();

    // rank1 = Metal worker on a spawned thread (pipeline_parallel_worker is spawned).
    let worker = std::thread::spawn(move || {
        let (mut sock, _) = listener.accept().unwrap();
        sock.set_nodelay(true).unwrap();
        let v = recv(&mut sock, n);
        // recv_left: upload f32 -> Metal -> cast to a GPU compute dtype.
        let h = Tensor::from_vec(v, (rows, cols), &dev).unwrap().to_dtype(DType::BF16).unwrap();
        // send_right: Metal -> host f32 readback (identity round-trip).
        let f = tensor_to_f32(&h).unwrap();
        send(&mut sock, &f);
    });

    let mut sock = TcpStream::connect(addr).unwrap();
    sock.set_nodelay(true).unwrap();
    sock.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
    send(&mut sock, &payload);
    let got = recv(&mut sock, n);
    worker.join().unwrap();

    for (i, (&e, &g)) in expect.iter().zip(&got).enumerate() {
        assert!((e - g).abs() <= 3e-1, "byte {i}: sent {e}, got {g}");
    }
    eprintln!("metal rank exchanged {n} activations over socket OK");
}
