//! Numeric oracle for the matrix-core (WMMA) flash-attention kernel: run the GPU kernel and an exact
//! CPU reference attention on the SAME inputs, assert max_abs_diff within f32 tolerance (nbad == 0).
//! Requires a gfx11 GPU + ROCm toolchain (kernels JIT-compile via hipcc). Skipped if no device.

use half::{bf16, f16};
use hanzo_rocm_kernels::ops::FlashAttnShape;
use hanzo_rocm_kernels::OpLauncher;
use rocm_rs::hip::{Device, DeviceMemory, Stream};

const DH: usize = 128;

struct Case {
    b: usize,
    hq: usize,
    hkv: usize,
    lq: usize,
    lk: usize,
    causal: bool,
}

fn cpu_reference(c: &Case, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let gqa = c.hq / c.hkv;
    let scale = 1.0f32 / (DH as f32).sqrt();
    let total = c.b * c.hq * c.lq * DH;
    let mut out = vec![0.0f32; total];

    // Per-(batch,head) attention is independent; fan it out over threads -- the O(L^2) scalar oracle
    // at L=2048 is otherwise the test's wall-clock bottleneck. Split `out` into N contiguous chunks,
    // each spanning a whole number of (b,h) slabs; thread `t` owns global slab indices [start, end).
    let bh = c.b * c.hq;
    let slab = c.lq * DH;
    let nthreads = std::thread::available_parallelism()
        .map_or(8, |n| n.get())
        .min(bh.max(1));
    let slabs_per = bh.div_ceil(nthreads);
    std::thread::scope(|scope| {
        let mut rest = out.as_mut_slice();
        let mut base = 0usize;
        while base < bh {
            let count = slabs_per.min(bh - base);
            let (chunk, tail) = rest.split_at_mut(count * slab);
            rest = tail;
            let start = base;
            scope.spawn(move || {
                for local in 0..count {
                    let idx = start + local;
                    let b = idx / c.hq;
                    let h = idx % c.hq;
                    let kvh = h / gqa;
                    let orows = &mut chunk[local * slab..][..slab];
                    for i in 0..c.lq {
                        let kmax = if c.causal { i + 1 } else { c.lk };
                        let qrow = &q[(((b * c.hq + h) * c.lq) + i) * DH..][..DH];
                        let mut scores = vec![0.0f32; kmax];
                        let mut m = f32::NEG_INFINITY;
                        for (j, s) in scores.iter_mut().enumerate() {
                            let krow = &k[(((b * c.hkv + kvh) * c.lk) + j) * DH..][..DH];
                            let mut acc = 0.0f32;
                            for d in 0..DH {
                                acc += qrow[d] * krow[d];
                            }
                            *s = acc * scale;
                            if *s > m {
                                m = *s;
                            }
                        }
                        let mut l = 0.0f32;
                        for s in scores.iter_mut() {
                            *s = (*s - m).exp();
                            l += *s;
                        }
                        let orow = &mut orows[i * DH..][..DH];
                        for (j, &p) in scores.iter().enumerate() {
                            let vrow = &v[(((b * c.hkv + kvh) * c.lk) + j) * DH..][..DH];
                            let w = p / l;
                            for d in 0..DH {
                                orow[d] += w * vrow[d];
                            }
                        }
                    }
                }
            });
            base += count;
        }
    });
    out
}

fn gen(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s >> 11) as f32) / (1u64 << 53) as f32 - 0.5) * 0.6
        })
        .collect()
}

fn run_case_f16(launcher: &OpLauncher, stream: &Stream, c: &Case) -> (f32, usize) {
    let qn = c.b * c.hq * c.lq * DH;
    let kn = c.b * c.hkv * c.lk * DH;
    let q = gen(qn, 1);
    let k = gen(kn, 2);
    let v = gen(kn, 3);

    let to16 = |x: &[f32]| x.iter().map(|&f| f16::from_f32(f)).collect::<Vec<_>>();
    let (qh, kh, vh) = (to16(&q), to16(&k), to16(&v));

    let mut dq = DeviceMemory::<f16>::new(qn).unwrap();
    let mut dk = DeviceMemory::<f16>::new(kn).unwrap();
    let mut dv = DeviceMemory::<f16>::new(kn).unwrap();
    let mut dout = DeviceMemory::<f16>::new(qn).unwrap();
    dq.copy_from_host(&qh).unwrap();
    dk.copy_from_host(&kh).unwrap();
    dv.copy_from_host(&vh).unwrap();

    let shape = FlashAttnShape {
        batch: c.b as i32,
        q_heads: c.hq as i32,
        kv_heads: c.hkv as i32,
        q_len: c.lq as i32,
        kv_len: c.lk as i32,
        head_dim: DH as i32,
        scale: 1.0f32 / (DH as f32).sqrt(),
        causal: c.causal,
    };
    launcher
        .launch_flash_attn(stream, "f16", shape, &dq, &dk, &dv, &mut dout)
        .unwrap();
    stream.synchronize().unwrap();

    let mut got = vec![f16::ZERO; qn];
    dout.copy_to_host(&mut got).unwrap();
    let got: Vec<f32> = got.iter().map(|x| x.to_f32()).collect();

    // reference computed on the f16-rounded inputs to isolate kernel error from input quantization.
    let qf: Vec<f32> = qh.iter().map(|x| x.to_f32()).collect();
    let kf: Vec<f32> = kh.iter().map(|x| x.to_f32()).collect();
    let vf: Vec<f32> = vh.iter().map(|x| x.to_f32()).collect();
    let reference = cpu_reference(c, &qf, &kf, &vf);

    let tol = 2e-2f32;
    let mut max_abs = 0.0f32;
    let mut nbad = 0usize;
    for (g, r) in got.iter().zip(reference.iter()) {
        let d = (g - r).abs();
        if d > max_abs {
            max_abs = d;
        }
        if d > tol {
            nbad += 1;
        }
    }
    (max_abs, nbad)
}

fn run_case_bf16(launcher: &OpLauncher, stream: &Stream, c: &Case) -> (f32, usize) {
    let qn = c.b * c.hq * c.lq * DH;
    let kn = c.b * c.hkv * c.lk * DH;
    let q = gen(qn, 11);
    let k = gen(kn, 22);
    let v = gen(kn, 33);

    let to16 = |x: &[f32]| x.iter().map(|&f| bf16::from_f32(f)).collect::<Vec<_>>();
    let (qh, kh, vh) = (to16(&q), to16(&k), to16(&v));

    let mut dq = DeviceMemory::<bf16>::new(qn).unwrap();
    let mut dk = DeviceMemory::<bf16>::new(kn).unwrap();
    let mut dv = DeviceMemory::<bf16>::new(kn).unwrap();
    let mut dout = DeviceMemory::<bf16>::new(qn).unwrap();
    dq.copy_from_host(&qh).unwrap();
    dk.copy_from_host(&kh).unwrap();
    dv.copy_from_host(&vh).unwrap();

    let shape = FlashAttnShape {
        batch: c.b as i32,
        q_heads: c.hq as i32,
        kv_heads: c.hkv as i32,
        q_len: c.lq as i32,
        kv_len: c.lk as i32,
        head_dim: DH as i32,
        scale: 1.0f32 / (DH as f32).sqrt(),
        causal: c.causal,
    };
    launcher
        .launch_flash_attn(stream, "bf16", shape, &dq, &dk, &dv, &mut dout)
        .unwrap();
    stream.synchronize().unwrap();

    let mut got = vec![bf16::ZERO; qn];
    dout.copy_to_host(&mut got).unwrap();
    let got: Vec<f32> = got.iter().map(|x| x.to_f32()).collect();

    let qf: Vec<f32> = qh.iter().map(|x| x.to_f32()).collect();
    let kf: Vec<f32> = kh.iter().map(|x| x.to_f32()).collect();
    let vf: Vec<f32> = vh.iter().map(|x| x.to_f32()).collect();
    let reference = cpu_reference(c, &qf, &kf, &vf);

    // bf16 has ~8 mantissa bits; the WMMA accumulate is f32 but the bf16 P round-trip widens the gap.
    let tol = 8e-2f32;
    let mut max_abs = 0.0f32;
    let mut nbad = 0usize;
    for (g, r) in got.iter().zip(reference.iter()) {
        let d = (g - r).abs();
        if d > max_abs {
            max_abs = d;
        }
        if d > tol {
            nbad += 1;
        }
    }
    (max_abs, nbad)
}

fn device_or_skip() -> Option<(OpLauncher, Stream)> {
    let device = Device::new(0).ok()?;
    device.set_current().ok()?;
    let launcher = OpLauncher::new(&device).ok()?;
    let stream = device.get_stream().ok()?;
    Some((launcher, stream))
}

/// Full coordinator matrix: L in {512,1024,2048} x GQA {8:1, 4:1, 1:1} x {causal, non-causal}.
/// The CPU oracle is O(L^2) scalar per head, so head counts shrink as L grows to keep it tractable
/// while still exercising every GQA ratio and the tile/boundary logic at each length.
fn matrix_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    // (L, hq): hq shrinks as L grows so the O(L^2) scalar oracle stays tractable.
    for &(lq, hq) in &[(512usize, 32usize), (1024, 16), (2048, 8)] {
        // GQA ratios 8:1, 4:1, 1:1 -> hkv = hq/8, hq/4, hq.
        for &hkv in &[hq / 8, hq / 4, hq] {
            let hkv = hkv.max(1);
            for &causal in &[true, false] {
                cases.push(Case {
                    b: 1,
                    hq,
                    hkv,
                    lq,
                    lk: lq,
                    causal,
                });
            }
        }
    }
    cases
}

#[test]
fn flash_attn_f16_oracle() {
    let Some((launcher, stream)) = device_or_skip() else {
        eprintln!("no ROCm device; skipping flash_attn_f16_oracle");
        return;
    };
    // full matrix + a few odd-boundary lengths (non-multiple-of-64 tile tails, single token).
    let mut cases = matrix_cases();
    cases.extend([
        Case {
            b: 1,
            hq: 32,
            hkv: 8,
            lq: 130,
            lk: 130,
            causal: true,
        },
        Case {
            b: 2,
            hq: 16,
            hkv: 4,
            lq: 33,
            lk: 33,
            causal: false,
        },
        Case {
            b: 1,
            hq: 8,
            hkv: 8,
            lq: 200,
            lk: 200,
            causal: true,
        },
        Case {
            b: 1,
            hq: 8,
            hkv: 2,
            lq: 1,
            lk: 1,
            causal: true,
        },
    ]);
    for c in &cases {
        let (max_abs, nbad) = run_case_f16(&launcher, &stream, c);
        eprintln!(
            "f16 b{} hq{} hkv{} L{} causal={}: max_abs={:.5} nbad={}",
            c.b, c.hq, c.hkv, c.lq, c.causal, max_abs, nbad
        );
        assert_eq!(nbad, 0, "f16 flash attn mismatch (max_abs={max_abs})");
    }
}

#[test]
fn flash_attn_bf16_oracle() {
    let Some((launcher, stream)) = device_or_skip() else {
        eprintln!("no ROCm device; skipping flash_attn_bf16_oracle");
        return;
    };
    let mut cases = matrix_cases();
    cases.push(Case {
        b: 1,
        hq: 8,
        hkv: 2,
        lq: 130,
        lk: 130,
        causal: true,
    });
    for c in &cases {
        let (max_abs, nbad) = run_case_bf16(&launcher, &stream, c);
        eprintln!(
            "bf16 b{} hq{} hkv{} L{} causal={}: max_abs={:.5} nbad={}",
            c.b, c.hq, c.hkv, c.lq, c.causal, max_abs, nbad
        );
        assert_eq!(nbad, 0, "bf16 flash attn mismatch (max_abs={max_abs})");
    }
}
