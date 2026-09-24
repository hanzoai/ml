//! The routed-expert path on the device, against the CPU, at a 512-expert model's shapes.
//!
//! A 512-expert, top-10 MoE (hidden 2560, expert width 640) stores gate/up as IQ3_S or IQ4_XS
//! and down as IQ4_NL or Q8_0. Each case runs `indexed_moe_forward` on ROCm for a decode step and
//! a prompt, with f16 and f32 activations, and compares every routed slot with the CPU's dequantized
//! weights. The router's top-k over 512 logits and the Q6_K vocabulary projection are checked too.
#![cfg(feature = "rocm")]

use hanzo_ml::quantized::{ggml_file::qtensor_from_ggml, GgmlDType, QMatMul};
use hanzo_ml::{DType, Device, Module, Result, RocmQuantType, Tensor};

/// Deterministic values: an LCG, so a failure reproduces without a seed file.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn byte(&mut self) -> u8 {
        self.next() as u8
    }
    fn unit(&mut self) -> f32 {
        (self.next() % 20001) as f32 / 10000.0 - 1.0
    }
}

/// Random blocks of `dtype` for `elems` weights, with each block's f16 scale set to a finite
/// value so the comparison is about the decode, not NaN.
fn blocks(dtype: GgmlDType, elems: usize, rng: &mut Lcg) -> Vec<u8> {
    let qt = RocmQuantType::from_ggml(dtype).expect("wired");
    let (size, per) = (qt.block_bytes(), qt.block_elems());
    // Where each type keeps its f16 super-scale: last for Q6_K, first for the rest.
    let at = if dtype == GgmlDType::Q6K { size - 2 } else { 0 };
    let mut raw = Vec::with_capacity(elems / per * size);
    for _ in 0..elems / per {
        let start = raw.len();
        raw.extend((0..size).map(|_| rng.byte()));
        let d = half::f16::from_f32(0.004 + 0.004 * rng.unit().abs()).to_bits();
        raw[start + at..start + at + 2].copy_from_slice(&d.to_le_bytes());
    }
    raw
}

/// Largest |gpu - cpu| over the largest |cpu|.
fn rel(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12);
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0f32, f32::max)
        / scale
}

/// One bank `[experts, n, k]` of `dtype`, `t` tokens, top-`topk` ids spread over every expert.
/// `per_slot`: down's input is one row per routed slot; gate/up share one row per token.
#[allow(clippy::too_many_arguments)]
fn moe_case(
    gpu: &Device,
    dtype: GgmlDType,
    experts: usize,
    n: usize,
    k: usize,
    t: usize,
    topk: usize,
    per_slot: bool,
    act: DType,
) -> Result<f32> {
    let mut rng = Lcg(0x5eed ^ (dtype as u64) << 8 ^ (n * k) as u64);
    let raw = blocks(dtype, experts * n * k, &mut rng);
    let bank = qtensor_from_ggml(dtype, &raw, vec![experts, n, k], &Device::Cpu)?
        .dequantize(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let rows = if per_slot { t * topk } else { t };
    let x: Vec<f32> = (0..rows * k).map(|_| rng.unit()).collect();
    // Distinct experts per token, reaching past 255.
    let ids: Vec<u32> = (0..t * topk)
        .map(|i| ((i / topk * 97 + (i % topk) * 53 + 11) % experts) as u32)
        .collect();

    let mut want = vec![0f32; t * topk * n];
    for slot in 0..t * topk {
        let e = ids[slot] as usize;
        let xr = if per_slot { slot } else { slot / topk };
        for r in 0..n {
            let w = &bank[(e * n + r) * k..(e * n + r + 1) * k];
            want[slot * n + r] = w
                .iter()
                .zip(&x[xr * k..(xr + 1) * k])
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum::<f64>() as f32;
        }
    }

    let q = qtensor_from_ggml(dtype, &raw, vec![experts, n, k], gpu)?;
    let shape = if per_slot { (t, topk, k) } else { (t, 1, k) };
    let xg = Tensor::from_vec(x, shape, gpu)?.to_dtype(act)?;
    let idg = Tensor::from_vec(ids, (t, topk), gpu)?;
    let got = q
        .indexed_moe_forward(&xg, &idg)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    Ok(rel(&got, &want))
}

#[test]
fn routed_experts_match_the_cpu_at_model_shapes() -> Result<()> {
    let gpu = Device::new_rocm(0)?;
    let mut worst = 0f32;
    // (dtype, experts, n, k, per_slot): gate/up then down, each with all 512 experts at a narrow
    // width (ids past 255) and with a few experts at the full width.
    let cases = [
        (GgmlDType::IQ3_S, 512, 32, 2560, false),
        (GgmlDType::IQ3_S, 16, 640, 2560, false),
        (GgmlDType::IQ4_XS, 16, 640, 2560, false),
        (GgmlDType::IQ4_NL, 512, 64, 640, true),
        (GgmlDType::IQ4_NL, 16, 2560, 640, true),
        (GgmlDType::Q8_0, 16, 2560, 640, true),
    ];
    for (dtype, experts, n, k, per_slot) in cases {
        for t in [1usize, 5] {
            for act in [DType::F16, DType::F32] {
                let e = moe_case(&gpu, dtype, experts, n, k, t, 10, per_slot, act)?;
                println!("{dtype:?} E={experts} n={n} k={k} t={t} {act:?}: rel {e:.2e}");
                worst = worst.max(e);
            }
        }
    }
    // f16 activations and the int8 paths' q8 activations land near 1e-3 of the largest output.
    assert!(
        worst < 1e-2,
        "routed experts off by {worst} of the largest output"
    );
    Ok(())
}

#[test]
fn top_k_of_512_router_logits_matches_the_cpu() -> Result<()> {
    let gpu = Device::new_rocm(0)?;
    let mut rng = Lcg(0x7009);
    let (t, e) = (7usize, 512usize);
    let logits: Vec<f32> = (0..t * e).map(|_| rng.unit() * 4.0).collect();
    let sort = |d: &Device| -> Result<Vec<u32>> {
        let (_, idx) = Tensor::from_vec(logits.clone(), (t, e), d)?.sort_last_dim(false)?;
        idx.narrow(1, 0, 10)?.flatten_all()?.to_vec1::<u32>()
    };
    assert_eq!(sort(&gpu)?, sort(&Device::Cpu)?);
    Ok(())
}

#[test]
fn q6k_vocabulary_projection_matches_the_cpu() -> Result<()> {
    let gpu = Device::new_rocm(0)?;
    let mut rng = Lcg(0x6b);
    // Rows as the vocabulary, k as the hidden width.
    let (rows, k) = (248_320usize, 2560usize);
    let raw = blocks(GgmlDType::Q6K, rows * k, &mut rng);
    let x: Vec<f32> = (0..k).map(|_| rng.unit()).collect();
    let reference = qtensor_from_ggml(GgmlDType::Q6K, &raw, vec![rows, k], &Device::Cpu)?
        .dequantize(&Device::Cpu)?;
    let want = Tensor::from_vec(x.clone(), (1, k), &Device::Cpu)?
        .matmul(&reference.t()?)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let head = QMatMul::from_qtensor(qtensor_from_ggml(
        GgmlDType::Q6K,
        &raw,
        vec![rows, k],
        &gpu,
    )?)?;
    for act in [DType::F16, DType::F32] {
        let got = head
            .forward(&Tensor::from_vec(x.clone(), (1, k), &gpu)?.to_dtype(act)?)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let e = rel(&got, &want);
        println!("Q6_K head {act:?}: rel {e:.2e}");
        assert!(e < 1e-2, "Q6_K head {act:?} off by {e}");
    }
    Ok(())
}

#[test]
fn an_f32_router_weight_keeps_its_precision() -> Result<()> {
    let gpu = Device::new_rocm(0)?;
    let mut rng = Lcg(0x60a7);
    let (experts, k) = (512usize, 2560usize);
    let w: Vec<f32> = (0..experts * k).map(|_| rng.unit() * 0.05).collect();
    let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    let router = QMatMul::from_qtensor(qtensor_from_ggml(
        GgmlDType::F32,
        &bytes,
        vec![experts, k],
        &gpu,
    )?)?;
    let reference = Tensor::from_vec(w, (experts, k), &Device::Cpu)?;
    // A prompt and a decode step, as the router sees them: f32 activations.
    for t in [5usize, 1] {
        let x: Vec<f32> = (0..t * k).map(|_| rng.unit()).collect();
        let want = Tensor::from_vec(x.clone(), (t, k), &Device::Cpu)?
            .matmul(&reference.t()?)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let got = router
            .forward(&Tensor::from_vec(x, (t, k), &gpu)?)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let e = rel(&got, &want);
        println!("f32 router t={t}: rel {e:.2e}");
        assert!(e < 1e-5, "f32 router t={t} off by {e}");
    }
    Ok(())
}
