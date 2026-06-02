//! Demonstrate federation flow: 1 base + 4 worker BitDelta adapters,
//! aggregate with trimmed mean, apply on top of base, dequantize for
//! inference.
//!
//! Run:
//!
//! ```bash
//! cargo run -p hanzo-quantize --example federation_apply --release
//! ```

use anyhow::Result;
use candle_core::{quantized::{GgmlDType, QTensor}, Device, Tensor};
use hanzo_quantize::{
    aggregate, ggml_bridge, AggregateMethod, BitDeltaAdapter,
};

fn worker_full(
    base_t: &Tensor,
    seed: u64,
    magnitude: f32,
    dev: &Device,
) -> Result<Tensor> {
    // Synthesize a "fine-tune": base + tiny pseudo-random perturbation.
    let n = base_t.elem_count();
    let mut x = (seed as u32).wrapping_mul(2654435761) ^ 0xDEAD_BEEF;
    let mut rng = || {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        ((x as i32 as f32) / i32::MAX as f32) * magnitude
    };
    let v: Vec<f32> = (0..n).map(|_| rng()).collect();
    let perturb = Tensor::from_vec(v, base_t.dims(), dev)?;
    Ok((base_t + &perturb)?)
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let rows = 256;
    let cols = 256;

    // 1. Create a base tensor and quantize it to Q8_0.
    let base_v: Vec<f32> =
        (0..rows * cols).map(|i| ((i as f32 - 32768.0) * 0.0001).tanh()).collect();
    let base_t = Tensor::from_vec(base_v, (rows, cols), &dev)?;
    let base_q = QTensor::quantize(&base_t, GgmlDType::Q8_0)?;
    let base_dq = base_q.dequantize(&dev)?;
    println!("Base: Q8_0 [{rows}, {cols}], {} bytes", base_q.storage_size_in_bytes());

    // 2. Four "workers" each produce a slightly different fine-tune and ship
    //    a BitDelta adapter.
    let mut adapters: Vec<BitDeltaAdapter> = Vec::new();
    for w in 0..4 {
        let full = worker_full(&base_dq, w as u64, 0.01, &dev)?;
        let a = BitDeltaAdapter::compress(&full, &base_q)?;
        println!(
            "  worker {w}: adapter {} B  (ratio vs raw delta: {:.1}x)",
            a.size_bytes(),
            a.compression_ratio()
        );
        adapters.push(a);
    }

    // 3. Coordinator: decode each adapter back into a delta tensor, aggregate
    //    with trimmed-mean (handles 1 byzantine outlier), apply on base.
    let deltas: Vec<Tensor> = adapters.iter().map(|a| a.decode(&dev).unwrap()).collect();
    let agg_delta = aggregate(AggregateMethod::TrimmedMean { trim: 0.2 }, &deltas)?;

    // Merged inference tensor (f32) — ready to multiply.
    let merged_f32 = (&base_dq + &agg_delta)?;

    // Or: re-compress the aggregated delta and apply to QTensor, getting back
    // a QTensor of the same dtype as base.
    let agg_adapter = BitDeltaAdapter::compress_against_full(
        &(&base_dq + &agg_delta)?,
        &base_dq,
    )?;
    let merged_q = ggml_bridge::merge(&base_q, &agg_adapter)?;

    println!("Aggregated adapter: {} B", agg_adapter.size_bytes());
    println!("Merged QTensor dtype: {:?}", merged_q.dtype());
    println!("Merged f32 shape: {:?}, dtype: {:?}", merged_f32.dims(), merged_f32.dtype());

    Ok(())
}
