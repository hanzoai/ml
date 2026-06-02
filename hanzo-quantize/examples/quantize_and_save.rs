//! Demonstrate: load an f32 tensor -> quantize a base to Q4_K -> compute
//! BitDelta of residual -> save both to disk.
//!
//! Run:
//!
//! ```bash
//! cargo run -p hanzo-quantize --example quantize_and_save --release
//! ```

use anyhow::Result;
use candle_core::{quantized::GgmlDType, DType, Device, Tensor};
use hanzo_quantize::{
    ggml_bridge,
    storage::{write_adapter, Adapter},
};

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let rows = 1024;
    let cols = 1024;

    // Synthesize a random-looking f32 tensor.
    let v: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32 * 0.0123).sin() + 0.5 * (i as f32 * 0.0231).cos())
        .collect();
    let full = Tensor::from_vec(v, (rows, cols), &dev)?;

    // Split: Q4_K base + BitDelta residual.
    let (base, adapter) = ggml_bridge::split(&full, GgmlDType::Q4K)?;

    // Save adapter to /tmp.
    let path = std::env::temp_dir().join("hzquantize_demo_adapter.bin");
    write_adapter(&path, &Adapter::BitDelta(adapter.clone()))?;

    // Sizes.
    let base_bytes = base.storage_size_in_bytes();
    let adapter_bytes = adapter.size_bytes();
    let bf16_bytes = ggml_bridge::full_size_bytes(&[rows, cols], DType::BF16);
    let f32_bytes = ggml_bridge::full_size_bytes(&[rows, cols], DType::F32);

    println!("Shape: [{rows}, {cols}]  numel = {}", rows * cols);
    println!("Q4_K base:        {:>9} B", base_bytes);
    println!("BitDelta adapter: {:>9} B   (saved -> {})", adapter_bytes, path.display());
    println!("(base + adapter): {:>9} B", base_bytes + adapter_bytes);
    println!("bf16 raw:         {:>9} B", bf16_bytes);
    println!("f32 raw:          {:>9} B", f32_bytes);
    println!("ratio vs bf16: {:.2}x", bf16_bytes as f32 / (base_bytes + adapter_bytes) as f32);
    println!("ratio vs f32:  {:.2}x", f32_bytes as f32 / (base_bytes + adapter_bytes) as f32);
    println!("BitDelta self-ratio (vs raw f32 delta): {:.1}x", adapter.compression_ratio());

    // Sanity: also verify merging the adapter back doesn't blow up.
    let merged = ggml_bridge::merge(&base, &adapter)?;
    let dq_dtype: GgmlDType = merged.dtype();
    println!("Merged QTensor dtype: {:?}", dq_dtype);

    // Round-trip QTensor back to f32 just to prove it works end-to-end.
    let _ = merged.dequantize(&dev)?;

    Ok(())
}
