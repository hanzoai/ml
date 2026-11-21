//! Test loading Bonsai 2 PQ2_0 layer directly in hanzo-ml.

use hanzo_ml::quantized::gguf_file::Content;
use hanzo_ml::quantized::GgmlDType;
use hanzo_ml::{Device, Module, Tensor};
use std::fs::File;
use std::io::BufReader;

fn main() -> anyhow::Result<()> {
    let model_path = "/home/z/work/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf";
    println!("Opening Bonsai-2 GGUF: {model_path}");
    let file = File::open(model_path)?;
    let mut reader = BufReader::new(file);

    let content = Content::read(&mut reader)?;
    println!("Loaded GGUF header, total tensors: {}", content.tensor_infos.len());

    let device = Device::new_cuda(0)?;
    println!("CUDA Device initialized: {:?}", device);

    // Find first PQ2_0 tensor (e.g. blk.0.attn_q.weight or ffn_gate.weight)
    let target_name = content.tensor_infos.keys()
        .find(|k| content.tensor_infos[*k].ggml_dtype == GgmlDType::PQ2_0)
        .expect("No PQ2_0 tensor found!");

    println!("Found PQ2_0 tensor: {target_name}");
    let info = &content.tensor_infos[target_name];
    println!("Shape: {:?}, DType: {:?}", info.shape, info.ggml_dtype);

    // Load tensor directly to CUDA device
    let qtensor = content.tensor(&mut reader, target_name, &device)?;
    println!("Successfully loaded QTensor to CUDA! Elements: {}", qtensor.shape().elem_count());

    // Test matrix-vector multiplication via QMatMul
    let k = qtensor.shape().dims()[1];
    println!("Testing forward MMVQ pass with input vector [1, {k}]...");
    let input = Tensor::randn(0.0f32, 1.0f32, (1, k), &device)?;
    let qmatmul = hanzo_ml::quantized::QMatMul::from_arc(std::sync::Arc::new(qtensor))?;
    let out = qmatmul.forward(&input)?;
    println!("Forward MMVQ successful! Output shape: {:?}", out.shape());

    Ok(())
}
