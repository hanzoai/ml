//! Quantization support

#[derive(Debug, Clone)]
pub enum QuantizationType {
    AFQ2,
    AFQ3, 
    AFQ4,
    AFQ6,
    AFQ8,
    GGUF,
    GGML,
    GPTQ,
    AWQ,
    FP8,
}

pub struct QuantizationConfig {
    pub qtype: QuantizationType,
    pub group_size: Option<usize>,
    pub bits: u8,
}