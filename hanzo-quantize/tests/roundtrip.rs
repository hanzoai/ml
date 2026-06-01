//! Round-trip tests: compress -> decompress -> verify error bounds.

use candle_core::{DType, Device, Tensor};
use hanzo_quantize::{
    bitdelta::BitDeltaAdapter,
    deltaquant::{DeltaQuantAdapter, QuantBits},
    storage::{decode, encode, Adapter},
};

#[test]
fn bitdelta_preserves_signs_per_channel() {
    let dev = Device::Cpu;
    let rows = 32;
    let cols = 64;
    let v: Vec<f32> = (0..rows * cols)
        .map(|i| if i % 3 == 0 { 0.5 } else if i % 3 == 1 { -0.5 } else { 0.0 })
        .collect();
    let full = Tensor::from_vec(v.clone(), (rows, cols), &dev).unwrap();
    let base = Tensor::zeros((rows, cols), DType::F32, &dev).unwrap();

    let a = BitDeltaAdapter::compress_against_full(&full, &base).unwrap();
    assert_eq!(a.header.scales.len(), rows);

    let recon: Vec<f32> = a.decode(&dev).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    // Mean reconstruction error should be modest (within the per-channel
    // scale itself, which is ~0.33 here).
    let mae: f32 = v.iter().zip(&recon).map(|(a, b)| (a - b).abs()).sum::<f32>() / v.len() as f32;
    assert!(mae < 0.4, "mae {mae} too large");
}

#[test]
fn bitdelta_storage_roundtrip_byte_equal() {
    let dev = Device::Cpu;
    let v: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
    let full = Tensor::from_vec(v, (16, 16), &dev).unwrap();
    let base = Tensor::zeros((16, 16), DType::F32, &dev).unwrap();
    let a = BitDeltaAdapter::compress_against_full(&full, &base).unwrap();

    let bytes = encode(&Adapter::BitDelta(a.clone())).unwrap();
    let dec = decode(&bytes).unwrap();
    let Adapter::BitDelta(b) = dec else { panic!() };
    assert_eq!(a.sign_bits, b.sign_bits);
    assert_eq!(a.header.scales, b.header.scales);
    assert_eq!(a.header.numel, b.header.numel);
    assert_eq!(a.header.shape, b.header.shape);
}

#[test]
fn deltaquant_int8_low_error() {
    let dev = Device::Cpu;
    let v: Vec<f32> = (0..512).map(|i| ((i as f32 - 256.0) * 0.001).sin()).collect();
    let t = Tensor::from_vec(v.clone(), 512, &dev).unwrap();
    let zero = Tensor::zeros(512, DType::F32, &dev).unwrap();
    let dq = DeltaQuantAdapter::compress_against_full(&t, &zero, QuantBits::Int8, Some(128))
        .unwrap();
    let back: Vec<f32> = dq.decode(&dev).unwrap().to_vec1().unwrap();
    let max_err =
        v.iter().zip(&back).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    let max_scale = dq.header.scales.iter().cloned().fold(0.0_f32, f32::max);
    // INT8 with 127 levels: error <= 0.5 * scale.
    assert!(max_err <= 0.5 * max_scale + 1e-6, "max_err {max_err}");
}

#[test]
fn scalar_tensor_rejected() {
    // Scalar (rank 0) -> empty error (BitDelta needs at least one dim to
    // pick a scale convention).
    let dev = Device::Cpu;
    let scalar = Tensor::new(0.0_f32, &dev).unwrap();
    let result = BitDeltaAdapter::compress_against_full(&scalar, &scalar);
    assert!(result.is_err());
}
