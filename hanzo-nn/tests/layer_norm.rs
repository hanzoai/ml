#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use hanzo_ml::{test_utils, Device, Tensor};
use hanzo_nn::{LayerNorm, Module};

#[test]
fn layer_norm() -> Result<()> {
    let device = &Device::Cpu;
    let w = Tensor::new(&[3f32], device)?;
    let b = Tensor::new(&[0.5f32], device)?;
    let ln2 = LayerNorm::new(Tensor::cat(&[&w, &w], 0)?, Tensor::cat(&[&b, &b], 0)?, 1e-8);
    let ln3 = LayerNorm::new(
        Tensor::cat(&[&w, &w, &w], 0)?,
        Tensor::cat(&[&b, &b, &b], 0)?,
        1e-8,
    );
    let ln = LayerNorm::new(w, b, 1e-8);
    assert_eq!(ln.eps(), 1e-8);
    assert!(ln.remove_mean());

    let two = Tensor::new(&[[[2f32]]], device)?;
    let res = ln.forward(&two)?.flatten_all()?;
    assert_eq!(res.to_vec1::<f32>()?, [0.5f32]);

    let inp = Tensor::new(&[[[4f32, 0f32]]], device)?;
    let res = ln2.forward(&inp)?;
    assert_eq!(res.to_vec3::<f32>()?, [[[3.5f32, -2.5]]]);

    let inp = Tensor::new(&[[[1f32, 2., 3.], [4., 5., 6.], [9., 8., 7.]]], device)?;
    let res = ln3.forward(&inp)?;
    assert_eq!(
        test_utils::to_vec3_round(&res, 4)?,
        [[
            [-3.1742, 0.5, 4.1742],
            [-3.1742, 0.5, 4.1742],
            [4.1742, 0.5, -3.1742]
        ]]
    );
    let mean = (res.sum_keepdim(2)? / 3.0)?;
    // The average value should be `b`.
    assert_eq!(
        test_utils::to_vec3_round(&mean, 4)?,
        [[[0.5], [0.5], [0.5]]]
    );
    let std = (res.broadcast_sub(&mean)?.sqr()?.sum_keepdim(2)?.sqrt()? / 3.0)?;
    // The standard deviation should be sqrt(`w`).
    assert_eq!(
        test_utils::to_vec3_round(&std, 4)?,
        [[[1.7321], [1.7321], [1.7321]]]
    );

    // Verify that rms_norm sets remove_mean to false.
    let rms = LayerNorm::rms_norm(Tensor::new(&[1f32], device)?, 1e-5);
    assert_eq!(rms.eps(), 1e-5);
    assert!(!rms.remove_mean());

    Ok(())
}

/// A norm loads from whatever its checkpoint holds: `weight` alone when it has no bias,
/// `gamma`/`beta` from an old state dict, and fresh parameters from an empty VarMap.
#[test]
fn layer_norm_loads_each_parameter_naming() -> Result<()> {
    use hanzo_ml::DType;
    use hanzo_nn::{layer_norm, layer_norm_no_bias, VarBuilder, VarMap};
    use std::collections::HashMap;
    let dev = &Device::Cpu;
    let w = Tensor::new(&[2f32, 2.], dev)?;
    let b = Tensor::new(&[1f32, 1.], dev)?;
    let x = Tensor::new(&[[1f32, 3.]], dev)?;

    let only_weight = VarBuilder::from_tensors(
        HashMap::from([("weight".into(), w.clone())]),
        DType::F32,
        dev,
    );
    let ln = layer_norm_no_bias(2, 1e-5, only_weight.clone())?;
    assert!(ln.bias().is_none());
    assert_eq!(
        test_utils::to_vec2_round(&ln.forward(&x)?, 3)?,
        [[-2.0, 2.0]]
    );
    assert!(
        layer_norm(2, 1e-5, only_weight).is_err(),
        "an affine norm needs its bias"
    );

    let old = VarBuilder::from_tensors(
        HashMap::from([("gamma".into(), w), ("beta".into(), b)]),
        DType::F32,
        dev,
    );
    let ln = layer_norm(2, 1e-5, old)?;
    assert_eq!(
        test_utils::to_vec2_round(&ln.forward(&x)?, 3)?,
        [[-1.0, 3.0]]
    );

    let vm = VarMap::new();
    let ln = layer_norm(2, 1e-5, VarBuilder::from_varmap(&vm, DType::F32, dev))?;
    assert_eq!(ln.weight().to_vec1::<f32>()?, [1.0, 1.0]);
    assert_eq!(vm.all_vars().len(), 2);
    Ok(())
}
