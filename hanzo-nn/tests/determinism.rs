//! The CUDA backward and the gradient norm are the same bits run after run, at sizes where many
//! blocks feed one sum: packed attention's dQ over key tiles, LayerNorm's dγ and dβ over row
//! blocks, the gradient norm over thread blocks.
#![cfg(feature = "cuda")]

use hanzo_ml::{DType, Device, Result, Tensor, Var};
use hanzo_nn::attention::packed;
use hanzo_nn::optim::grad_norm;
use hanzo_nn::{LayerNorm, Module};

const RUNS: usize = 4;

fn bits(t: &Tensor) -> Result<Vec<u32>> {
    Ok(t.to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f32::to_bits)
        .collect())
}

/// Elements of `a` and `b` whose bits differ.
fn differ(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

fn tables(p: usize, dim: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
    let half = dim / 2;
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for pos in 0..p {
        for i in 0..half {
            let f = pos as f64 / 10000f64.powf(2.0 * i as f64 / dim as f64);
            cos.push(f.cos() as f32);
            sin.push(f.sin() as f32);
        }
    }
    let t = |v: Vec<f32>| Tensor::from_vec(v, (p, half), dev)?.to_dtype(DType::BF16);
    Ok((t(cos)?, t(sin)?))
}

#[test]
fn packed_attention_backward_is_the_same_bits() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let heads = 12;
    let lens: Vec<usize> = (0..16).map(|i| 300 + (i * 157) % 700).collect();
    let t: usize = lens.iter().sum();
    let hd = heads * 64;
    let (cos, sin) = tables(1024, 64, &dev)?;
    let qkv = Tensor::randn(0f32, 1.0, (t, 3 * hd), &dev)?.to_dtype(DType::BF16)?;
    let w = Tensor::randn(0f32, 1.0, (t, hd), &dev)?;
    for window in [None, Some(64)] {
        let run = || -> Result<Vec<u32>> {
            let x = Var::from_tensor(&qkv)?;
            let y = packed(x.as_tensor(), &lens, heads, window, &cos, &sin, 0.125)?;
            let g = (y.to_dtype(DType::F32)? * &w)?.sum_all()?.backward()?;
            bits(g.get(x.as_tensor()).expect("a gradient"))
        };
        let first = run()?;
        for n in 1..RUNS {
            let d = differ(&first, &run()?);
            assert_eq!(
                d,
                0,
                "window {window:?}, run {n}: {d} of {} gradient bits differ",
                first.len()
            );
        }
    }
    Ok(())
}

#[test]
fn layer_norm_backward_is_the_same_bits() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let (n, d) = (16384, 768);
    let x = Tensor::randn(0f32, 1.0, (n, d), &dev)?.to_dtype(DType::BF16)?;
    let dy = Tensor::randn(0f32, 1.0, (n, d), &dev)?;
    let alpha = Var::from_tensor(&Tensor::randn(1f32, 0.1, d, &dev)?.to_dtype(DType::BF16)?)?;
    let beta = Var::from_tensor(&Tensor::randn(0f32, 0.1, d, &dev)?.to_dtype(DType::BF16)?)?;
    let run = || -> Result<(Vec<u32>, Vec<u32>)> {
        let ln = LayerNorm::new(alpha.as_tensor().clone(), beta.as_tensor().clone(), 1e-5);
        let y = ln.forward(&x)?;
        let g = (y.to_dtype(DType::F32)? * &dy)?.sum_all()?.backward()?;
        Ok((
            bits(g.get(alpha.as_tensor()).expect("dγ"))?,
            bits(g.get(beta.as_tensor()).expect("dβ"))?,
        ))
    };
    let first = run()?;
    for n in 1..RUNS {
        let next = run()?;
        let (a, b) = (differ(&first.0, &next.0), differ(&first.1, &next.1));
        assert_eq!((a, b), (0, 0), "run {n}: dγ {a}, dβ {b} of {d} bits differ");
    }
    Ok(())
}

#[test]
fn gradient_norm_is_the_same_bits() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let vars: Vec<Var> = [1 << 24, 3 << 20, 4097]
        .iter()
        .map(|&n| Var::from_tensor(&Tensor::zeros(n, DType::F32, &dev)?))
        .collect::<Result<_>>()?;
    let loss = vars
        .iter()
        .map(|v| {
            let c = Tensor::randn(0f32, 1.0, v.elem_count(), &dev)?;
            (v.as_tensor() * c)?.sum_all()
        })
        .collect::<Result<Vec<_>>>()?;
    let total = loss
        .iter()
        .skip(1)
        .try_fold(loss[0].clone(), |a, b| a + b)?;
    let grads = total.backward()?;
    let first = grad_norm(&grads, &vars)?;
    for n in 1..RUNS {
        let next = grad_norm(&grads, &vars)?;
        assert_eq!(
            first.to_bits(),
            next.to_bits(),
            "run {n}: {first} against {next}"
        );
    }
    Ok(())
}
