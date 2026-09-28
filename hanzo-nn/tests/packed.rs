//! The CUDA packed attention against the unfused composite, outputs and gradients.
#![cfg(feature = "cuda")]

use hanzo_ml::{DType, Device, Result, Tensor, Var};
use hanzo_nn::attention::packed;

const HEADS: usize = 3;
const DIM: usize = 64;

/// Standard normals from a fixed seed.
struct Normal(u64);

impl Normal {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn tensor(&mut self, shape: &[usize], std: f64) -> Result<Tensor> {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n)
            .map(|_| {
                let (u, v) = (self.uniform().max(1e-12), self.uniform());
                ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos() * std) as f32
            })
            .collect();
        Tensor::from_vec(v, shape, &Device::Cpu)
    }
}

/// The composite over every sequence, in the dtype and on the device of `qkv`.
fn composite(
    qkv: &Tensor,
    lens: &[usize],
    window: Option<usize>,
    cos: &Tensor,
    sin: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    let hd = HEADS * DIM;
    let mut outs = Vec::new();
    let mut at = 0;
    for &l in lens {
        let x = qkv.narrow(0, at, l)?;
        let part = |i: usize| -> Result<Tensor> {
            x.narrow(1, i * hd, hd)?
                .reshape((1, l, HEADS, DIM))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = hanzo_nn::rotary_emb::rope(&part(0)?, cos, sin)?;
        let k = hanzo_nn::rotary_emb::rope(&part(1)?, cos, sin)?;
        let v = part(2)?;
        let mut s = (q * scale as f64)?.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        if let Some(w) = window {
            let mask: Vec<f32> = (0..l)
                .flat_map(|i| {
                    (0..l).map(move |j| {
                        if i.abs_diff(j) > w {
                            f32::NEG_INFINITY
                        } else {
                            0.
                        }
                    })
                })
                .collect();
            let mask = Tensor::from_vec(mask, (l, l), qkv.device())?.to_dtype(qkv.dtype())?;
            s = s.broadcast_add(&mask)?;
        }
        let o = hanzo_nn::ops::softmax_last_dim(&s)?
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((l, hd))?;
        outs.push(o);
        at += l;
    }
    Tensor::cat(&outs, 0)
}

fn tables(p: usize) -> Result<(Tensor, Tensor)> {
    let half = DIM / 2;
    let mut cos = Vec::with_capacity(p * half);
    let mut sin = Vec::with_capacity(p * half);
    for pos in 0..p {
        for i in 0..half {
            let f = pos as f64 / 10000f64.powf(2.0 * i as f64 / DIM as f64);
            cos.push(f.cos() as f32);
            sin.push(f.sin() as f32);
        }
    }
    let round = |v: Vec<f32>| -> Result<Tensor> {
        Tensor::from_vec(v, (p, half), &Device::Cpu)?
            .to_dtype(DType::BF16)?
            .to_dtype(DType::F32)
    };
    Ok((round(cos)?, round(sin)?))
}

/// `‖a − b‖ / ‖b‖`, in F32 on the CPU.
fn gap(a: &Tensor, b: &Tensor) -> Result<f32> {
    let f = |t: &Tensor| t.to_dtype(DType::F32)?.to_device(&Device::Cpu);
    let (a, b) = (f(a)?, f(b)?);
    let norm = |t: &Tensor| -> Result<f32> { t.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>() };
    Ok(norm(&(a - &b)?)? / norm(&b)?)
}

/// The output and the gradient of `Σ y ⊙ w` with respect to `qkv`.
fn run(y: impl Fn(&Tensor) -> Result<Tensor>, qkv: &Tensor, w: &Tensor) -> Result<[Tensor; 2]> {
    let x = Var::from_tensor(qkv)?;
    let y = y(x.as_tensor())?;
    let g = (y.to_dtype(DType::F32)? * w)?.sum_all()?.backward()?;
    let dx = g.get(x.as_tensor()).expect("a gradient").clone();
    Ok([y, dx])
}

#[test]
fn packed_is_the_composite() -> Result<()> {
    let cuda = Device::new_cuda(0)?;
    let lens = [5usize, 70, 130, 1, 64, 200];
    let t: usize = lens.iter().sum();
    let hd = HEADS * DIM;
    let scale = (DIM as f32).powf(-0.5);
    let (cos, sin) = tables(256)?;
    let mut rng = Normal(5);
    let bits = |x: Tensor| x.to_dtype(DType::BF16)?.to_dtype(DType::F32);
    let qkv = bits(rng.tensor(&[t, 3 * hd], 0.7)?)?;
    let w = bits(rng.tensor(&[t, hd], 1.0)?)?;
    let half = |x: &Tensor| x.to_dtype(DType::BF16)?.to_device(&cuda);
    let (hcos, hsin, hw) = (half(&cos)?, half(&sin)?, w.to_device(&cuda)?);
    for window in [None, Some(16), Some(64)] {
        let exact = run(|x| composite(x, &lens, window, &cos, &sin, scale), &qkv, &w)?;
        let unfused = run(
            |x| composite(x, &lens, window, &hcos, &hsin, scale),
            &half(&qkv)?,
            &hw,
        )?;
        let fused = run(
            |x| packed(x, &lens, HEADS, window, &hcos, &hsin, scale),
            &half(&qkv)?,
            &hw,
        )?;
        assert_eq!(fused[0].dims(), &[t, hd]);
        for (i, what) in ["output", "gradient"].iter().enumerate() {
            let (f, u) = (gap(&fused[i], &exact[i])?, gap(&unfused[i], &exact[i])?);
            println!("window {window:?} {what}: fused {f:.5}, unfused {u:.5}");
            assert!(f < 1e-2, "window {window:?} {what}: fused off by {f}");
            assert!(f <= u, "window {window:?} {what}: fused {f}, unfused {u}");
        }
    }
    Ok(())
}
