//! ModernBERT's CPU inference pass against its training pass, on a tiny random backbone.

use hanzo_ml::quantized::GgmlDType;
use hanzo_ml::{DType, Device, Result, Tensor};
use hanzo_nn::{VarBuilder, VarMap};
use hanzo_transformers::models::modernbert::{Config, ModernBert};

fn config() -> Config {
    Config {
        vocab_size: 50,
        hidden_size: 64,
        num_hidden_layers: 3,
        num_attention_heads: 2,
        intermediate_size: 64,
        max_position_embeddings: 64,
        layer_norm_eps: 1e-5,
        pad_token_id: 0,
        global_attn_every_n_layers: 3,
        global_rope_theta: 160000.,
        local_attention: 8,
        local_rope_theta: 10000.,
        classifier_config: None,
    }
}

/// Standard normals from a fixed seed.
struct Normal(u64);

impl Normal {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn tensor(&mut self, shape: &[usize], std: f64, dev: &Device) -> Result<Tensor> {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n)
            .map(|_| {
                let (u, v) = (self.uniform().max(1e-12), self.uniform());
                ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos() * std) as f32
            })
            .collect();
        Tensor::from_vec(v, shape, dev)
    }
}

/// A backbone whose weight matrices are drawn from `rng`.
fn backbone(cfg: &Config, rng: &mut Normal, dev: &Device) -> Result<(ModernBert, VarMap)> {
    let vm = VarMap::new();
    let bert = ModernBert::new(VarBuilder::from_varmap(&vm, DType::F32, dev), cfg)?;
    let data = vm.data().lock().expect("varmap lock");
    let mut names: Vec<&String> = data.keys().collect();
    names.sort();
    for n in names {
        let t = &data[n];
        if let [_, fan_in] = t.dims() {
            t.set(&rng.tensor(t.dims(), (2.0 / *fan_in as f64).sqrt(), dev)?)?;
        }
    }
    drop(data);
    Ok((bert, vm))
}

/// Rows of 12, 7 and 3 tokens, right-padded, and their mask.
fn rows(dev: &Device) -> Result<(Tensor, Tensor)> {
    let lens = [12usize, 7, 3];
    let ids: Vec<u32> = lens
        .iter()
        .enumerate()
        .flat_map(|(r, &n)| {
            (0..12).map(move |t| {
                if t < n {
                    1 + ((r * 13 + t * 7) % 49) as u32
                } else {
                    0
                }
            })
        })
        .collect();
    let mask: Vec<u32> = lens
        .iter()
        .flat_map(|&n| (0..12).map(move |t| u32::from(t < n)))
        .collect();
    Ok((
        Tensor::from_vec(ids, (3, 12), dev)?,
        Tensor::from_vec(mask, (3, 12), dev)?,
    ))
}

/// The packed pass gives the padded pass's real positions and zero pads.
#[test]
fn the_packed_pass_is_the_padded_pass() -> Result<()> {
    let dev = Device::Cpu;
    let (bert, _vm) = backbone(&config(), &mut Normal(7), &dev)?;
    let (ids, mask) = rows(&dev)?;
    let keep = mask.to_dtype(DType::F32)?.unsqueeze(2)?;
    let want = bert.forward(&ids, &mask)?.broadcast_mul(&keep)?;
    let scale = want.abs()?.max_all()?.to_scalar::<f32>()?;
    for (dtype, tol) in [(GgmlDType::F32, 1e-5f32), (GgmlDType::Q8_0, 0.08)] {
        let mut q = bert.clone();
        q.quantize(dtype)?;
        let got = q.forward(&ids, &mask)?;
        let pads = got
            .broadcast_mul(&keep.affine(-1.0, 1.0)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert_eq!(pads, 0.0);
        let gap = (got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(gap <= tol * scale, "{dtype:?}: {gap} against {scale}");
    }
    Ok(())
}

/// Output and every parameter's gradient of `Σ y ⊙ w` for a backbone of `named` parameters,
/// in `dtype` on `dev`.
#[allow(clippy::type_complexity)]
fn pass(
    cfg: &Config,
    named: &[(String, hanzo_ml::Var)],
    dtype: DType,
    dev: &Device,
    w: &Tensor,
) -> Result<(bool, Tensor, Vec<Option<Tensor>>)> {
    use hanzo_ml::Var;
    let vm = VarMap::new();
    let vars: Vec<Var> = {
        let mut data = vm.data().lock().expect("varmap lock");
        named
            .iter()
            .map(|(n, t)| {
                let v = Var::from_tensor(&t.as_tensor().to_dtype(dtype)?.to_device(dev)?)?;
                data.insert(n.clone(), v.clone());
                Ok(v)
            })
            .collect::<Result<_>>()?
    };
    let bert = ModernBert::new(VarBuilder::from_varmap(&vm, dtype, dev), cfg)?;
    let (ids, mask) = rows(dev)?;
    let keep = mask.to_dtype(DType::F32)?.unsqueeze(2)?;
    let raw = bert.forward(&ids, &mask)?.to_dtype(DType::F32)?;
    let pads = raw
        .broadcast_mul(&keep.affine(-1.0, 1.0)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    if bert.packs() {
        assert_eq!(pads, 0.0);
    }
    let y = raw.broadcast_mul(&keep)?;
    let g = (&y * w.to_device(dev)?)?.sum_all()?.backward()?;
    let grads = vars
        .iter()
        .map(|v| {
            g.get(v.as_tensor())
                .map(|t| t.to_dtype(DType::F32)?.to_device(&Device::Cpu))
                .transpose()
        })
        .collect::<Result<_>>()?;
    Ok((bert.packs(), y.to_device(&Device::Cpu)?, grads))
}

/// `max|a − b| / max|b|` and `‖a − b‖ / ‖b‖`.
fn gap(a: &Tensor, b: &Tensor) -> Result<(f32, f32)> {
    let d = (a - b)?;
    let max = |t: &Tensor| -> Result<f32> { t.abs()?.max_all()?.to_scalar::<f32>() };
    let norm = |t: &Tensor| -> Result<f32> { t.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>() };
    Ok((max(&d)? / max(b)?.max(1e-6), norm(&d)? / norm(b)?.max(1e-6)))
}

/// The packed bf16 pass on CUDA against the padded F32 pass on the CPU, output and every
/// gradient, no further from it than the padded bf16 pass on the CPU.
#[cfg(feature = "cuda")]
#[test]
fn the_cuda_pass_packs() -> Result<()> {
    let cfg = Config {
        hidden_size: 128,
        intermediate_size: 96,
        ..config()
    };
    let cpu = Device::Cpu;
    let cuda = Device::new_cuda(0)?;
    let mut rng = Normal(11);
    let (_, vm) = backbone(&cfg, &mut rng, &cpu)?;
    let named: Vec<(String, hanzo_ml::Var)> = {
        let data = vm.data().lock().expect("varmap lock");
        let mut v: Vec<_> = data.iter().map(|(n, t)| (n.clone(), t.clone())).collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    };
    let w = rng
        .tensor(&[3, 12, 128], 1.0, &cpu)?
        .to_dtype(DType::BF16)?
        .to_dtype(DType::F32)?;
    let (_, exact, eg) = pass(&cfg, &named, DType::F32, &cpu, &w)?;
    let (_, unfused, ug) = pass(&cfg, &named, DType::BF16, &cpu, &w)?;
    let (packs, fused, fg) = pass(&cfg, &named, DType::BF16, &cuda, &w)?;
    assert!(packs);
    let mut rows = vec![("output".to_string(), fused, unfused, exact)];
    for (i, (n, _)) in named.iter().enumerate() {
        if let (Some(f), Some(u), Some(e)) = (&fg[i], &ug[i], &eg[i]) {
            rows.push((n.clone(), f.clone(), u.clone(), e.clone()));
        }
    }
    for (n, f, u, e) in &rows {
        let ((fm, ff), (um, uf)) = (gap(f, e)?, gap(u, e)?);
        println!("{n}: against F32, packed {fm:.2e} / {ff:.2e}, padded bf16 {um:.2e} / {uf:.2e}");
        assert!(ff <= 1.5 * uf + 2e-3, "{n}: packed {ff}, padded bf16 {uf}");
    }
    Ok(())
}
