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
        yarn: None,
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

/// A 2-D mask that lets every query read every real key, at each token's index, is the padded
/// pass bit for bit.
#[test]
fn a_full_mask_is_the_padded_pass() -> Result<()> {
    let dev = Device::Cpu;
    let (bert, _vm) = backbone(&config(), &mut Normal(3), &dev)?;
    let (ids, mask) = rows(&dev)?;
    let (b, l) = ids.dims2()?;
    let pos = Tensor::arange(0u32, l as u32, &dev)?
        .unsqueeze(0)?
        .broadcast_as((b, l))?
        .contiguous()?;
    let allow = mask.unsqueeze(1)?.broadcast_as((b, l, l))?.contiguous()?;
    let keep = mask.to_dtype(DType::F32)?.unsqueeze(2)?;
    let want = bert.forward(&ids, &mask)?.broadcast_mul(&keep)?;
    let got = bert
        .forward_masked(&ids, &pos, &allow)?
        .broadcast_mul(&keep)?;
    assert_eq!(
        want.flatten_all()?.to_vec1::<f32>()?,
        got.flatten_all()?.to_vec1::<f32>()?
    );
    Ok(())
}

/// A shared prefix and two segments that start at the same position, each reading the prefix
/// and itself: every token's state is the same whichever segment comes first, and a segment's
/// tokens move neither the prefix nor the other segment.
#[test]
fn segments_apart_are_alike_in_any_order() -> Result<()> {
    let dev = Device::Cpu;
    let (bert, _vm) = backbone(&config(), &mut Normal(4), &dev)?;
    // (token, position, segment): the prefix is segment 0
    let prefix = [(5u32, 0u32, 0u32), (6, 1, 0), (7, 2, 0), (8, 3, 0)];
    let a = [(9u32, 4u32, 1u32), (10, 5, 1), (11, 6, 1)];
    let b = |x: u32| [(x, 4u32, 2u32), (x + 1, 5, 2)];
    let run = |parts: &[&[(u32, u32, u32)]]| -> Result<Vec<Vec<f32>>> {
        let toks: Vec<(u32, u32, u32)> = parts.concat();
        let l = toks.len();
        let ids = Tensor::from_vec(toks.iter().map(|t| t.0).collect(), (1, l), &dev)?;
        let pos = Tensor::from_vec(toks.iter().map(|t| t.1).collect(), (1, l), &dev)?;
        let allow: Vec<u32> = toks
            .iter()
            .flat_map(|q| toks.iter().map(move |k| u32::from(k.2 == 0 || k.2 == q.2)))
            .collect();
        let allow = Tensor::from_vec(allow, (1, l, l), &dev)?;
        let h = bert.forward_masked(&ids, &pos, &allow)?.squeeze(0)?;
        let rows = h.to_vec2::<f32>()?;
        // each token's state by (segment, position)
        let mut out: Vec<(u32, u32, Vec<f32>)> =
            toks.iter().zip(rows).map(|(t, r)| (t.2, t.1, r)).collect();
        out.sort_by_key(|o| (o.0, o.1));
        Ok(out.into_iter().map(|o| o.2).collect())
    };
    let close = |x: &[Vec<f32>], y: &[Vec<f32>], n: usize| {
        for (u, v) in x[..n].iter().zip(&y[..n]) {
            for (p, q) in u.iter().zip(v) {
                assert!((p - q).abs() <= 1e-5 * (1.0 + p.abs()), "{p} against {q}");
            }
        }
    };
    let first = run(&[&prefix, &a, &b(12)])?;
    let second = run(&[&prefix, &b(12), &a])?;
    close(&first, &second, first.len());
    // another second segment: the prefix and the first segment stay where they were
    let other = run(&[&prefix, &a, &b(20)])?;
    close(&first, &other, prefix.len() + a.len());
    let moved = first[prefix.len() + a.len()]
        .iter()
        .zip(&other[prefix.len() + a.len()])
        .map(|(p, q)| (p - q).abs())
        .fold(0f32, f32::max);
    assert!(moved > 1e-3, "{moved}");
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

/// Right-padded rows of `seqs`' tokens and their mask.
fn padded(seqs: &[Vec<u32>], dev: &Device) -> Result<(Tensor, Tensor)> {
    let l = seqs.iter().map(Vec::len).max().unwrap_or(0);
    let ids: Vec<u32> = seqs
        .iter()
        .flat_map(|s| s.iter().copied().chain(std::iter::repeat_n(0, l - s.len())))
        .collect();
    let mask: Vec<u32> = seqs
        .iter()
        .flat_map(|s| (0..l).map(move |t| u32::from(t < s.len())))
        .collect();
    Ok((
        Tensor::from_vec(ids, (seqs.len(), l), dev)?,
        Tensor::from_vec(mask, (seqs.len(), l), dev)?,
    ))
}

/// `n` tokens of sequence `r`.
fn tokens(r: usize, n: usize) -> Vec<u32> {
    (0..n).map(|t| 1 + ((r * 13 + t * 7) % 49) as u32).collect()
}

/// Row `r` of `h [B, L, d]`, its first `n` positions, as bits.
fn row(h: &Tensor, r: usize, n: usize) -> Result<Vec<f32>> {
    h.get(r)?.narrow(0, 0, n)?.flatten_all()?.to_vec1::<f32>()
}

fn yarn(s: f64) -> Config {
    Config {
        yarn: Some(s),
        ..config()
    }
}

/// Within the trained positions a backbone with YaRN is the backbone without it, bit for bit,
/// on every pass: padded, packed (F32 and Q8_0 projections) and at given positions.
#[test]
fn within_the_trained_positions_yarn_changes_nothing() -> Result<()> {
    let dev = Device::Cpu;
    let (plain, vm) = backbone(&config(), &mut Normal(5), &dev)?;
    let long = ModernBert::new(VarBuilder::from_varmap(&vm, DType::F32, &dev), &yarn(4.0))?;
    // 64 is max_position_embeddings
    let seqs = [tokens(0, 64), tokens(1, 40), tokens(2, 3)];
    let (ids, mask) = padded(&seqs, &dev)?;
    let bits = |t: Tensor| t.flatten_all()?.to_vec1::<f32>();
    assert_eq!(
        bits(plain.forward(&ids, &mask)?)?,
        bits(long.forward(&ids, &mask)?)?
    );
    for dtype in [GgmlDType::F32, GgmlDType::Q8_0] {
        let (mut p, mut l) = (plain.clone(), long.clone());
        p.quantize(dtype)?;
        l.quantize(dtype)?;
        assert!(l.packs());
        assert_eq!(
            bits(p.forward(&ids, &mask)?)?,
            bits(l.forward(&ids, &mask)?)?
        );
    }
    let (b, n) = ids.dims2()?;
    let pos = Tensor::arange(0u32, n as u32, &dev)?
        .unsqueeze(0)?
        .broadcast_as((b, n))?
        .contiguous()?;
    let allow = mask.unsqueeze(1)?.broadcast_as((b, n, n))?.contiguous()?;
    assert_eq!(
        bits(plain.forward_masked(&ids, &pos, &allow)?)?,
        bits(long.forward_masked(&ids, &pos, &allow)?)?
    );
    Ok(())
}

/// Past the trained positions a sequence reads YaRN's tables: refused without them and past
/// them; the packed pass is the padded pass and the masked pass; a short sequence beside a long
/// one reads the trained tables, bit for bit as alone in the packed pass; and the stretch moves
/// the long one only.
#[test]
fn past_the_trained_positions_a_sequence_reads_yarn() -> Result<()> {
    let dev = Device::Cpu;
    let (bert, vm) = backbone(&yarn(4.0), &mut Normal(6), &dev)?;
    let at = |cfg: &Config| ModernBert::new(VarBuilder::from_varmap(&vm, DType::F32, &dev), cfg);
    let plain = at(&config())?;
    // 200 tokens: past the 64 trained, within the 256 stretched
    let seqs = [tokens(0, 200), tokens(1, 12)];
    let (ids, mask) = padded(&seqs, &dev)?;
    assert!(plain.forward(&ids, &mask).is_err());
    let mut q = plain.clone();
    q.quantize(GgmlDType::F32)?;
    assert!(q.forward(&ids, &mask).is_err());

    let keep = mask.to_dtype(DType::F32)?.unsqueeze(2)?;
    let want = bert.forward(&ids, &mask)?.broadcast_mul(&keep)?;
    let scale = want.abs()?.max_all()?.to_scalar::<f32>()?;
    let mut packs = bert.clone();
    packs.quantize(GgmlDType::F32)?;
    let got = packs.forward(&ids, &mask)?;
    let gap = (&got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(gap <= 1e-5 * scale, "packed {gap} against {scale}");
    let (b, n) = ids.dims2()?;
    let pos = Tensor::arange(0u32, n as u32, &dev)?
        .unsqueeze(0)?
        .broadcast_as((b, n))?
        .contiguous()?;
    let allow = mask.unsqueeze(1)?.broadcast_as((b, n, n))?.contiguous()?;
    let masked = bert
        .forward_masked(&ids, &pos, &allow)?
        .broadcast_mul(&keep)?;
    assert_eq!(
        want.flatten_all()?.to_vec1::<f32>()?,
        masked.flatten_all()?.to_vec1::<f32>()?
    );

    // the short sequence alone, packed, with and without YaRN: the same bits as beside the long one
    let (one, one_mask) = padded(&seqs[1..], &dev)?;
    let mut flat = plain.clone();
    flat.quantize(GgmlDType::F32)?;
    let alone = row(&packs.forward(&one, &one_mask)?, 0, 12)?;
    assert_eq!(alone, row(&got, 1, 12)?);
    assert_eq!(alone, row(&flat.forward(&one, &one_mask)?, 0, 12)?);
    // padded, beside it: the trained tables to rounding
    let near = row(&want, 1, 12)?;
    let alone = row(&bert.forward(&one, &one_mask)?, 0, 12)?;
    let worst = near
        .iter()
        .zip(&alone)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(worst <= 1e-5 * scale, "{worst}");

    // another stretch moves the long sequence and not the short
    let mut wider = at(&yarn(8.0))?;
    wider.quantize(GgmlDType::F32)?;
    let other = wider.forward(&ids, &mask)?;
    assert_eq!(row(&other, 1, 12)?, row(&got, 1, 12)?);
    let moved = (other.get(0)? - got.get(0)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(moved > 1e-3 * scale, "{moved}");
    // and past the stretched positions the sequence is refused
    let (far, far_mask) = padded(&[tokens(0, 257)], &dev)?;
    assert!(packs.forward(&far, &far_mask).is_err());
    Ok(())
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

/// A long sequence beside a short one: the packed bf16 pass on CUDA against the padded F32 pass
/// on the CPU, no further from it than the padded bf16 pass on the CPU.
#[cfg(feature = "cuda")]
#[test]
fn a_long_sequence_packs_on_cuda() -> Result<()> {
    let cfg = Config {
        hidden_size: 128,
        intermediate_size: 96,
        ..yarn(4.0)
    };
    let cpu = Device::Cpu;
    let cuda = Device::new_cuda(0)?;
    let (_, vm) = backbone(&cfg, &mut Normal(12), &cpu)?;
    let seqs = [tokens(0, 200), tokens(1, 12)];
    let run = |dtype: DType, dev: &Device| -> Result<(bool, Tensor)> {
        let vb = VarBuilder::from_tensors(
            vm.data()
                .lock()
                .expect("varmap lock")
                .iter()
                .map(|(n, v)| Ok((n.clone(), v.as_tensor().to_dtype(dtype)?.to_device(dev)?)))
                .collect::<Result<std::collections::HashMap<_, _>>>()?,
            dtype,
            dev,
        );
        let bert = ModernBert::new(vb, &cfg)?;
        let (ids, mask) = padded(&seqs, dev)?;
        let keep = mask.to_dtype(DType::F32)?.unsqueeze(2)?;
        let y = bert
            .forward(&ids, &mask)?
            .to_dtype(DType::F32)?
            .broadcast_mul(&keep)?;
        Ok((bert.packs(), y.to_device(&cpu)?))
    };
    let (_, exact) = run(DType::F32, &cpu)?;
    let (_, unfused) = run(DType::BF16, &cpu)?;
    let (packs, fused) = run(DType::BF16, &cuda)?;
    assert!(packs);
    let ((fm, ff), (um, uf)) = (gap(&fused, &exact)?, gap(&unfused, &exact)?);
    println!("long against F32: packed {fm:.2e} / {ff:.2e}, padded bf16 {um:.2e} / {uf:.2e}");
    assert!(ff <= 1.5 * uf + 2e-3, "packed {ff}, padded bf16 {uf}");
    Ok(())
}
