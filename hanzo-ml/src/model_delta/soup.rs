//! Model / delta "soups": average several checkpoints (or their fine-tune deltas)
//! into a single merged checkpoint.

use super::{fmt_shape, load_checkpoint, save_checkpoint};
use crate::{bail, DType, Result, Tensor};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Resolve the per-model mixing weights, normalising them so they sum to 1.
///
/// `None` yields a uniform average (`1/n` each). Explicit weights must be
/// non-negative, finite, have the same length as `models`, and not all be zero.
fn resolve_weights(n: usize, weights: Option<&[f32]>) -> Result<Vec<f64>> {
    if n == 0 {
        bail!("soup: no models provided");
    }
    let raw: Vec<f64> = match weights {
        None => vec![1.0; n],
        Some(w) => {
            if w.len() != n {
                bail!(
                    "soup: got {} weights for {n} models (lengths must match)",
                    w.len()
                );
            }
            for (i, &wi) in w.iter().enumerate() {
                if !wi.is_finite() {
                    bail!("soup: weight[{i}] = {wi} is not finite");
                }
                if wi < 0.0 {
                    bail!("soup: weight[{i}] = {wi} is negative");
                }
            }
            w.iter().map(|&x| x as f64).collect()
        }
    };
    let sum: f64 = raw.iter().sum();
    if sum <= 0.0 {
        bail!("soup: weights sum to {sum}, must be > 0");
    }
    Ok(raw.into_iter().map(|x| x / sum).collect())
}

/// Accumulate `acc += weight * tensor`, computing in f32 for numerical stability.
///
/// `acc` is lazily initialised to the first weighted contribution so we never
/// need a separately allocated zero buffer or the original dtype up front.
fn accumulate(acc: &mut Option<Tensor>, tensor: &Tensor, weight: f64) -> Result<()> {
    let contribution = tensor.to_dtype(DType::F32)?.affine(weight, 0.0)?;
    *acc = Some(match acc.take() {
        None => contribution,
        Some(prev) => prev.add(&contribution)?,
    });
    Ok(())
}

/// Average matching tensors across `models` into a single merged checkpoint.
///
/// * `weights` — optional per-model mixing weights (any non-negative scale; they
///   are normalised internally). `None` performs a uniform average.
/// * Tensors are matched by name. A key present in the first model but missing in
///   a later one, or a shape mismatch on a shared key, is a hard error — we never
///   silently drop or mis-average weights.
/// * Each merged tensor is cast back to the dtype it had in the first model so the
///   output checkpoint is drop-in compatible with the inputs.
pub fn soup(models: &[PathBuf], weights: Option<&[f32]>, out: &Path) -> Result<()> {
    let norm = resolve_weights(models.len(), weights)?;

    // The first checkpoint defines the reference key set, shapes and dtypes.
    let first = load_checkpoint(&models[0])?;
    let mut ref_shapes: HashMap<String, Vec<usize>> = HashMap::with_capacity(first.len());
    let mut ref_dtypes: HashMap<String, DType> = HashMap::with_capacity(first.len());
    let mut acc: HashMap<String, Option<Tensor>> = HashMap::with_capacity(first.len());
    for (name, tensor) in first.iter() {
        ref_shapes.insert(name.clone(), tensor.dims().to_vec());
        ref_dtypes.insert(name.clone(), tensor.dtype());
        let mut slot = None;
        accumulate(&mut slot, tensor, norm[0])?;
        acc.insert(name.clone(), slot);
    }
    drop(first);

    for (path, &w) in models.iter().zip(norm.iter()).skip(1) {
        let ckpt = load_checkpoint(path)?;
        if ckpt.len() != ref_shapes.len() {
            bail!(
                "soup: {} has {} tensors but {} has {} (key sets must match)",
                path.display(),
                ckpt.len(),
                models[0].display(),
                ref_shapes.len()
            );
        }
        for (name, tensor) in ckpt.iter() {
            let Some(ref_shape) = ref_shapes.get(name) else {
                bail!(
                    "soup: tensor '{name}' in {} is absent from {}",
                    path.display(),
                    models[0].display()
                );
            };
            if tensor.dims() != ref_shape.as_slice() {
                bail!(
                    "soup: tensor '{name}' shape mismatch: {} has {} but {} has {}",
                    models[0].display(),
                    fmt_shape(ref_shape),
                    path.display(),
                    fmt_shape(tensor.dims())
                );
            }
            // Safe: key set sizes match and every key was inserted from model 0.
            let slot = acc.get_mut(name).expect("key validated above");
            accumulate(slot, tensor, w)?;
        }
    }

    let mut merged: HashMap<String, Tensor> = HashMap::with_capacity(acc.len());
    for (name, slot) in acc {
        let tensor = slot.expect("every slot was accumulated at least once");
        let dtype = ref_dtypes[&name];
        merged.insert(name, tensor.to_dtype(dtype)?);
    }
    save_checkpoint(&merged, out)
}

/// Average the fine-tune *deltas* `(ftᵢ − base)` and add the (weighted) mean back
/// onto `base` — i.e. `base + Σ wᵢ (ftᵢ − base)`.
///
/// This is the task-arithmetic / delta-soup variant: with uniform weights it is
/// algebraically identical to averaging `base` with the fine-tunes, but exposing
/// the deltas lets callers scale the combined fine-tune signal (e.g. weights that
/// sum to >1 to amplify, or per-model weights to emphasise one fine-tune).
///
/// * `weights` — optional per-fine-tune weights. Unlike [`soup`] these are *not*
///   renormalised: they are applied directly to each delta, so a uniform request
///   (`None`) uses `1/n` and the result lands at the mean fine-tune.
/// * Only keys present in `base` are updated; a fine-tune key missing from `base`,
///   or a shape mismatch, is an error. Base tensors that no fine-tune touches are
///   passed through unchanged.
pub fn delta_soup(base: &Path, finetunes: &[PathBuf], out: &Path) -> Result<()> {
    if finetunes.is_empty() {
        bail!("delta_soup: no fine-tuned models provided");
    }
    // Uniform 1/n averaging of the deltas (applied directly, not renormalised).
    let weights: Vec<f64> = vec![1.0 / finetunes.len() as f64; finetunes.len()];

    let base_map = load_checkpoint(base)?;
    let mut base_f32: HashMap<String, Tensor> = HashMap::with_capacity(base_map.len());
    let mut delta_acc: HashMap<String, Tensor> = HashMap::with_capacity(base_map.len());
    for (name, tensor) in base_map.iter() {
        base_f32.insert(name.clone(), tensor.to_dtype(DType::F32)?);
    }

    for (path, &w) in finetunes.iter().zip(weights.iter()) {
        let ft = load_checkpoint(path)?;
        for (name, tensor) in ft.iter() {
            let Some(base_t) = base_f32.get(name) else {
                bail!(
                    "delta_soup: tensor '{name}' in {} is absent from base {}",
                    path.display(),
                    base.display()
                );
            };
            if tensor.dims() != base_t.dims() {
                bail!(
                    "delta_soup: tensor '{name}' shape mismatch: base {} has {} but {} has {}",
                    base.display(),
                    fmt_shape(base_t.dims()),
                    path.display(),
                    fmt_shape(tensor.dims())
                );
            }
            let delta = tensor.to_dtype(DType::F32)?.sub(base_t)?;
            let weighted = delta.affine(w, 0.0)?;
            delta_acc
                .entry(name.clone())
                .and_modify(|acc| {
                    // add can only fail on shape mismatch, which we already ruled out.
                    *acc = acc.add(&weighted).expect("validated matching shapes");
                })
                .or_insert(weighted);
        }
    }

    let mut merged: HashMap<String, Tensor> = HashMap::with_capacity(base_map.len());
    for (name, base_t) in base_map.iter() {
        let dtype = base_t.dtype();
        let out_t = match delta_acc.get(name) {
            Some(delta) => base_f32[name].add(delta)?.to_dtype(dtype)?,
            None => base_t.clone(),
        };
        merged.insert(name.clone(), out_t);
    }
    save_checkpoint(&merged, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, Tensor};
    use std::collections::HashMap;

    fn tmp(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("hanzo_soup_{pid}_{nanos}_{name}"));
        p
    }

    fn write(path: &Path, tensors: &[(&str, Tensor)]) {
        let map: HashMap<String, Tensor> = tensors
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        save_checkpoint(&map, path).unwrap();
    }

    fn t(data: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_slice(data, shape, &Device::Cpu).unwrap()
    }

    #[test]
    fn uniform_soup_is_elementwise_mean() {
        let a = tmp("u_a.safetensors");
        let b = tmp("u_b.safetensors");
        let o = tmp("u_o.safetensors");
        write(&a, &[("w", t(&[0.0, 2.0, 4.0, 6.0], &[2, 2]))]);
        write(&b, &[("w", t(&[2.0, 4.0, 6.0, 8.0], &[2, 2]))]);

        soup(&[a.clone(), b.clone()], None, &o).unwrap();

        let merged = load_checkpoint(&o).unwrap();
        let w: Vec<f32> = merged["w"].flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(w, vec![1.0, 3.0, 5.0, 7.0]);

        for p in [a, b, o] {
            let _ = std::fs::remove_file(p);
        }
    }

    #[test]
    fn weighted_soup_normalises_weights() {
        let a = tmp("w_a.safetensors");
        let b = tmp("w_b.safetensors");
        let o = tmp("w_o.safetensors");
        write(&a, &[("w", t(&[0.0, 0.0], &[2]))]);
        write(&b, &[("w", t(&[4.0, 8.0], &[2]))]);

        // weights [1, 3] -> normalised [0.25, 0.75] -> 0.75 * b.
        soup(&[a.clone(), b.clone()], Some(&[1.0, 3.0]), &o).unwrap();

        let merged = load_checkpoint(&o).unwrap();
        let w: Vec<f32> = merged["w"].flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(w, vec![3.0, 6.0]);

        for p in [a, b, o] {
            let _ = std::fs::remove_file(p);
        }
    }

    #[test]
    fn soup_preserves_input_dtype() {
        let a = tmp("d_a.safetensors");
        let b = tmp("d_b.safetensors");
        let o = tmp("d_o.safetensors");
        let ta = t(&[1.0, 3.0], &[2]).to_dtype(DType::BF16).unwrap();
        let tb = t(&[3.0, 5.0], &[2]).to_dtype(DType::BF16).unwrap();
        write(&a, &[("w", ta)]);
        write(&b, &[("w", tb)]);

        soup(&[a.clone(), b.clone()], None, &o).unwrap();

        let merged = load_checkpoint(&o).unwrap();
        assert_eq!(merged["w"].dtype(), DType::BF16);
        let w: Vec<f32> = merged["w"]
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(w, vec![2.0, 4.0]);

        for p in [a, b, o] {
            let _ = std::fs::remove_file(p);
        }
    }

    #[test]
    fn soup_errors_on_shape_mismatch() {
        let a = tmp("s_a.safetensors");
        let b = tmp("s_b.safetensors");
        let o = tmp("s_o.safetensors");
        write(&a, &[("w", t(&[1.0, 2.0], &[2]))]);
        write(&b, &[("w", t(&[1.0, 2.0, 3.0], &[3]))]);

        let err = soup(&[a.clone(), b.clone()], None, &o).unwrap_err();
        assert!(err.to_string().contains("shape mismatch"), "got: {err}");

        for p in [a, b, o] {
            let _ = std::fs::remove_file(p);
        }
    }

    #[test]
    fn soup_errors_on_missing_key() {
        let a = tmp("k_a.safetensors");
        let b = tmp("k_b.safetensors");
        let o = tmp("k_o.safetensors");
        write(&a, &[("w", t(&[1.0], &[1])), ("extra", t(&[2.0], &[1]))]);
        write(&b, &[("w", t(&[3.0], &[1])), ("other", t(&[4.0], &[1]))]);

        let err = soup(&[a.clone(), b.clone()], None, &o).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("absent") || msg.contains("key sets must match"),
            "got: {msg}"
        );

        for p in [a, b, o] {
            let _ = std::fs::remove_file(p);
        }
    }

    #[test]
    fn delta_soup_matches_direct_soup_uniform() {
        // base + mean(ft_i - base) == mean(base, ft...) when weights are uniform
        // and we include the deltas of every fine-tune. Here with one fine-tune,
        // base + (ft - base) == ft exactly.
        let base = tmp("ds_base.safetensors");
        let ft = tmp("ds_ft.safetensors");
        let o = tmp("ds_o.safetensors");
        write(&base, &[("w", t(&[1.0, 2.0, 3.0], &[3]))]);
        write(&ft, &[("w", t(&[4.0, 0.0, 9.0], &[3]))]);

        delta_soup(&base, std::slice::from_ref(&ft), &o).unwrap();

        let merged = load_checkpoint(&o).unwrap();
        let w: Vec<f32> = merged["w"].flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(w, vec![4.0, 0.0, 9.0]);

        for p in [base, ft, o] {
            let _ = std::fs::remove_file(p);
        }
    }

    #[test]
    fn delta_soup_averages_two_finetunes() {
        // base=0, ft1 delta=+2, ft2 delta=+4 -> base + mean = +3.
        let base = tmp("ds2_base.safetensors");
        let ft1 = tmp("ds2_ft1.safetensors");
        let ft2 = tmp("ds2_ft2.safetensors");
        let o = tmp("ds2_o.safetensors");
        write(&base, &[("w", t(&[0.0, 0.0], &[2]))]);
        write(&ft1, &[("w", t(&[2.0, 2.0], &[2]))]);
        write(&ft2, &[("w", t(&[4.0, 4.0], &[2]))]);

        delta_soup(&base, &[ft1.clone(), ft2.clone()], &o).unwrap();

        let merged = load_checkpoint(&o).unwrap();
        let w: Vec<f32> = merged["w"].flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(w, vec![3.0, 3.0]);

        for p in [base, ft1, ft2, o] {
            let _ = std::fs::remove_file(p);
        }
    }

    #[test]
    fn delta_soup_passes_through_untouched_base_tensors() {
        let base = tmp("dp_base.safetensors");
        let ft = tmp("dp_ft.safetensors");
        let o = tmp("dp_o.safetensors");
        write(
            &base,
            &[
                ("w", t(&[1.0, 1.0], &[2])),
                ("frozen", t(&[7.0, 8.0], &[2])),
            ],
        );
        write(&ft, &[("w", t(&[3.0, 3.0], &[2]))]);

        delta_soup(&base, std::slice::from_ref(&ft), &o).unwrap();

        let merged = load_checkpoint(&o).unwrap();
        let frozen: Vec<f32> = merged["frozen"].flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(frozen, vec![7.0, 8.0]);
        let w: Vec<f32> = merged["w"].flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(w, vec![3.0, 3.0]);

        for p in [base, ft, o] {
            let _ = std::fs::remove_file(p);
        }
    }
}
