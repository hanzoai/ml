//! CPU runtime smoke benchmark, synthetic token IDs only. Never trains or reads evaluation data.
//!
//! Usage:
//!   modernbert_adapter --synthetic output.json
//!   modernbert_adapter config.json weights.safetensors tensor-prefix output.json
use hanzo_ml::quantized::GgmlDType;
use hanzo_ml::{DType, Device, Result, Tensor};
use hanzo_nn::{VarBuilder, VarMap};
use hanzo_transformers::models::modernbert::{
    adapter::{Adapter, Layer, Lora},
    Config, ModernBert,
};
use serde_json::json;
use std::time::Instant;

fn synthetic_backbone(cfg: &Config, dev: &Device) -> Result<(ModernBert, VarMap)> {
    let vm = VarMap::new();
    let model = ModernBert::new(VarBuilder::from_varmap(&vm, DType::F32, dev), cfg)?;
    let data = vm.data().lock().expect("varmap lock");
    let mut names: Vec<&String> = data.keys().collect();
    names.sort();
    let mut state: u64 = 0x517cc1b727220a95;
    for name in names {
        let var = &data[name];
        let n = var.elem_count();
        let scale = match var.dims() {
            [_, fan_in] => (2.0 / *fan_in as f64).sqrt() as f32 * 0.1,
            _ => 0.02,
        };
        let vals: Vec<f32> = (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (((state >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0) * scale
            })
            .collect();
        var.set(&Tensor::from_vec(vals, var.dims(), dev)?)?;
    }
    drop(data);
    Ok((model, vm))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (cfg, mut model, out_path, _vm_guard) = if args.len() == 3 && args[1] == "--synthetic" {
        let cfg = Config {
            vocab_size: 2048,
            hidden_size: 384,
            num_hidden_layers: 12,
            num_attention_heads: 6,
            intermediate_size: 768,
            max_position_embeddings: 2048,
            layer_norm_eps: 1e-5,
            pad_token_id: 0,
            global_attn_every_n_layers: 3,
            global_rope_theta: 160000.,
            local_attention: 128,
            local_rope_theta: 10000.,
            yarn: None,
            classifier_config: None,
        };
        let (model, vm) = synthetic_backbone(&cfg, &Device::Cpu)?;
        (cfg, model, args[2].clone(), Some(vm))
    } else if args.len() == 5 {
        let cfg: Config =
            serde_json::from_slice(&std::fs::read(&args[1])?).map_err(hanzo_ml::Error::wrap)?;
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&args[2]], DType::F32, &Device::Cpu)? };
        let model = ModernBert::new(vb.pp(&args[3]), &cfg)?;
        (cfg, model, args[4].clone(), None)
    } else {
        hanzo_ml::bail!(
            "usage: modernbert_adapter --synthetic output.json | config.json weights.safetensors tensor-prefix output.json"
        );
    };

    model.quantize(GgmlDType::Q8_0)?;
    let d = cfg.hidden_size;
    let factor = |a: usize, b: usize| -> Result<Tensor> {
        Tensor::from_vec(
            (0..a * b)
                .map(|i| (((i * 17 + 11) % 97) as f32 - 48.0) * 0.0002)
                .collect::<Vec<_>>(),
            (a, b),
            &Device::Cpu,
        )
    };
    let r = 4;
    let mut layers = Vec::new();
    for _ in 0..2 {
        layers.push(Layer {
            qkv: Some(Lora::new(factor(r, d)?, factor(3 * d, r)?, 4.0)?),
            output: Some(Lora::new(factor(r, d)?, factor(d, r)?, 4.0)?),
        });
    }
    let first = cfg.num_hidden_layers - 2;
    let bank = [Adapter { first, layers }];
    // Isolate incremental tensor operations: excludes the shared frozen matrix products.
    let mut isolated = Vec::new();
    for length in [32usize, 128, 512, 1024] {
        let x = factor(length, d)?;
        let q = factor(length, 3 * d)?;
        let o = factor(length, d)?;
        let mut us = Vec::new();
        for round in 0..220 {
            let start = Instant::now();
            for layer in &bank[0].layers {
                std::hint::black_box((&q + layer.qkv.as_ref().unwrap().delta(&x)?)?);
                std::hint::black_box((&o + layer.output.as_ref().unwrap().delta(&x)?)?);
            }
            if round >= 20 {
                us.push(start.elapsed().as_secs_f64() * 1e6);
            }
        }
        us.sort_by(f64::total_cmp);
        let cell = json!({
            "tokens": length,
            "mean_us": us.iter().sum::<f64>() / us.len() as f64,
            "p50_us": us[100],
            "p95_us": us[190],
            "samples": 200
        });
        eprintln!("isolated {cell}");
        isolated.push(cell);
    }
    std::fs::write(
        format!("{out_path}.isolated.json"),
        serde_json::to_vec_pretty(&json!({
            "stage": "untrained incremental-kernel diagnostic",
            "layers": 2,
            "rank": 4,
            "hidden_size": d,
            "factor_bytes": bank[0].bytes(),
            "cells": isolated,
            "excludes": ["router", "pooling", "projection splitting/concatenation", "production head", "cache effects"],
            "final_harness": "NOT_READ"
        }))
        .map_err(hanzo_ml::Error::wrap)?,
    )?;

    let mut cells = Vec::new();
    for l in [32usize, 128, 512, 1024] {
        let ids = Tensor::from_vec(
            (0..l)
                .map(|i| (1 + (i * 7) % (cfg.vocab_size - 1)) as u32)
                .collect::<Vec<_>>(),
            (1, l),
            &Device::Cpu,
        )?;
        let mask = Tensor::ones((1, l), DType::U32, &Device::Cpu)?;
        let base = model.forward(&ids, &mask)?;
        let none = model.forward_routed(&ids, &mask, first, &bank, |_| Ok(vec![None]))?;
        let bits = |h: Tensor| -> Result<Vec<u32>> {
            Ok(h.flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .map(|v| v.to_bits())
                .collect())
        };
        let exact = bits(base)? == bits(none)?;
        let mut times = [Vec::new(), Vec::new(), Vec::new()];
        for round in 0..45 {
            for offset in 0..3 {
                let arm = (round + offset) % 3;
                let start = Instant::now();
                let h = match arm {
                    0 => model.forward(&ids, &mask)?,
                    1 => model.forward_routed(&ids, &mask, first, &bank, |_| Ok(vec![None]))?,
                    _ => model.forward_routed(&ids, &mask, first, &bank, |_| Ok(vec![Some(0)]))?,
                };
                std::hint::black_box(h);
                if round >= 5 {
                    times[arm].push(start.elapsed().as_secs_f64() * 1e6);
                }
            }
        }
        let stat = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            let n = v.len();
            json!({
                "mean_us": v.iter().sum::<f64>() / n as f64,
                "p50_us": v[n / 2],
                "p95_us": v[(n * 95) / 100],
                "samples": n
            })
        };
        let [t0, t1, t2] = times;
        let cell = json!({
            "tokens": l,
            "exact_none_parity": exact,
            "base": stat(t0),
            "routed_none": stat(t1),
            "routed_selected": stat(t2),
        });
        eprintln!("whole_pass {cell}");
        cells.push(cell);
    }
    std::fs::write(
        &out_path,
        serde_json::to_vec_pretty(&json!({
            "stage": "untrained whole-pass diagnostic",
            "hidden_size": d,
            "num_hidden_layers": cfg.num_hidden_layers,
            "routed_first_layer": first,
            "rank": r,
            "factor_bytes": bank[0].bytes(),
            "cells": cells,
        }))
        .map_err(hanzo_ml::Error::wrap)?,
    )?;
    Ok(())
}
