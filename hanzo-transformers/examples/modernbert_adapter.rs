//! CPU runtime smoke benchmark, synthetic token IDs only. Never trains or reads evaluation data.
use hanzo_ml::quantized::GgmlDType;
use hanzo_ml::{DType, Device, Result, Tensor};
use hanzo_nn::VarBuilder;
use hanzo_transformers::models::modernbert::{
    adapter::{Adapter, Layer, Lora},
    Config, ModernBert,
};
use serde_json::json;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        hanzo_ml::bail!(
            "usage: modernbert_adapter config.json weights.safetensors tensor-prefix output.json"
        );
    }
    let cfg: Config =
        serde_json::from_slice(&std::fs::read(&args[1])?).map_err(hanzo_ml::Error::wrap)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&args[2]], DType::F32, &Device::Cpu)? };
    let mut model = ModernBert::new(vb.pp(&args[3]), &cfg)?;
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
        let cell = json!({"tokens":length,"mean_us":us.iter().sum::<f64>()/us.len() as f64,"p50_us":us[100],"p95_us":us[190],"samples":200});
        eprintln!("isolated {cell}");
        isolated.push(cell);
    }
    std::fs::write(format!("{}.isolated.json",args[4]), serde_json::to_vec_pretty(&json!({"stage":"untrained incremental-kernel diagnostic", "layers":2,"rank":4,"factor_bytes":bank[0].bytes(),"cells":isolated,"excludes":["router","pooling","projection splitting/concatenation","production head","cache effects"],"final_harness":"NOT_READ"})).map_err(hanzo_ml::Error::wrap)?)?;
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
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let delta: Vec<f64> = times[2].iter().zip(&times[0]).map(|(a, b)| a - b).collect();
        let avg = mean(&delta);
        let se = (delta.iter().map(|x| (x - avg).powi(2)).sum::<f64>()
            / (delta.len() - 1) as f64
            / delta.len() as f64)
            .sqrt();
        eprintln!(
            "tokens={l} base_us={:.1} none_added_us={:.1} routed_added_us={avg:.1} se_us={se:.1}",
            mean(&times[0]),
            mean(&times[1]) - mean(&times[0])
        );
        cells.push(json!({"tokens":l,"none_exact":exact,"base_mean_us":mean(&times[0]),"none_added_mean_us":mean(&times[1])-mean(&times[0]),"routed_added_mean_us":avg,"paired_standard_error_us":se,"raw_us":times}));
        let result = json!({"purpose":"untrained runtime smoke; synthetic token IDs; no benchmark harness","backend":"cpu-q8_0","first_layer":first,"layers":2,"rank":r,"scale":4.0,"factor_bytes":bank[0].bytes(),"rounds":40,"warmups":5,"cells":cells,"limitations":"Factor bytes exclude activation scratch. Fixed routing excludes router head compute. Whole-forward timing includes host noise. This is not a production pre-gate pass."});
        std::fs::write(
            &args[4],
            serde_json::to_vec_pretty(&result).map_err(hanzo_ml::Error::wrap)?,
        )?;
    }
    Ok(())
}
