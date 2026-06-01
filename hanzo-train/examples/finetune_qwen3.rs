//! End-to-end demo of LoRA fine-tuning a Qwen3-shaped decoder block.
//!
//! We build a single decoder layer with the exact 7-linear footprint
//! (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
//! `down_proj`) directly out of `candle_nn::linear_no_bias` so the
//! example is self-contained — no model checkpoint download required.
//! The same `attach_lora` call works against a real Qwen3 / DeepSeek /
//! GLM-4 model once the user has exposed their linears as
//! [`AttachTarget`]s; see `lora::attach::AttachTarget`'s doc.
//!
//! Run with:
//!
//! ```text
//! cargo run -p hanzo-train --example finetune_qwen3 --release
//! ```

use anyhow::Result;
use candle::{DType, Device, Tensor};
use candle_nn::{linear_no_bias, VarBuilder, VarMap};

use hanzo_train::{
    cosine_with_warmup, AttachReport, AttachTarget, LoraConfig, MoeMode, TrainableAdamW,
    TrainableAdamWConfig, TrainableLoraLinear,
};

const HIDDEN: usize = 64;
const INTER: usize = 128;
const HEADS: usize = 4;
const SEQ: usize = 16;
const BATCH: usize = 2;
const STEPS: usize = 25;

fn main() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // ----- "Base model": one decoder layer of 7 linears, frozen. -----
    let layer_vb = vb.pp("model.layers.0");
    let attn_vb = layer_vb.pp("self_attn");
    let mlp_vb = layer_vb.pp("mlp");

    let q_proj = linear_no_bias(HIDDEN, HIDDEN, attn_vb.pp("q_proj"))?;
    let k_proj = linear_no_bias(HIDDEN, HIDDEN / HEADS * 2, attn_vb.pp("k_proj"))?;
    let v_proj = linear_no_bias(HIDDEN, HIDDEN / HEADS * 2, attn_vb.pp("v_proj"))?;
    let o_proj = linear_no_bias(HIDDEN, HIDDEN, attn_vb.pp("o_proj"))?;
    let gate_proj = linear_no_bias(HIDDEN, INTER, mlp_vb.pp("gate_proj"))?;
    let up_proj = linear_no_bias(HIDDEN, INTER, mlp_vb.pp("up_proj"))?;
    let down_proj = linear_no_bias(INTER, HIDDEN, mlp_vb.pp("down_proj"))?;

    // ----- Tell hanzo-train which leaves to wrap. -----
    let targets = vec![
        AttachTarget::new("model.layers.0.self_attn.q_proj", q_proj.clone()),
        AttachTarget::new("model.layers.0.self_attn.k_proj", k_proj.clone()),
        AttachTarget::new("model.layers.0.self_attn.v_proj", v_proj.clone()),
        AttachTarget::new("model.layers.0.self_attn.o_proj", o_proj.clone()),
        AttachTarget::new("model.layers.0.mlp.gate_proj", gate_proj.clone()),
        AttachTarget::new("model.layers.0.mlp.up_proj", up_proj.clone()),
        AttachTarget::new("model.layers.0.mlp.down_proj", down_proj.clone()),
    ];

    let lora_cfg = LoraConfig::all_linear(8, 16.0);
    let (wrapped, report): (Vec<(String, TrainableLoraLinear)>, AttachReport) =
        hanzo_train::attach_lora(targets, &lora_cfg, MoeMode::SharedOnly)?;
    println!("attached {} layers, skipped {}", report.attached.len(), report.skipped.len());

    // Trainable params: A and B for each of the 7 wrapped linears.
    let trainable_vars: Vec<_> = wrapped
        .iter()
        .flat_map(|(_, w)| w.trainable_vars())
        .collect();
    println!("trainable Vars: {}", trainable_vars.len());

    // ----- Optimizer + schedule. -----
    let optim_cfg = TrainableAdamWConfig {
        lr: 1e-3,
        ..Default::default()
    };
    let schedule = cosine_with_warmup(2, STEPS, 0.0);
    let mut optim = TrainableAdamW::new(trainable_vars, optim_cfg, schedule)?;

    // ----- Tiny synthetic dataset: predict a fixed target hidden state.
    // The "task" is to make the wrapped layer's output approach `target`.
    let target = Tensor::randn(0f32, 1f32, (BATCH, SEQ, HIDDEN), &device)?;

    // ----- Forward closure that strings together the wrapped layers.
    //
    // For demo purposes we run a single MLP-style pass:
    //   h = down_proj( silu(gate(x)) * up(x) )
    // using the wrapped versions. q/k/v/o are not used here (would
    // require a full attention block); attach still wraps them so the
    // adapter saved at the end covers all seven targets.
    let by_path = |path: &str| -> &TrainableLoraLinear {
        &wrapped.iter().find(|(p, _)| p == path).unwrap().1
    };

    for step in 0..STEPS {
        let x = Tensor::randn(0f32, 1f32, (BATCH, SEQ, HIDDEN), &device)?;
        let gate_out = by_path("model.layers.0.mlp.gate_proj")
            .forward_with_training(&x, true)?;
        let up_out = by_path("model.layers.0.mlp.up_proj")
            .forward_with_training(&x, true)?;
        let silu = candle_nn::ops::silu(&gate_out)?;
        let inner = silu.mul(&up_out)?;
        let out = by_path("model.layers.0.mlp.down_proj")
            .forward_with_training(&inner, true)?;
        let loss = out.sub(&target)?.sqr()?.mean_all()?;
        let lr = optim.backward_step(&loss)?;
        println!(
            "step {step:3}  lr {lr:.6}  loss {:.6}",
            loss.to_scalar::<f32>()?
        );
    }

    // ----- Save adapter in PEFT format. -----
    let out_dir = std::env::temp_dir().join("hanzo-train-demo-adapter");
    hanzo_train::lora::save_peft_adapter(
        &out_dir,
        &wrapped,
        &lora_cfg,
        Some("zenlm/zen-nano-0.6b".into()),
    )?;
    println!("wrote adapter to {}", out_dir.display());

    // Sanity-check that the file we wrote round-trips back.
    let (cfg_back, tensors) =
        hanzo_train::lora::load_peft_adapter(&out_dir, &device)?;
    assert_eq!(cfg_back.r, lora_cfg.rank);
    assert_eq!(cfg_back.target_modules, lora_cfg.target_modules);
    println!(
        "round-trip OK: {} tensor entries, r={}, alpha={}",
        tensors.len(),
        cfg_back.r,
        cfg_back.lora_alpha
    );

    // Avoid unused warnings on the q/k/v/o linears (they are still
    // wrapped in `wrapped` — we just don't run a full attention pass).
    let _ = (&q_proj, &k_proj, &v_proj, &o_proj);
    Ok(())
}
