# hanzo-train

LoRA + QLoRA trainer for the model families that `hanzo-engine`
(`mistralrs-core`) already serves: DeepSeek V3 (Kimi K2.5 base),
Qwen3 / Qwen3-MoE / Qwen3-Next, GLM-4 / GLM-4-MoE.

Produces PEFT-compatible `adapter_model.safetensors` files that load
directly through `mistralrs-core::lora::LoraLinear` and `QLoraLinear`.

## Scope

* In: LoRA + QLoRA supervised fine-tuning. AdamW + cosine warmup.
  PEFT-format save/load. MoE-routing-aware attach (shared experts
  only by default).
* Out: full-parameter SFT, GRPO/DPO/RLHF. Those stay in Python until
  Rust has a vectorised sampler — see `src/federation_hook.rs` for
  the hand-off shape.

## Quick start

```bash
cd ~/work/hanzo/ml/hanzo-train
cargo test                    # AdamW + LoRA forward equivalence
cargo run --example finetune_qwen3 --release
```

See `examples/finetune_qwen3.rs` for the attach + train + save loop.

## Layout

```
src/
  lora/      LoRA modules, attach helpers, PEFT save/load
  qlora/     4-bit base + fp16 LoRA on top (uses candle Q4_K)
  optim/     AdamW wrapper + LR schedules
  data/      JSONL reader, tokeniser bridge, batch packer
  trainer.rs Glue
  federation_hook.rs   Stub: push BitDeltas to a coordinator
```

## Adapter format

```
adapter_config.json     PEFT v0.x subset: r, lora_alpha, target_modules, ...
adapter_model.safetensors
    base_model.model.<dotted_path>.lora_A.weight  (rank, in_features)
    base_model.model.<dotted_path>.lora_B.weight  (out_features, rank)
```

Verified to round-trip through `mistralrs-core::lora::make_adapter`'s
expected `(rank, in_features)` and `(out_features, rank)` shape contract.
