# hanzo-transformers

Model implementations on [hanzo-ml](https://crates.io/crates/hanzo-ml) and
[hanzo-nn](https://crates.io/crates/hanzo-nn). 22 have a quantized (GGUF) variant
(`quantized_*`).

```toml
[dependencies]
hanzo-transformers = "0.11"
```

`src/models` holds one module per architecture:

- **Language**: Llama, Qwen2 and Qwen3 (dense, MoE, VL), Gemma 1–4, Mistral, Mixtral, Phi,
  DeepSeek-V2, GLM4, Granite, LFM2, OLMo, StableLM, StarCoder2, RWKV v5–v7, Mamba and Mamba2.
- **Encoders**: BERT, DistilBERT, DeBERTa-v2, ModernBERT, XLM-RoBERTa, T5, Marian.
- **Vision**: ViT, DINOv2, CLIP, SigLIP, Segment Anything, SegFormer, ConvNeXt, EfficientNet,
  FastViT, Hiera, I-JEPA, LLaVA, Pixtral, Moondream, PaliGemma.
- **Audio**: Whisper, EnCodec, Mimi, SNAC, DAC, Voxtral, Parler-TTS, MetaVoice, CSM.
- **Diffusion**: Stable Diffusion, MMDiT, Flux, Wuerstchen, Z-Image.

Also here: `generation` (sampling), `fused_moe`, `quantized_nn` and `quantized_var_builder` for
loading GGUF weights.

Runnable programs for most models are in
[hanzo-ml-examples](https://github.com/hanzoai/ml/tree/main/hanzo-ml-examples/examples).

## License

MIT OR Apache-2.0. See [NOTICE](https://github.com/hanzoai/ml/blob/main/NOTICE) for
third-party attributions.
