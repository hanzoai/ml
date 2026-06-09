# Native MuseTalk audio-feature stage (Rust whisper-tiny)

Ports MuseTalk's audio-feature pipeline to Rust, eliminating the Python `whisper-tiny`
dependency in the native dub. The UNet cross-attention context is now produced end-to-end in
hanzo-ml: **wav → log-mel → whisper-tiny encoder (5 hidden states) → per-frame 5×10 chunk →
PositionalEncoding → `[num_frames, 50, 384]`**.

## What MuseTalk actually feeds the UNet

From `musetalk/utils/audio_processor.py` (`AudioProcessor`, the path used by the proven
`musetalk_dub.py`), NOT the local `whisper/audio2feature.py`:

1. **mel**: HF `WhisperFeatureExtractor` → `[1, 80, 3000]` per 30 s segment.
2. **encoder**: `WhisperModel.encoder(mel, output_hidden_states=True).hidden_states` →
   **5 tensors** for whisper-tiny (`hs[0]` = post-conv + position embedding; `hs[1..3]` = block
   outputs; **`hs[4]` = `layer_norm(block3_output)`**, i.e. the final encoder LayerNorm is applied
   to the *last* hidden state). `torch.stack(.., dim=2)` → `[1, T, 5, 384]`.
3. **chunk geometry** (`get_whisper_chunk`, fps=25, pad_left=pad_right=2): trim to
   `actual_length = floor(len/16000 * 50)`, zero-pad `[left = ceil(50/fps)*2, right = ceil(50/fps)*3*2]`,
   then per video frame gather `audio_feature_length_per_frame = 2*(2+2+1) = 10` audio steps at
   `audio_index = floor(frame * 50/fps)`, giving `[1, 10, 5, 384]` → rearrange `b c h w -> b (c h) w`
   → `[1, 50, 384]`.
4. **PositionalEncoding** (`unet.py`, standard sinusoidal `d_model=384`, max_len 5000) added to the
   `[B, 50, 384]` chunk → the UNet `encoder_hidden_states`.

The "5-window" is the **5 whisper-tiny encoder hidden states**; the "10" is the temporal context
(10 audio frames at 50 fps centered on the video frame). 10×5 = 50 rows of 384.

## Architecture note

MuseTalk uses **vanilla HF/OpenAI whisper-tiny** (`d_model=384`, 4 layers, 6 heads, ffn 1536,
80 mel bins, learned absolute `embed_positions`, bidirectional attention, LayerNorm, GELU MLP).
This is *not* the causal Voxtral encoder (`hanzo-engine/.../voxtral/encoder.rs`, which uses RoPE +
SwiGLU + RmsNorm + sliding-window causal attention). The reusable encoder used here is the stock
Whisper `AudioEncoder` (`hanzo-transformers/src/models/whisper/model.rs`); the new
`musetalk_audio.rs` adds the hidden-state capture + mel + chunk + PE that MuseTalk needs.

## Verified parity (vs PyTorch reference, demo clip `clip/tts_en.wav`, 8.2 s)

| stage | cpu f32 | cuda f16 |
|-------|---------|----------|
| mel `[1,80,3000]` | cosine 1.000000 / 120 dB | cosine 1.000000 / 120 dB (mel runs in f32) |
| hidden_stack `[1,1500,5,384]` | cosine 1.000000 / 101.5 dB | cosine 0.999994 / 49.2 dB |
| chunks_all `[205,50,384]` | cosine 1.000000 / 109.6 dB | cosine 0.999999 / 55.9 dB |
| per-frame post-PE `[1,50,384]` | cosine 1.000000 / ~110 dB | cosine ≥0.999999 / ~58 dB |

The Rust `dump`ed `audio_*.npy` also match the Python harness's `dubin/audio_*.npy` at
cosine ≥0.9999992 / ~58 dB (f16). The remaining gap is pure f16 round-off, well above the
matching threshold (cosine > 0.9999).

### The one divergence we hit and fixed

Initial impl had the last hidden state wrong: HF's `output_hidden_states` applies the encoder's
final `layer_norm` to the **last** hidden state (`hs[4] = layer_norm(block3_out) == last_hidden_state`).
Pushing the raw block-3 output gave `hs[4]` cosine 0.54 (hs[0..3] were already exact). Applying
`ln_post` to the final hidden state fixed it → cosine 1.0.

## Layout

- `../hanzo-transformers/src/models/whisper/musetalk_audio.rs` — the reusable library port
  (`hanzo_nn::VarBuilder`): `WhisperEncoder` (+ `forward_hidden_states`/`forward_debug`),
  `log_mel_spectrogram`, `PositionalEncoding`, `AudioFeatureExtractor` (`mel` / `encode_hidden` /
  `whisper_chunks` / `positional_encoding`).
- `whisper-feats-verify/` — standalone runnable crate (`hanzo_quant::ShardedVarBuilder` flavor,
  `whisper_shardedvb.rs`) used by the `musetalk-bench` harness. Two modes:
  - default: parity harness (prints cosine/PSNR per stage vs `refdump/wf/*.npy`).
  - `dump`: `WF_AUDIO=<wav> WF_OUTDIR=<dir> [WF_FPS=25] MUSETALK_DEV=cuda MUSETALK_DTYPE=f16` →
    writes `audio_{i:06}.npy` (post-PE) — the drop-in replacement for the Python whisper stage.
- `musetalk_dub_native.py` — the native dub harness, now with `--native_whisper 1` (default):
  skips loading the Python `WhisperModel`/`AudioProcessor` and calls the Rust `dump` tool.
- `reference/dump_whisper_feats.py` — dumps the PyTorch reference tensors for parity checking.

## Reproduce (on spark, CUDA)

```sh
# 1) stage weights + HF mel filterbank
cp .../models/whisper/model.safetensors   ~/work/zen-dub-run/rustweights/whisper.safetensors
#   mel_filters.npy = WhisperFeatureExtractor(...).mel_filters  ([201,80] f32)

# 2) python reference (CPU is fine; deterministic)
python native-dub/reference/dump_whisper_feats.py

# 3) build + verify (CUDA env: CUDA_HOME=/usr/local/cuda CPATH=$CUDA_HOME/include LD_LIBRARY_PATH=$CUDA_HOME/lib64)
cargo build --release --features cuda           # in whisper-feats-verify
MUSETALK_DEV=cuda MUSETALK_DTYPE=f16 ./target/release/whisper-feats-verify         # parity
WF_AUDIO=clip/tts_en.wav WF_OUTDIR=/tmp/wf MUSETALK_DEV=cuda MUSETALK_DTYPE=f16 \
  ./target/release/whisper-feats-verify dump    # produce audio_*.npy
```
