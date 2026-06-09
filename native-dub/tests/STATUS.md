# Honest status of every check

Updated after a live run on **spark** (NVIDIA GB10, CUDA). "Live" = runs real weights
on the GPU and asserts a numeric threshold. "Skip" = the check is implemented and gated,
but its inputs are not present in this checkout so it no-ops loudly.

## Tier 3 -- structural CI (`cargo test`, no GPU)

| check | status | notes |
|-------|--------|-------|
| charmatch helper self-test | **LIVE** | pure-Rust metric; 4 unit tests + a runner self-test on the real ASR/ref strings (0.9655). |
| musetalk-bench structural | **LIVE** | 12 unit tests: MuseTalkConfig invariants (8->4ch, 256px, GroupNorm32, x-attn 384), `get_crop_box`/`resize`/`dilate`/`erode` geometry, `iou`/`musetalk_bbox` formula. |
| engine zen3 structural | **LIVE** | 2 unit tests: parse the real zen-3-asr/tts `config.json`, assert 16 code groups, 16kHz, vocab>0, distinct special tokens. JSON-only, no weights. |

## Tier 2 -- per-component GPU regression

| check | metric / threshold | status | notes |
|-------|--------------------|--------|-------|
| zen3-ASR | char-match >= 0.95 | **LIVE** | `zen3-serving asr` on `sun.wav`, vs `transcript_zh.txt`. Real ~0.966. |
| zen3-TTS round-trip | char-match >= 0.60 | **LIVE** | `zen3-serving tts` -> `zen3-serving asr` (English). Composes 2 sampled/AR stages; loose by design. |
| MuseTalk render | min-stage cosine >= 0.99 | **LIVE** | `musetalk-bench realverify` CUDA f16 vs PyTorch refdump. Real ~0.99999. |
| whisper-feats | min cosine >= 0.999 | **LIVE** | `musetalk-bench whisperfeat` CUDA f32 vs PyTorch refdump/wf. Real ~1.000. |
| SFD+FAN | IoU >= 0.95, 100% frames | **LIVE** | `face-detect-run` over all 550 ref frames vs `face_alignment`. Real mean ~0.999. |
| BiSeNet blend | SSIM >= 0.99, IoU >= 0.99 | **LIVE** | `musetalk-bench face-blend` CUDA f32 vs PyTorch blendref. Real SSIM ~1.000. |
| engine `codec_validation` | cosine > 0.99 / 0.999 | **LIVE** (one-command-enable) | Real assertive cargo tests in `qwen3_tts/mod.rs` (codec, talker, prefill, full-gen greedy) vs PyTorch reference tensors. The raw dumps (`codes_QT.i64`, `tk_prefill.f32`, `ref_*.f32`, ...) are **not checked into git** (they are derived data), so the tests `return` early until `ZEN3_*` is exported. They are now **one-command regenerable** from the real PyTorch zen-3-tts reference via `reference/dump_tts_ref.py` (wrapped by `reference/gen_tts_ref.sh`), and the runner regenerates+runs them when `ZEN3_GEN_REF=1`. **Measured on spark (CPU, transformers==4.57.3 vs the Rust CPU decoder, 2026-06):** all 4 pass -- `prefill` cos=1.000000 (mad 2e-6); `talker` hidden cos=0.999999 / logits cos=0.999999 and frame-0 codes bit-exact; `codec` quant/pretrans/upsample/wav cos=1.000000; `full-gen greedy` codebook-0 48/48 and full-frame (all 16 codes) 8/8. See "Enabling codec_validation" below. |

## Tier 1 -- full e2e dub

| check | status | notes |
|-------|--------|-------|
| pipeline exit 0 | **LIVE** | runs `run_fullnative_dub.sh` verbatim (`ZEN3_DEVICE=cuda`). |
| video H.264 | **LIVE** | `ffmpeg -i` probe of the output. |
| video 576x768 | **LIVE** | probe. |
| video duration > 0 | **LIVE** | probe. |
| audio stream present | **LIVE** | probe. |
| audio 24000 Hz | **LIVE** | probe. |
| pipeline ASR char-match >= 0.95 | **LIVE** | parses the `    zh:` line from the run log. |
| translation non-empty English | **LIVE** | parses `    en:` line, asserts ascii + >=3 words. |
| pipeline TTS round-trip >= 0.60 | **LIVE** | re-transcribes `fn_work/tts_16k.wav`. |
| MuseTalk render cosine >= 0.99 | **LIVE** | reuses `realverify` (the render weights/refs are identical to what `dub-full` runs). |
| no Python/torch/onnx in ML path | **LIVE** | greps the run log; the only `ffmpeg` is the explicit video<->frame / mux glue. |

## Known caveats (read these)

1. **CPU does not pass MuseTalk.** `realverify` on `MUSETALK_DEV=cpu MUSETALK_DTYPE=f32`
   gives VAE-encode cos 0.896 and VAE-decode cos 0.099 -- there is a real CPU-path VAE
   discrepancy. The production/verified path is **CUDA f16**, where every stage is >= 0.99999.
   The MuseTalk checks therefore deliberately run on CUDA only; they are GPU-gated, not CPU.

2. **The known transcript is the first sentence.** `transcript_zh.txt` is the first sentence
   of the `sun` clip; the audio continues past it. The char-match metric aligns the reference
   as a prefix of the (longer) ASR output, so a correct full transcription scores ~0.97
   (one homophone 地/的), not 1.0. Threshold 0.95 holds with margin.

3. **TTS round-trip is intentionally loose (0.60).** It composes a non-deterministic
   translation + sampled TTS + a second ASR. It proves the synthesized audio is intelligible
   English close to the text, not bit-exactness. Determinism is the job of `codec_validation`.

4. **Reference dumps + weights are not in git** (multi-GB safetensors, PyTorch `.npy`/`.qt`).
   They live on spark `~/work/zen-dub-run/`. Every path is env-overridable. A check whose
   inputs are absent prints `[SKIP]` with the reason rather than failing.

## Enabling `codec_validation` (one command)

The deterministic zen-3-TTS verifier needs PyTorch reference tensors. Regenerate them from the
real QwenLM/Qwen3-TTS model + the `zen-3-tts-0.6B` weights, then the cargo tests run + assert.

**As part of the suite** (regenerates, then runs the check):

```bash
ZEN3_GEN_REF=1 native-dub/tests/run_e2e_tests.sh components
```

**Standalone** (regenerate once, then run the cargo tests directly):

```bash
native-dub/reference/gen_tts_ref.sh                 # -> ~/work/zen-dub-run/tts-ref/*.{f32,i64} + meta.env
source ~/work/zen-dub-run/tts-ref/meta.env           # exports ZEN3_MAIN_WEIGHTS, ZEN3_TK_PREFILL, ...
( cd "$HANZO_ENGINE" && cargo test -p hanzo-engine --lib codec_validation -- --nocapture --test-threads=1 )
```

What it does:

- `reference/dump_tts_ref.py` loads the real PyTorch zen-3-tts (`modeling_qwen3_tts.py` +
  `modeling_qwen3_tts_tokenizer_v2.py`) on a fixed prompt + fixed seed, runs a fully **greedy**
  (deterministic) decode, and writes the exact per-stage tensors the four tests consume as **raw
  little-endian** `f32`/`i64`: the talker prefill / hidden / logits / frame-0 codes, the full
  greedy code grid, and the codec decode stages (SplitRVQ `quant` -> pre-transformer `pretrans` ->
  post-upsample -> final `wav`). It writes `meta.txt` (shapes + scalars) and `meta.env` (the
  exact `export ZEN3_*` block).
- `reference/gen_tts_ref.sh` provisions a `--system-site-packages` venv with the **pinned**
  `transformers==4.57.3` (the modeling code's API; the system transformers is a different major
  and will not load the model) and runs the dump. Override the interpreter with `TTS_REF_PY=...`.
- The dump runs on **CPU by default** (`ZEN3_REF_DEVICE=cpu`) to match the cargo tests, which load
  the weights as f32 on `Device::Cpu` and need no GPU. (On the GB10 the unified memory is often
  already held by a running `hanzo` server, so CPU is also the robust default.)

**Honest note on tolerances.** `codec` and `prefill` match to cos=1.000000 (the prefill is the same
embedding arithmetic on both sides; the codec decode is f32 convs/attention with only float
accumulation-order noise -- e.g. `upsample` max-abs-diff ~5e-3 but cos still 1.0). The talker is a
28-layer transformer: hidden/logits cos=0.999999 with max-abs-diff ~0.04-0.07 on individual
elements (f32 accumulation order across 28 layers + attention), comfortably above the 0.999 gate,
and the greedy argmax codes are **bit-exact**. Full-gen matches codebook-0 48/48 over the dumped
length; the test allows late tie-break flips beyond the first 8 frames (this degenerate greedy
sequence has long repeated runs whose near-tie logits are sensitive to accumulation order), which
did not occur here.
