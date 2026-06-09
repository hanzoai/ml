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
| engine `codec_validation` | cosine > 0.99 / 0.999 | **SKIP** | Real assertive cargo tests in `qwen3_tts/mod.rs` (codec, talker, prefill, full-gen greedy). They `return` early unless `ZEN3_*` PyTorch reference dumps (`codes_QT.i64`, `tk_prefill.f32`, ...) are set. Those raw dumps are **not checked into the repo** and were not present on spark at suite-build time, so they no-op. To enable: regenerate the dumps from the Python zen-3-tts reference and export `ZEN3_CODEC_WEIGHTS`/`ZEN3_MAIN_WEIGHTS`/etc., then `run_e2e_tests.sh components` runs them. The TTS path is still covered end-to-end and approximately by the round-trip check above; `codec_validation` is the exact/deterministic complement. |

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
