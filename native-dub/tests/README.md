# Native-dub e2e test suite

Repeatable, assertive tests for the **fully-native** (zero-Python-in-the-ML-path)
zen-dub pipeline:

```
clip(video) + clip(audio, zh)
  -> zen3-ASR        (transcribe zh)              [engine speech_models]
  -> Qwen3 translate (zh -> en)                   [hanzo engine, GGUF]
  -> zen3-TTS        (synthesize en, 24kHz)       [engine speech_models]
  -> Rust render: SFD+FAN -> whisper-feats        [musetalk-bench `dub-full`]
                  -> MuseTalk -> BiSeNet blend
  -> ffmpeg mux
```

The components were previously verified ad-hoc by hand (cosine/PSNR/IoU printed to
a terminal). This suite turns those checks into **committed, repeatable assertions
with explicit thresholds**, adds a **full-pipeline** test, and a **runner** that
prints PASS/FAIL per check.

## Tiers

| Tier | needs | what it proves | how to run |
|------|-------|----------------|------------|
| **3 CI** (cheap) | nothing (no GPU / no big weights) | modules compile, configs parse, tensor shapes/geometry correct, char-match metric sound | `cargo test` (see below) |
| **2 component** (regression) | GPU + per-component weights (`MUSETALK_*` / `ZEN3_*`) | each native stage matches its PyTorch reference to threshold | `run_e2e_tests.sh components` |
| **1 full e2e** | GPU + ALL weights + demo clip | the whole dub runs exit-0, emits a valid H.264/AAC video, hits accuracy thresholds, and has NO Python ML in the path | `run_e2e_tests.sh full` |

Run **everything** (CI + components + full):

```bash
native-dub/tests/run_e2e_tests.sh all
# or
make -C native-dub/tests e2e
```

## Thresholds (per the spec)

| check | metric | threshold |
|-------|--------|-----------|
| zen3-ASR | char-match of known `sun` zh transcript prefix | >= 0.95 |
| translation | non-empty, ascii English, >= 3 words | (sanity) |
| zen3-TTS round-trip | TTS(en) -> zen3-ASR(en) char-match to translation | >= 0.60 (see note) |
| MuseTalk render | per-stage + e2e cosine vs PyTorch (CUDA f16) | >= 0.99 |
| whisper-feats | cosine vs PyTorch | >= 0.999 |
| SFD+FAN | MuseTalk-bbox IoU vs `face_alignment` ref | >= 0.95 (and 100% of frames >= 0.95) |
| BiSeNet blend | composite SSIM / mask IoU vs Python | SSIM >= 0.99, IoU >= 0.99 |
| full-pipeline video | ffmpeg probe | H.264, 576x768, dur>0, audio present, 24000 Hz |
| no-Python-ML | grep run log for `python\|torch\|onnx` | empty |

**TTS round-trip note.** The round-trip threshold is the loosest because it composes
three lossy stages (Qwen3 translation is non-deterministic across temperatures, zen3-TTS
samples, and the re-transcription is itself an ASR). It asserts the TTS audio is *real
intelligible English close to the translation* (>=60% char overlap of the longest common
content), not bit-exactness. The deterministic TTS *internals* are covered separately and
exactly by the engine `codec_validation` cargo tests (cosine > 0.99 / 0.999 vs PyTorch) --
those are gated on `ZEN3_*` reference dumps.

## Environment

Reference artifacts + weights live on **spark** under `~/work/zen-dub-run/`:

```
clip/sun6.mp4                       demo talking-head (576x768, 25fps)
clip/transcript_zh.txt              known zh transcript (sun, first sentence)
zen-dub/data/audio/sun.wav          source zh speech (44.1k)
rustweights/{vae,unet,whisper,bisenet}.safetensors + mel_filters.npy
facedump/{s3fd,fan2d}.safetensors + ref_face.json + frames/
refdump/*.npy                       MuseTalk per-stage PyTorch refs
refdump/wf/*.npy                    whisper-feats PyTorch refs
blendref/*.npy                      BiSeNet blend PyTorch refs
```

The runner reads paths from env (all overridable); defaults point at the spark layout
above. See the top of `run_e2e_tests.sh` for every knob.

## Binaries it drives

- `zen3-serving` (engine `native/zen3-serving`) -- `asr` / `tts` subcommands, built `--features cuda`.
- `hanzo` (engine `hanzo-cli`) -- Qwen3 GGUF translate, built `--features cuda`.
- `musetalk-bench` (`sw-perf/musetalk-bench`) -- `realverify` / `whisperfeat` / `face-blend` / `dub-full`, built `--features cuda`.

The orchestrator `../run_fullnative_dub.sh` is the single source of truth for the full
pipeline; the full-e2e test runs it verbatim and asserts on its output + log.

## Honest status of each check

See `STATUS.md` for exactly which checks run live vs which are skipped (and why).
The short version: ASR / TTS-round-trip / MuseTalk / whisper-feats / SFD+FAN / BiSeNet
/ full-pipeline all run **live with real assertions** on spark's GPU. The engine
`codec_validation` cargo tests are real assertions but **skip** unless their `ZEN3_*`
PyTorch reference dumps are regenerated (they are not checked into the repo).
