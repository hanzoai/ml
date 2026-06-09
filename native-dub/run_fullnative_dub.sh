#!/usr/bin/env bash
# Full-native zen-dub pipeline: ZERO Python in the ML path. Only ffmpeg + this shell glue.
#
#   clip(video) + clip(audio, zh)
#     -> zen3-ASR        (transcribe zh)            [hanzo-engine speech_models, zen3-serving]
#     -> Qwen3 translate (zh -> en)                 [hanzo CLI, GGUF]
#     -> zen3-TTS        (synthesize en, SAMPLING)  [hanzo-engine speech_models, zen3-serving]
#     -> Rust whisper feats + Rust MuseTalk + Rust SFD/FAN + Rust BiSeNet blend  (one binary)
#        [musetalk-bench `dub-full`: detect -> crop -> whisper -> unet -> vae -> parse -> blend]
#     -> ffmpeg mux
#
# All ML stages are native Rust. ffmpeg only does video<->frames and audio resample/mux.
set -euo pipefail

RUN="${RUN:-$HOME/work/zen-dub-run}"
FF="${FF:-$RUN/bin/ffmpeg}"
SRC_VIDEO="${SRC_VIDEO:-$RUN/clip/sun6.mp4}"        # video-only talking head (sun)
# Source speech must match SRC_VIDEO's speaker. The validated sun clip (sun.wav,
# zh, 44.1k) transcribes correctly. zen3-ASR resamples 44.1k->16k internally.
SRC_AUDIO_ZH="${SRC_AUDIO_ZH:-$RUN/zen-dub/data/audio/sun.wav}"
# Force the source language; the dub is always zh->en, and pinning avoids the
# model's autoregressive language-ID picking the wrong language on hard clips.
SRC_LANG="${SRC_LANG:-Chinese}"
OUT="${OUT:-$RUN/zen-dub-fullnative.mp4}"
FPS="${FPS:-25}"

# zen3-serving device: ZEN3_DEVICE=cuda -> pass `--cuda` (+ optional ordinal). Anything
# else (or unset) -> CPU. The standalone harness takes the device as a global flag.
ZEN3_DEVICE="${ZEN3_DEVICE:-cpu}"
ZEN3_CUDA_DEVICE="${ZEN3_CUDA_DEVICE:-0}"
ZEN3_DEV_ARGS=()
if [ "$ZEN3_DEVICE" = "cuda" ]; then
  ZEN3_DEV_ARGS=(--cuda --cuda-device "$ZEN3_CUDA_DEVICE")
fi

ZEN3="${ZEN3:-$HOME/work/hanzo/engine/native/zen3-serving/target/release/zen3-serving}"
HANZO="${HANZO:-$HOME/work/hanzo/engine/target/release/hanzo}"
BENCH="${BENCH:-$HOME/work/sw-perf/musetalk-bench/target/release/musetalk-bench}"
ASR_MODEL="${ASR_MODEL:-$HOME/work/zen/hf/zen-3-asr-0.6B}"
TTS_MODEL="${TTS_MODEL:-$HOME/work/zen/hf/zen-3-tts-0.6B}"
# Translator: a properly-quantized native GGUF the hanzo engine runs on CUDA. The
# big qwen3.6-35B-A3B MoE GGUF dequantizes to BF16 (~92GB) and is not viable here;
# zen-eco-4b (Qwen3 4B) runs quantized. -m points at the dir, -f at the file.
QWEN_DIR="${QWEN_DIR:-$HOME/work/zen-eco-4b}"
QWEN_GGUF_FILE="${QWEN_GGUF_FILE:-zen-eco-4b.gguf}"

WORK="$RUN/fn_work"; FRAMES="$WORK/frames"; OUTF="$WORK/out"
mkdir -p "$FRAMES" "$OUTF"
find "$FRAMES" -name '*.png' -delete 2>/dev/null || true
find "$OUTF"   -name '*.png' -delete 2>/dev/null || true

ts() { date +%s.%N; }
dur() { echo "$(echo "$2 - $1" | bc)s"; }

echo "[1/6] zen3-ASR transcribe (zh)  [dev=$ZEN3_DEVICE]"
T0=$(ts)
TRANSCRIPT_ZH=$("$ZEN3" "${ZEN3_DEV_ARGS[@]}" asr --model "$ASR_MODEL" --audio "$SRC_AUDIO_ZH" --lang "$SRC_LANG" --max-new 128 2>"$WORK/asr.err" | tail -1)
T1=$(ts)
echo "    zh: $TRANSCRIPT_ZH"
echo "    [stage1 ASR wall=$(dur "$T0" "$T1")]"

echo "[2/6] Qwen3 translate zh->en (native hanzo engine)"
PROMPT="Translate this Chinese sentence to natural English. Output ONLY the English translation on a single line, no notes, no quotes.\n\n$TRANSCRIPT_ZH"
T0=$(ts)
# Strip ANSI/INFO log lines, drop empty/think lines, keep the last real text line.
TRANSLATION_EN=$("$HANZO" run --paged-attn off --format gguf -m "$QWEN_DIR" -f "$QWEN_GGUF_FILE" -i "$PROMPT" 2>"$WORK/qwen.err" \
  | sed -r 's/\x1b\[[0-9;]*m//g' \
  | grep -vE 'INFO|hanzo_|^Stats:|tokens,|Time to first|Decode:|Prompt:|Prefix cache|Sampling:|^[[:space:]]*$|</?think>' \
  | tail -1)
T1=$(ts)
echo "    en: $TRANSLATION_EN"
echo "    [stage2 translate wall=$(dur "$T0" "$T1")]"

echo "[3/6] zen3-TTS synthesize en (sampling; greedy collapses to silence)  [dev=$ZEN3_DEVICE]"
T0=$(ts)
"$ZEN3" "${ZEN3_DEV_ARGS[@]}" tts --model "$TTS_MODEL" --text "$TRANSLATION_EN" --out "$WORK/tts_en.wav" --max-tokens 1200
"$FF" -v error -y -i "$WORK/tts_en.wav" -ac 1 -ar 16000 "$WORK/tts_16k.wav"
T1=$(ts)
echo "    [stage3 TTS wall=$(dur "$T0" "$T1")]"

echo "[4/6] extract source frames (ffmpeg)"
T0=$(ts)
"$FF" -v error -y -i "$SRC_VIDEO" -start_number 0 "$FRAMES/%08d.png"
T1=$(ts)
echo "    [stage4 frames wall=$(dur "$T0" "$T1")]"

echo "[5/6] native render: SFD/FAN detect -> crop -> whisper feats -> MuseTalk -> BiSeNet blend"
T0=$(ts)
MUSETALK_DEV=cuda MUSETALK_DTYPE=f16 \
MUSETALK_FRAMES="$FRAMES" MUSETALK_AUDIO="$WORK/tts_16k.wav" MUSETALK_OUTFRAMES="$OUTF" \
MUSETALK_WDIR="$RUN/rustweights" MUSETALK_FACEDIR="$RUN/facedump" \
MUSETALK_FPS="$FPS" MUSETALK_BATCH=8 \
  "$BENCH" dub-full
T1=$(ts)
echo "    [stage5 render wall=$(dur "$T0" "$T1")]"

echo "[6/6] encode + mux (ffmpeg)"
T0=$(ts)
"$FF" -y -v warning -r "$FPS" -f image2 -i "$OUTF/%08d.png" -vcodec libx264 -vf format=yuv420p -crf 18 "$WORK/video.mp4"
"$FF" -y -v warning -i "$WORK/video.mp4" -i "$WORK/tts_en.wav" -c:v copy -c:a aac -shortest "$OUT"
T1=$(ts)
echo "    [stage6 mux wall=$(dur "$T0" "$T1")]"
echo "DONE -> $OUT"
