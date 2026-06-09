#!/usr/bin/env bash
# Regenerate the zen-3-TTS PyTorch reference dumps that enable the engine `codec_validation`
# cargo tests, and print the env block the runner / cargo need.
#
# One-command enable:
#     native-dub/reference/gen_tts_ref.sh && source ~/work/zen-dub-run/tts-ref/meta.env
#     ( cd "$HANZO_ENGINE" && cargo test -p hanzo-engine --lib codec_validation -- --nocapture )
#
# or just let the e2e runner do it:
#     ZEN3_GEN_REF=1 native-dub/tests/run_e2e_tests.sh components
#
# Env knobs (all optional; defaults = spark layout):
#   TTS_MODEL        zen-3-tts model dir            (~/work/zen/hf/zen-3-tts-0.6B)
#   QWEN3_TTS_REPO   QwenLM/Qwen3-TTS checkout      (~/work/zen/repos/Qwen3-TTS)
#   TTS_REF_PY       python with transformers==4.57.3 + torch + librosa + soundfile
#                    (default: a --system-site-packages venv we create on demand)
#   ZEN3_REF_OUT     output dir for the dumps       (~/work/zen-dub-run/tts-ref)
#   ZEN3_REF_DEVICE  cpu|cuda                        (cpu -- matches the cargo tests, which run on CPU)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TTS_MODEL="${TTS_MODEL:-$HOME/work/zen/hf/zen-3-tts-0.6B}"
QWEN3_TTS_REPO="${QWEN3_TTS_REPO:-$HOME/work/zen/repos/Qwen3-TTS}"
ZEN3_REF_OUT="${ZEN3_REF_OUT:-$HOME/work/zen-dub-run/tts-ref}"
ZEN3_REF_DEVICE="${ZEN3_REF_DEVICE:-cpu}"
DUMP_PY="$SCRIPT_DIR/dump_tts_ref.py"

# Find / build a python that has the pinned transformers==4.57.3 (the modeling code's API).
# The system transformers is often a different major (5.x) and the modeling code won't load on it.
pick_python() {
  if [ -n "${TTS_REF_PY:-}" ]; then echo "$TTS_REF_PY"; return; fi
  local venv="$HOME/work/zen-dub-run/.venv-tts-ref"
  if [ ! -x "$venv/bin/python" ]; then
    echo "creating reference venv (--system-site-packages + transformers==4.57.3) ..." >&2
    python3 -m venv --system-site-packages "$venv" >&2 || return 1
    "$venv/bin/python" -m pip install -q "transformers==4.57.3" >&2 || return 1
  fi
  # ensure the pin is present (a stale venv may not have it)
  "$venv/bin/python" -c "import transformers,sys; sys.exit(0 if transformers.__version__.startswith('4.57') else 1)" 2>/dev/null \
    || "$venv/bin/python" -m pip install -q "transformers==4.57.3" >&2
  echo "$venv/bin/python"
}

PY="$(pick_python)" || { echo "FATAL: could not provision a transformers==4.57.3 python" >&2; exit 1; }
echo "[gen_tts_ref] python = $PY"
echo "[gen_tts_ref] model  = $TTS_MODEL"
echo "[gen_tts_ref] out    = $ZEN3_REF_OUT  (device=$ZEN3_REF_DEVICE)"

ZEN3_TTS_MODEL="$TTS_MODEL" QWEN3_TTS_REPO="$QWEN3_TTS_REPO" \
  ZEN3_REF_OUT="$ZEN3_REF_OUT" ZEN3_REF_DEVICE="$ZEN3_REF_DEVICE" \
  "$PY" "$DUMP_PY" || { echo "FATAL: dump_tts_ref.py failed" >&2; exit 1; }

# Emit a clean, sourceable env file (just the `export ...` lines from meta.txt).
grep '^export ' "$ZEN3_REF_OUT/meta.txt" > "$ZEN3_REF_OUT/meta.env"
echo "[gen_tts_ref] env -> $ZEN3_REF_OUT/meta.env"
echo "[gen_tts_ref] enable the cargo tests with:  source $ZEN3_REF_OUT/meta.env"
