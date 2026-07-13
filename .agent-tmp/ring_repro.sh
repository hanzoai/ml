#!/bin/bash
# 2-node loopback ring on dbc: rank0 = CPU head, rank1 = Metal worker.
# Reproduces "a Metal rank in the PP ring deadlocks". Dense Qwen3-0.6B first (transport discriminator).
set -u
BIN=~/work/hanzo/engine-ring/target/release/hanzo
MODEL_REPO="Qwen/Qwen3-0.6B-GGUF"
MODEL_FILE="Qwen3-0.6B-Q8_0.gguf"
SPLIT="${SPLIT:-14,14}"
D=/tmp/ringrepro
rm -rf "$D"; mkdir -p "$D"

cat > "$D/rank0.json" <<'EOF'
{"master_ip": null, "master_port": 9100, "port": 9000, "right_port": 9001, "right_ip": "127.0.0.1", "rank": 0, "world_size": 2}
EOF
cat > "$D/rank1.json" <<'EOF'
{"master_ip": "127.0.0.1", "master_port": 9100, "port": 9001, "right_port": 9000, "right_ip": "127.0.0.1", "rank": 1, "world_size": 2}
EOF

echo "=== launching rank1 (Metal worker) ==="
RING_CONFIG="$D/rank1.json" RING_LAYER_SPLIT="$SPLIT" RUST_LOG=info \
  "$BIN" serve -p 9235 --no-ui --no-advertise -m "$MODEL_REPO" --format gguf -f "$MODEL_FILE" \
  > "$D/rank1.log" 2>&1 &
R1=$!
echo "rank1 pid=$R1"

sleep 2
echo "=== launching rank0 (CPU head) ==="
RING_CONFIG="$D/rank0.json" RING_LAYER_SPLIT="$SPLIT" RUST_LOG=info \
  "$BIN" serve -p 9234 --no-ui --no-advertise -m "$MODEL_REPO" --format gguf -f "$MODEL_FILE" --cpu --dtype f32 \
  > "$D/rank0.log" 2>&1 &
R0=$!
echo "rank0 pid=$R0"

echo "=== waiting for head /v1/models (up to 120s) ==="
UP=0
for i in $(seq 1 60); do
  if curl -s -m 2 http://127.0.0.1:9234/v1/models >/dev/null 2>&1; then UP=1; break; fi
  if ! kill -0 $R0 2>/dev/null; then echo "rank0 DIED"; break; fi
  if ! kill -0 $R1 2>/dev/null; then echo "rank1 DIED"; break; fi
  sleep 2
done
echo "head_up=$UP"
MODELID=$(curl -s -m 3 http://127.0.0.1:9234/v1/models 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
echo "modelid=$MODELID"

echo "=== sending completion (60s timeout) ==="
REQ_START=$(date +%s)
RESP=$(curl -s -m 60 http://127.0.0.1:9234/v1/chat/completions \
  -H 'content-type: application/json' \
  -d "{\"model\":\"${MODELID:-default}\",\"messages\":[{\"role\":\"user\",\"content\":\"count to five\"}],\"max_tokens\":20,\"stream\":false}" 2>&1)
RC=$?
REQ_END=$(date +%s)
echo "curl_rc=$RC elapsed=$((REQ_END-REQ_START))s"
echo "RESP=${RESP:0:400}"

if [ "$RC" != "0" ] || [ -z "$RESP" ]; then
  echo "=== DEADLOCK SUSPECTED -> sampling rank1 (Metal worker) ==="
  sample $R1 4 -file "$D/rank1.sample.txt" 2>/dev/null
  echo "--- rank1 sample (thread summary) ---"
  grep -A40 "Call graph" "$D/rank1.sample.txt" 2>/dev/null | head -80
  echo "=== CPU usage snapshot ==="
  ps -p $R1 -o pid,%cpu,%mem,command 2>/dev/null | head -2
  ps -p $R0 -o pid,%cpu,%mem,command 2>/dev/null | head -2
  echo "=== socket state (ring ports) ==="
  netstat -an 2>/dev/null | grep -E "9000|9001" | head
else
  echo "=== NO DEADLOCK: dense Metal worker completed the forward ==="
fi

echo "=== rank1 log tail ==="; tail -25 "$D/rank1.log"
echo "=== rank0 log tail ==="; tail -25 "$D/rank0.log"

echo "=== cleanup ==="
kill $R0 $R1 2>/dev/null; sleep 1; kill -9 $R0 $R1 2>/dev/null
echo DONE
