#!/bin/bash
# Hanzo ML Setup Script
# Updates from upstream Candle, checks compilation, sets up training

set -e

HANZO_ML_DIR="/Users/z/work/hanzo/ml"
UPSTREAM_CANDLE="/Users/z/work/hf/candle"
ZEN_DATASET="/Users/z/work/zen/zen-agentic-dataset"
HANZO_ENGINE="/Users/z/work/hanzo/engine"

cd "$HANZO_ML_DIR"

# Pull latest upstream changes
echo "Pulling latest Candle upstream changes..."
if [ -d "$UPSTREAM_CANDLE" ]; then
    cd "$UPSTREAM_CANDLE"
    git pull origin main || true
    LATEST_COMMIT=$(git rev-parse HEAD)
    echo "Latest upstream: $LATEST_COMMIT"
    cd "$HANZO_ML_DIR"
fi

# Test compilation
echo "Testing hanzo-training compilation..."
if cargo check --package hanzo-training --quiet 2>/dev/null; then
    echo "hanzo-training compiles successfully"
else
    echo "hanzo-training has compilation errors"
    cargo check --package hanzo-training 2>&1 | head -20
fi

# Check published crates
echo "Checking published crates..."
CRATES=("hanzo-ml" "hanzo-nn" "hanzo-transformers" "hanzo-datasets" "hanzo-ug")
for crate in "${CRATES[@]}"; do
    if cargo search "$crate" --limit 1 2>/dev/null | grep -q "0.9.2-alpha.2"; then
        echo "  $crate v0.9.2-alpha.2 published"
    else
        echo "  $crate v0.9.2-alpha.2 not found"
    fi
done

# Verify licenses
BSD_COUNT=$(find . -name "Cargo.toml" -not -path "./target/*" -exec grep -l "BSD-3-Clause" {} \; 2>/dev/null | wc -l)
echo "Found $BSD_COUNT crates with BSD-3-Clause license"

# Check dataset
if [ -d "$ZEN_DATASET" ]; then
    echo "Zen Agentic Dataset found"
    DATASET_SIZE=$(find "$ZEN_DATASET" -name "*.json*" 2>/dev/null | wc -l)
    echo "  Dataset files: $DATASET_SIZE"
else
    echo "Zen Agentic Dataset not found at $ZEN_DATASET"
fi

# Check engine
if [ -d "$HANZO_ENGINE" ]; then
    echo "Hanzo Engine found"
    if [ -f "$HANZO_ENGINE/Cargo.toml" ]; then
        echo "  Engine version: $(grep '^version' "$HANZO_ENGINE/Cargo.toml" | head -1)"
    fi
else
    echo "Hanzo Engine not found at $HANZO_ENGINE"
fi

# Build optimized training binaries
echo "Building hanzo-training with release optimizations..."
cargo build --release --package hanzo-training --features "metal" 2>/dev/null || echo "Build failed (expected if metal feature unavailable)"

if [ -f "target/release/hanzo-train" ]; then
    echo "hanzo-train binary ready"
    ls -lh target/release/hanzo-train
fi

echo "Setup complete"
