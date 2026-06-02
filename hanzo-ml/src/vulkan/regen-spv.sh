#!/usr/bin/env bash
# Regenerate the committed pre-compiled SPIR-V from the GLSL sources.
# Run on any host with `glslc` (shaderc) -- or, failing that, `glslangValidator`
# (the `glslang-tools` package) -- whenever a *.comp changes; commit the resulting
# src/vulkan/spv/*.spv so compiler-less hosts (e.g. native Windows) can still build
# the `vulkan` feature. Mirrors the compiler selection in ../../build.rs.
set -euo pipefail
cd "$(dirname "$0")"
glslc="${GLSLC:-glslc}"
glslang="${GLSLANG:-glslangValidator}"
mkdir -p spv

compile() { # $1=src $2=dst
  if command -v "$glslc" >/dev/null 2>&1; then
    "$glslc" --target-env=vulkan1.2 "$1" -o "$2"
  elif command -v "$glslang" >/dev/null 2>&1; then
    "$glslang" --target-env vulkan1.2 -V "$1" -o "$2"
  else
    echo "error: neither $glslc nor $glslang found" >&2
    exit 1
  fi
}

n=0
for f in shaders/*.comp; do
  stem="$(basename "$f" .comp)"
  compile "$f" "spv/$stem.spv"
  n=$((n+1))
done
echo "regenerated $n SPIR-V modules into spv/"
