#!/usr/bin/env bash
# Regenerate the committed pre-compiled SPIR-V from the GLSL sources.
# Run on any host with `glslc` (shaderc) whenever a *.comp changes; commit the
# resulting src/vulkan/spv/*.spv so glslc-less hosts (e.g. native Windows) can
# still build the `vulkan` feature.
set -euo pipefail
cd "$(dirname "$0")"
glslc="${GLSLC:-glslc}"
mkdir -p spv
n=0
for f in shaders/*.comp; do
  stem="$(basename "$f" .comp)"
  "$glslc" --target-env=vulkan1.2 "$f" -o "spv/$stem.spv"
  n=$((n+1))
done
echo "regenerated $n SPIR-V modules into spv/"
