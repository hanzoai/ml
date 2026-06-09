# MuseTalk BiSeNet face-parse + feathered blend — native Rust (hanzo-ml)

Ports MuseTalk's seam-free mouth paste-back to Rust, eliminating the Python BiSeNet
(`torch` face-parsing) dependency from the native dub. After the UNet re-renders the mouth
crop, MuseTalk pastes it into the full frame using a **BiSeNet face-parsing mask** + a
**feathered (Gaussian-blurred) alpha blend** — the `utils/blending.py::get_image` path. This
crate reimplements both in hanzo-ml so the whole post-processing runs without Python.

## What's here

- `src/face_parse.rs` — **BiSeNet** (the face-parsing segmentation network), ported to hanzo-ml:
  ResNet-18 context path (feat8/16/32) + AttentionRefinementModule(16/32) + global-avg branch,
  FeatureFusionModule(256,256), and the three `BiSeNetOutput` 19-class seg heads with bilinear
  upsample (`align_corners=true`). Runs on **CUDA** (or CPU). Weight names match the converted
  checkpoint verbatim (PyTorch module paths).
- `src/blend.rs` — the `get_image` blend, as CPU image ops mirroring cv2/PIL exactly:
  jaw-mode morphology (binary dilate with the cone+tail kernel, 2x erode with the 3x35 cheek
  kernel, cheek-protect masking), bilinear mask resize, crop/paste, upper-boundary cut,
  `cv2.GaussianBlur` (sigma auto-derived from ksize, BORDER_REFLECT_101), and the final
  alpha-feathered `PIL.Image.paste` composite.
- `src/verify_excerpt.rs` — the `face-blend` verify harness (mask IoU + composite PSNR/SSIM vs
  the PyTorch reference dumps). Wired into `musetalk-bench` as the `face-blend` subcommand.
- `src/layers.rs` — shared layer constructors incl. the `batch_norm` helper added for BiSeNet.
- `python/convert_bisenet.py` — converts `79999_iter.pth` -> `bisenet.safetensors` (drops
  `num_batches_tracked`; bakes the fixed jaw-morphology kernels for `cheek_width=90` as f32).
- `python/dump_blend_ref.py` — dumps the PyTorch reference tensors (BiSeNet input, argmax parse,
  jaw mask, feathered alpha, rendered crop, and the proven `get_image` composite) per frame.

## Weights

Real MuseTalk BiSeNet face-parse weights (`models/face-parse-bisent/79999_iter.pth`,
`resnet18-5c106cde.pth` already merged in). 160 NN tensors + 3 baked morphology kernels ->
`rustweights/bisenet.safetensors`.

## Verification (vs the proven Python blend at `zen-dub-run/musetalk_dub.py`)

5 demo frames (576x768), Rust BiSeNet on CUDA + CPU blend vs PyTorch `get_image`:

| dtype  | mask IoU (net-isolated) | mask IoU (full Rust path) | alpha PSNR | composite PSNR | composite SSIM |
|--------|-------------------------|---------------------------|-----------|----------------|----------------|
| f32 (CUDA) | 0.99999             | 0.99924                   | 53.47 dB  | 77.12 dB       | 1.00000        |
| f16 (CUDA) | 0.99984             | 0.99926                   | 53.47 dB  | 77.11 dB       | 1.00000        |
| f32 (CPU)  | 0.99999             | 0.99924                   | 53.47 dB  | 77.12 dB       | 1.00000        |

Per-frame composite vs Python: **max |diff| = 3-4 / 255**, mean |diff| ~= 0.001, only
~0.18-0.20% of pixels differ at all (by <=1 gray level, at the mask boundary). Jaw mask: 26-74
of 262,144 pixels disagree, all on the feathered edge. **No seams** — the 20x-amplified diff is
black save a faint mask-boundary trace.

"net-isolated" feeds the Rust BiSeNet the exact PyTorch-normalized 512x512 input (isolates the
network port); "full Rust path" runs the Rust resize+ImageNet-normalize too. The tiny full-path
gap (~0.999) is the bilinear-vs-PIL-bicubic mask downscale, smoothed out by the Gaussian feather.

## Run

```sh
# convert weights + dump PyTorch refs (one-time)
python python/convert_bisenet.py
python python/dump_blend_ref.py

# build (CUDA include path needed for the quant kernels)
CPATH=/usr/local/cuda/include CARGO_TARGET_DIR=target-blend \
  cargo build --release --features cuda

# verify: mask IoU + composite PSNR/SSIM vs PyTorch
MUSETALK_DEV=cuda MUSETALK_DTYPE=f16 ./musetalk-bench face-blend
```

The network is `face_parse::BiSeNet::{forward_logits, parse}`; the blend is the free functions in
`blend` (`jaw_mask`, `feather_alpha`, `composite`). Together they replace the Python BiSeNet +
`get_image` so the native dub's paste-back is fully Rust.
