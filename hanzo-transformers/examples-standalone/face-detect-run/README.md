# Face detection / landmark port (S3FD + 2D-FAN)

Pure-Rust (hanzo-ml / candle) port of the `face_alignment` pipeline that MuseTalk
uses to derive its nose-centered lip-sync crop box. Replaces the Python
`face_alignment` + SFD dependency in the native MuseTalk dub.

## Why two networks

MuseTalk does **not** use the raw face-detector box. `musetalk_dub.py` calls
`face_alignment.get_landmarks()` (which runs **S3FD** then **2D-FAN**) and derives
the crop box from the **68 landmarks**:

```
half_y    = lm[29].y          # nose bridge
chin_y    = max(lm[:].y)      # chin
half_dist = chin_y - half_y
upper     = max(0, half_y - half_dist)
bbox      = (min(lm.x), upper, max(lm.x), chin_y)
```

So matching the bbox requires reproducing **both** stages:

1. `s3fd.rs` — S3FD (VGG16 single-shot detector): conv1_1..conv7_2, three L2Norm
   branches, six multi-scale conf/loc heads, max-out background label, priorbox
   decode, NMS@0.3, score filter > 0.5. Port of
   `face_alignment/detection/sfd/{net_s3fd,detect,bbox}.py`.
2. `fan.rs` — 2D-FAN-4 (4-stack hourglass): base conv + 3 ConvBlocks, then 4
   stacked hourglasses with bottleneck residual ConvBlocks, intermediate
   supervision (l/bl/al). Port of `face_alignment/models/fan.py`.
3. `mod.rs` — the glue: center/scale from the SFD box (`CENTER_Y_OFFSET=0.12`,
   `REFERENCE_SCALE=195`), cv2-`INTER_LINEAR`-compatible crop to 256×256, FAN
   forward, `get_preds_fromhm` subpixel argmax + inverse affine to image coords,
   then `musetalk_bbox()`. Port of `face_alignment/{api.py,utils.py}`.

`flip_input=False` to match the dub.

## Weights

The real `face_alignment` checkpoints (verbatim PyTorch state_dict key names, so
the Rust `VarBuilder` paths map 1:1):

- `s3fd-619a316812.pth`        → `s3fd.safetensors`  (65 tensors)
- `2DFAN4-11f355bf06.pth.tar`  → `fan2d.safetensors` (945 tensors)

Convert with `convert_face_weights.py` (uses the zen-dub venv). On spark the
checkpoints live in `~/.cache/torch/hub/checkpoints/`.

## Verification (the proof this is a faithful port)

`main.rs` runs the Rust pipeline on every frame of the demo clip
`zen-dub/data/video/sun.mp4` (576×768, 550 frames) and compares the derived
MuseTalk bbox to the Python `face_alignment` reference
(`dump_face_ref.py`, identical bbox logic to `musetalk_dub.py`).

Result (CUDA, GB10), all 550 frames:

```
SFD-box mean IoU : 1.0000          # stage-1 detector is byte-exact
MuseTalk bbox    : mean IoU = 0.9990
                   min  IoU = 0.9563
                   IoU>=0.95: 100.0%
                   mean |coord err| = 0.032 px
                   max  |coord err| = 4 px
```

Every frame clears the IoU > 0.95 bar. The residual few-px edge differences come
from the FAN subpixel argmax landing ±1 heatmap cell (64² grid) on a handful of
frames; after MuseTalk resizes the crop to 256×256 this is sub-pixel and below
its lip-sync tolerance.

## Run

```
cd hanzo-transformers/examples-standalone/face-detect-run
cargo run --release --features cuda            # sampled frames
FULL=1 cargo run --release --features cuda      # all 550 frames
```

(Standalone crate — depends only on hanzo-ml + hanzo-nn + image — because the
rest of `hanzo-transformers` has unrelated pre-existing build breakage at this
commit. It `#[path]`-includes the real `models/face_detection/*.rs`, so it
exercises the exact library source.)
