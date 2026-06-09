#!/usr/bin/env python3
"""Convert s3fd-*.pth and 2DFAN4-*.pth.tar -> safetensors for the Rust port.
Keys are kept verbatim (PyTorch state_dict names) so the Rust VarBuilder paths
map 1:1. L2Norm 'weight' and BatchNorm buffers are preserved."""
import os, torch
from safetensors.torch import save_file

CKPT = os.path.expanduser("~/.cache/torch/hub/checkpoints")
OUT = os.path.expanduser("~/work/zen-dub-run/facedump")
os.makedirs(OUT, exist_ok=True)

def conv(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=False)
    # FAN .pth.tar is a plain OrderedDict of tensors; S3FD .pth too.
    if not isinstance(sd, dict) or "state_dict" in sd:
        sd = sd.get("state_dict", sd)
    out = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        out[k] = v.contiguous().to(torch.float32)
    save_file(out, dst)
    print(f"{os.path.basename(src)} -> {dst}: {len(out)} tensors")
    return list(out.keys())

s = conv(os.path.join(CKPT, "s3fd-619a316812.pth"), os.path.join(OUT, "s3fd.safetensors"))
f = conv(os.path.join(CKPT, "2DFAN4-11f355bf06.pth.tar"), os.path.join(OUT, "fan2d.safetensors"))
print("\nS3FD keys (all):")
for k in s: print(" ", k)
print("\nFAN keys (first 60):")
for k in f[:60]: print(" ", k)
print("...")
# print the hourglass module key shapes to learn the dynamic names
print("\nFAN hourglass/branch keys sample:")
for k in f:
    if any(t in k for t in ["m0.", "_4.", "_1.", "b2_plus"]):
        print(" ", k)
