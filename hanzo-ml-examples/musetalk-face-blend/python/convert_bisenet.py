#!/usr/bin/env python3
"""Convert MuseTalk BiSeNet face-parse weights (79999_iter.pth) -> safetensors for the hanzo
Rust face_parse module, and bake the fixed jaw-mode morphology kernels (cone+tail dilate kernel,
3x35 cheek erode kernel, 512x512 cheek protect-mask) for left/right_cheek_width=90. Drop
num_batches_tracked; keep NN names verbatim so the Rust VarBuilder paths match PyTorch.
"""
import torch, os, sys
import numpy as np
from safetensors.torch import save_file
sys.path.insert(0, os.path.expanduser('~/work/zen-dub-run/zen-dub'))
os.chdir(os.path.expanduser('~/work/zen-dub-run/zen-dub'))
_o=torch.load; torch.load=lambda *a,**k:(k.setdefault('weights_only',False),_o(*a,**k))[1]
from musetalk.utils.face_parsing import FaceParsing

SRC='/home/z/work/zen-dub-run/zen-dub/models/face-parse-bisent/79999_iter.pth'
OUT='/home/z/work/zen-dub-run/rustweights/bisenet.safetensors'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

sd=torch.load(SRC, map_location='cpu')
out={}; dropped=0
for k,v in sd.items():
    if k.endswith('num_batches_tracked'): dropped+=1; continue
    out[k]=v.detach().to(torch.float32).contiguous()

# bake morphology kernels (fixed for cheek_width=90)
fp=FaceParsing(left_cheek_width=90,right_cheek_width=90)
out['morph.cone_kernel']=torch.from_numpy(fp.kernel.astype(np.int8)).contiguous()       # 33x33
out['morph.cheek_kernel']=torch.from_numpy(fp.cheek_kernel.astype(np.int8)).contiguous() # 3x35
out['morph.cheek_mask']=torch.from_numpy((fp.cheek_mask//255).astype(np.int8)).contiguous() # 512x512 {0,1}
print(f'[bisenet] {len(out)} tensors (dropped {dropped} nbt), baked cone{tuple(fp.kernel.shape)} '
      f'cheek{tuple(fp.cheek_kernel.shape)} mask{tuple(fp.cheek_mask.shape)}')
save_file(out, OUT, metadata={'format':'pt'})
print('done ->', OUT)
