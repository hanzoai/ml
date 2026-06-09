#!/usr/bin/env python3
"""Dump the FULL MuseTalk whisper audio-feature pipeline intermediates for Rust parity.

Saves to ~/work/zen-dub-run/refdump/wf/:
  mel.npy                 [1, 80, 3000]   HF WhisperFeatureExtractor.input_features (one 30s segment)
  hidden_stack.npy        [1, T, 5, 384]  torch.stack(encoder.hidden_states, dim=2)  (full seq, pre-chunk)
  whisper_feature.npy     [1, Tpad, 5, 384] after trim-to-actual + zero pad (the array sliced per-frame)
  chunk_raw_{i}.npy       [1, 50, 384]    per-frame chunk (rearranged 10x5 -> 50), pre-PE, for i in idxs
  chunk_pe_{i}.npy        [1, 50, 384]    + PositionalEncoding, the UNet cross-attn context
  meta.txt                shapes + scalars (num_frames, actual_length, padding_nums, lib_len)

Uses float32 throughout (the native dub runs the CV models in float32).
"""
import os, sys, math
import numpy as np, torch

os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
_orig = torch.load
torch.load = lambda *a, **k: (k.setdefault("weights_only", False), _orig(*a, **k))[1]

REPO = os.path.expanduser("~/work/zen-dub-run/zen-dub")
sys.path.insert(0, REPO); os.chdir(REPO)
from musetalk.models.unet import PositionalEncoding
from musetalk.utils.audio_processor import AudioProcessor
from transformers import WhisperModel

OUT = os.path.expanduser("~/work/zen-dub-run/refdump/wf")
os.makedirs(OUT, exist_ok=True)
M = os.path.join(REPO, "models")
AUDIO = os.path.expanduser("~/work/zen-dub-run/clip/tts_en.wav")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wd = torch.float32

def sv(n, t):
    a = t.detach().cpu().float().numpy() if isinstance(t, torch.Tensor) else np.asarray(t, np.float32)
    np.save(os.path.join(OUT, n), a)
    print(" saved", n, a.shape)

print(f"[wf] dev={dev} audio={AUDIO}")
ap = AudioProcessor(feature_extractor_path=os.path.join(M, "whisper"))
whisper = WhisperModel.from_pretrained(os.path.join(M, "whisper")).to(device=dev, dtype=wd).eval()
whisper.requires_grad_(False)
pe = PositionalEncoding(d_model=384).to(dev)

# --- 1) mel features (list of [1,80,3000] per 30s segment) ---
feats, lib_len = ap.get_audio_feature(AUDIO)
print(f"[wf] segments={len(feats)} lib_len={lib_len}")
sv("mel.npy", feats[0].to(wd))   # first 30s segment mel

# --- 2) replicate get_whisper_chunk but capture intermediates ---
fps = 25
audio_padding_length_left = 2
audio_padding_length_right = 2
audio_feature_length_per_frame = 2 * (audio_padding_length_left + audio_padding_length_right + 1)  # 10
whisper_feature = []
for input_feature in feats:
    input_feature = input_feature.to(dev).to(wd)
    audio_feats = whisper.encoder(input_feature, output_hidden_states=True).hidden_states
    print(f"[wf] num hidden_states={len(audio_feats)} each {tuple(audio_feats[0].shape)}")
    audio_feats = torch.stack(audio_feats, dim=2)   # [1, T, 5, 384]
    whisper_feature.append(audio_feats)
hidden_stack = torch.cat(whisper_feature, dim=1)
sv("hidden_stack.npy", hidden_stack)

sr = 16000; audio_fps = 50; fps = int(fps)
whisper_idx_multiplier = audio_fps / fps
num_frames = math.floor((lib_len / sr) * fps)
actual_length = math.floor((lib_len / sr) * audio_fps)
wf = hidden_stack[:, :actual_length, ...]
padding_nums = math.ceil(whisper_idx_multiplier)
wf = torch.cat([
    torch.zeros_like(wf[:, :padding_nums * audio_padding_length_left]),
    wf,
    torch.zeros_like(wf[:, :padding_nums * 3 * audio_padding_length_right]),
], 1)
sv("whisper_feature.npy", wf)  # [1, Tpad, 5, 384] -- the array indexed per-frame

# also run the real fn to be 100% sure our chunks match the library
chunks = ap.get_whisper_chunk(feats, dev, wd, whisper, lib_len, fps=25,
                              audio_padding_length_left=2, audio_padding_length_right=2)
print(f"[wf] official chunks {tuple(chunks.shape)}")
sv("chunks_all.npy", chunks)   # [num_frames, 50, 384]

idxs = [0, 1, 5, 30, min(num_frames - 1, chunks.shape[0]-1)]
for i in idxs:
    wb = chunks[i:i+1].to(dev).to(wd)     # [1,50,384]
    sv(f"chunk_raw_{i}.npy", wb)
    sv(f"chunk_pe_{i}.npy", pe(wb))

with open(os.path.join(OUT, "meta.txt"), "w") as f:
    f.write(f"lib_len={lib_len}\n")
    f.write(f"sr={sr} audio_fps={audio_fps} fps={fps}\n")
    f.write(f"whisper_idx_multiplier={whisper_idx_multiplier}\n")
    f.write(f"num_frames={num_frames}\n")
    f.write(f"actual_length={actual_length}\n")
    f.write(f"padding_nums={padding_nums}\n")
    f.write(f"audio_feature_length_per_frame={audio_feature_length_per_frame}\n")
    f.write(f"hidden_stack_shape={tuple(hidden_stack.shape)}\n")
    f.write(f"whisper_feature_shape={tuple(wf.shape)}\n")
    f.write(f"chunks_shape={tuple(chunks.shape)}\n")
    f.write(f"mel_shape={tuple(feats[0].shape)}\n")
    f.write(f"idxs={idxs}\n")
print("[wf] DONE ->", OUT)
