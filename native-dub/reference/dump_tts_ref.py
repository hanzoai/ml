#!/usr/bin/env python3
"""Dump zen-3-TTS PyTorch reference tensors for the engine `codec_validation` cargo tests.

Runs the *real* QwenLM/Qwen3-TTS PyTorch model (`modeling_qwen3_tts.py` +
`modeling_qwen3_tts_tokenizer_v2.py`) on a fixed prompt with a fixed seed and a fully
deterministic (greedy) decode, and writes the exact per-stage tensors the four tests in
`hanzo-engine/src/speech_models/qwen3_tts/mod.rs::codec_validation` consume.

The tests read **raw little-endian** binary (NOT .npy):
  f32 files via 4-byte chunks, i64 files via 8-byte chunks.

Produced files (default OUT=~/work/zen-dub-run/tts-ref):
  # talker / prefill (test: prefill_matches_reference, talker_matches_reference)
  input_ids.i64        (T_ids,)            the tokenized prompt ids
  tk_prefill.f32       (1, T, H) flat      PyTorch talker prefill inputs_embeds
  tk_hidden.f32        (1, T, H) flat      talker.model(...).last_hidden_state (post-norm)
  tk_logits.f32        (1, T, V) flat      talker.codec_head(hidden)
  tk_frame0_codes.i64  (G,)                greedy frame-0 codes (codebook 0..G-1)
  # full greedy generation (test: full_generation_greedy_matches_reference)
  greedy_TQ.i64        (T_gen, G) row-major  greedy codes, codebook-0 == col 0
  # codec decode (test: codec_matches_reference)
  codes_QT.i64         (Q, T_gen) row-major  the SAME greedy codes, transposed for the codec
  ref_quant.f32        (1, codebook_dim, T_gen) flat   SplitRVQ.decode output
  ref_pretrans.f32     (1, T_gen, latent_dim) flat     pre_conv(quant).transpose + pre_transformer? NO:
                                                        == pre_conv(quant).transpose(1,2)  (matches Rust pc_t)
  ref_pretrans_out.f32 (1, T_gen, latent_dim) flat     pre_transformer(pc_t).last_hidden_state
  ref_upsample.f32     (1, latent_dim, T_gen*up) flat  post-upsample (matches Rust `up`)
  ref_wav.f32          (1, 1, samples) flat            final clamped waveform
  meta.txt             shapes + scalars + the env block to export

Stage boundaries mirror Qwen3TTSTokenizerV2Decoder.forward and the Rust CodecDecoder::decode_debug:
  quant = quantizer.decode(codes)                       -> (b, codebook_dim, t)
  pc_t  = pre_conv(quant).transpose(1,2)                -> (b, t, latent_dim)
  pt    = pre_transformer(inputs_embeds=pc_t).last...   -> (b, t, latent_dim)
  up    = permute->upsample blocks                      -> (b, latent_dim, t*up)
  wav   = decoder blocks -> clamp(-1,1)                 -> (b, 1, samples)

Everything runs in float32 on the requested device (codec decode is f32 in the Rust too).
The talker/code-predictor greedy decode here is reimplemented directly on the HF modules (no
GenerationMixin sampling machinery) so it is bit-deterministic and matches the Rust loop exactly:
argmax codebook-0 (after the suppress_tokens mask), then argmax groups 1..G via the sub-talker,
feed back sum(group embeds)+trailing_pad, stop on codec_eos.
"""
import importlib.machinery
import os
import sys
import types

import numpy as np
import torch

# ----------------------------------------------------------------------------- config (env)
QWEN_REPO = os.path.expanduser(os.environ.get("QWEN3_TTS_REPO", "~/work/zen/repos/Qwen3-TTS"))
MODEL = os.path.expanduser(os.environ.get("ZEN3_TTS_MODEL", "~/work/zen/hf/zen-3-tts-0.6B"))
CODEC_DIR = os.path.join(MODEL, "speech_tokenizer")
OUT = os.path.expanduser(os.environ.get("ZEN3_REF_OUT", "~/work/zen-dub-run/tts-ref"))
TEXT = os.environ.get("ZEN3_REF_TEXT", "Everyone becomes calm with age and lives naturally.")
LANGUAGE = os.environ.get("ZEN3_REF_LANG", "english")
SEED = int(os.environ.get("ZEN3_REF_SEED", "0"))
MAX_FRAMES = int(os.environ.get("ZEN3_REF_MAX_FRAMES", "48"))
DEVICE = os.environ.get("ZEN3_REF_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUT, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
dev = torch.device(DEVICE)
DT = torch.float32

# ----------------------------------------------------------------------------- import qwen_tts
# The package __init__ chain imports the 25Hz tokenizer (onnxruntime/torchaudio/sox), which we
# don't need. Register namespace-package stubs so relative imports resolve without running it.
sys.path.insert(0, QWEN_REPO)


def _stub_pkg(name, relpath):
    m = types.ModuleType(name)
    p = os.path.join(QWEN_REPO, relpath)
    m.__path__ = [p]
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [p]
    m.__spec__ = spec
    sys.modules[name] = m
    return m


_stub_pkg("qwen_tts", "qwen_tts")
_core = _stub_pkg("qwen_tts.core", "qwen_tts/core")
_stub_pkg("qwen_tts.inference", "qwen_tts/inference")
_stub_pkg("qwen_tts.core.models", "qwen_tts/core/models")
_stub_pkg("qwen_tts.core.tokenizer_12hz", "qwen_tts/core/tokenizer_12hz")

from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (  # noqa: E402
    Qwen3TTSTokenizerV2Config,
)
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (  # noqa: E402
    Qwen3TTSTokenizerV2Model,
)

# expose the V2 symbols on the `core` stub (the inference tokenizer imports them from `..core`);
# V1 is stubbed out so its onnxruntime/torchaudio deps are never touched.
_core.Qwen3TTSTokenizerV2Config = Qwen3TTSTokenizerV2Config
_core.Qwen3TTSTokenizerV2Model = Qwen3TTSTokenizerV2Model
_core.Qwen3TTSTokenizerV1Config = type("Qwen3TTSTokenizerV1Config", (), {})
_core.Qwen3TTSTokenizerV1Model = type("Qwen3TTSTokenizerV1Model", (), {})

from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig  # noqa: E402
from qwen_tts.core.models.modeling_qwen3_tts import (  # noqa: E402
    Qwen3TTSForConditionalGeneration,
)
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


# ----------------------------------------------------------------------------- io helpers
def save_f32(name, t):
    if isinstance(t, torch.Tensor):
        a = t.detach().to("cpu", torch.float32).contiguous().numpy()
    else:
        a = np.ascontiguousarray(t, dtype=np.float32)
    a.astype("<f4").tofile(os.path.join(OUT, name))
    print(f"  saved {name:24s} {tuple(a.shape)}  ({a.size} f32)")
    return a


def save_i64(name, t):
    if isinstance(t, torch.Tensor):
        a = t.detach().to("cpu", torch.int64).contiguous().numpy()
    else:
        a = np.ascontiguousarray(t, dtype=np.int64)
    a.astype("<i8").tofile(os.path.join(OUT, name))
    print(f"  saved {name:24s} {tuple(a.shape)}  ({a.size} i64)")
    return a


# ----------------------------------------------------------------------------- load
print(f"[tts-ref] device={dev} model={MODEL}")
print(f"[tts-ref] text={TEXT!r} lang={LANGUAGE} seed={SEED}")

# Construct + load weights from safetensors directly, bypassing the custom from_pretrained (which
# would also try to load/instantiate the inference speech-tokenizer wrapper, dragging in the 25Hz
# onnxruntime/torchaudio path). We load the codec separately below.
config = Qwen3TTSConfig.from_pretrained(MODEL)
config._attn_implementation = "eager"
model = Qwen3TTSForConditionalGeneration(config)
_sd = load_file(os.path.join(MODEL, "model.safetensors"))
_missing, _unexpected = model.load_state_dict(_sd, strict=False)
_missing = [k for k in _missing if not k.startswith("speaker_encoder")]
assert not _missing, f"missing talker weights: {_missing[:8]}"
print(f"[tts-ref] main model loaded ({len(_sd)} tensors; {len(_unexpected)} unexpected ignored)")
model = model.to(dev, DT).eval()

codec_config = Qwen3TTSTokenizerV2Config.from_pretrained(CODEC_DIR)
codec_config._attn_implementation = "eager"
codec = Qwen3TTSTokenizerV2Model(codec_config)
_csd = load_file(os.path.join(CODEC_DIR, "model.safetensors"))
_cmiss, _cunexp = codec.load_state_dict(_csd, strict=False)
# the encoder half is unused for decode; only assert the decoder is fully populated
_cmiss_dec = [k for k in _cmiss if k.startswith("decoder.")]
assert not _cmiss_dec, f"missing codec decoder weights: {_cmiss_dec[:8]}"
print(f"[tts-ref] codec loaded ({len(_csd)} tensors; {len(_cmiss)} missing[{len(_cmiss_dec)} dec], {len(_cunexp)} unexpected)")
codec = codec.to(dev, DT).eval()

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

tc = config.talker_config
G = tc.num_code_groups
codec_eos = tc.codec_eos_token_id
talker = model.talker  # Qwen3TTSTalkerForConditionalGeneration
cp = talker.code_predictor  # Qwen3TTSTalkerCodePredictorModelForConditionalGeneration


def text_proj(ids):
    return talker.text_projection(talker.get_text_embeddings()(ids))


def codec_emb(ids):
    return talker.get_input_embeddings()(ids)


# language id (english -> 2050)
lang_id = tc.codec_language_id[LANGUAGE.lower()]

# ----------------------------------------------------------------------------- tokenize prompt
prompt = f"<|im_start|>assistant\n{TEXT}<|im_end|>\n<|im_start|>assistant\n"
input_ids = tok(prompt, return_tensors="pt")["input_ids"].to(dev)  # (1, T_ids)
T_ids = input_ids.shape[1]
assert T_ids >= 9, f"prompt too short ({T_ids})"
save_i64("input_ids.i64", input_ids[0])

# ----------------------------------------------------------------------------- build prefill
# Mirrors Qwen3TTSForConditionalGeneration.generate (non_streaming_mode=True, speaker=None,
# voice_clone=None) and the Rust Qwen3TtsPipeline::build_prefill.
with torch.no_grad():
    tts_bos_e, tts_eos_e, tts_pad_e = text_proj(
        torch.tensor(
            [[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]],
            device=dev,
        )
    ).chunk(3, dim=1)  # 3 * (1,1,H)

    codec_prefill_0 = torch.tensor(
        [[tc.codec_think_id, tc.codec_think_bos_id, lang_id, tc.codec_think_eos_id]], device=dev
    )
    codec_prefill_1 = torch.tensor([[tc.codec_pad_id, tc.codec_bos_id]], device=dev)
    codec_input_embedding = torch.cat(
        [codec_emb(codec_prefill_0), codec_emb(codec_prefill_1)], dim=1
    )  # (1, n, H)
    n = codec_input_embedding.shape[1]

    role = text_proj(input_ids[:, :3])
    body = (
        torch.cat([tts_pad_e.expand(-1, n - 2, -1), tts_bos_e], dim=1)
        + codec_input_embedding[:, :-1]
    )
    prefill = torch.cat([role, body], dim=1)

    # non-streaming chunk: text_proj(text[3:-5]) ++ tts_eos  +  codec_pad*(len+1) ; then tts_pad+codec_bos
    body_text_ids = input_ids[:, 3:-5]
    text_with_eos = torch.cat([text_proj(body_text_ids), tts_eos_e], dim=1)
    pad_count = body_text_ids.shape[1] + 1
    codec_pad_ids = torch.tensor([[tc.codec_pad_id] * pad_count], device=dev)
    chunk_a = text_with_eos + codec_emb(codec_pad_ids)
    chunk_b = tts_pad_e + codec_emb(torch.tensor([[tc.codec_bos_id]], device=dev))
    prefill = torch.cat([prefill, chunk_a, chunk_b], dim=1)  # (1, T, H)

T = prefill.shape[1]
H = prefill.shape[2]
trailing_pad = tts_pad_e  # non-streaming trailing_text_hidden
save_f32("tk_prefill.f32", prefill)
print(f"[tts-ref] prefill T={T} H={H} (n={n}, body_text={body_text_ids.shape[1]})")

# ----------------------------------------------------------------------------- talker forward (frame 0)
suppress = [
    i for i in range(tc.vocab_size - 1024, tc.vocab_size) if i != codec_eos
]


def talker_forward(embeds):
    """Full-sequence talker forward, no cache -> (hidden post-norm, codec_head logits)."""
    out = talker.model(inputs_embeds=embeds, use_cache=False, output_hidden_states=False)
    hidden = out.last_hidden_state
    logits = talker.codec_head(hidden)
    return hidden, logits


def sub_talker_groups(last_hidden, code0):
    """Greedy groups 1..G-1 via the sub-talker, recomputing the sequence each step (matches Rust)."""
    codes = [int(code0)]
    seq = torch.cat([last_hidden, codec_emb(torch.tensor([[code0]], device=dev))], dim=1)
    for group in range(1, G):
        out = cp.model(inputs_embeds=seq, use_cache=False)
        h = out.last_hidden_state
        logit = cp.lm_head[group - 1](h[:, -1, :])  # (1, V)
        idx = int(logit.argmax(-1).item())
        codes.append(idx)
        if group < G - 1:
            emb = cp.model.get_input_embeddings()[group - 1](torch.tensor([[idx]], device=dev))
            seq = torch.cat([seq, emb], dim=1)
    return codes


with torch.no_grad():
    hidden, logits = talker_forward(prefill)
    save_f32("tk_hidden.f32", hidden)
    save_f32("tk_logits.f32", logits)

    last_logits = logits[0, -1, :].clone()
    last_logits[suppress] = float("-inf")
    code0 = int(last_logits.argmax(-1).item())
    frame0 = sub_talker_groups(hidden[:, -1:, :], code0)
    save_i64("tk_frame0_codes.i64", torch.tensor(frame0))
    print(f"[tts-ref] frame0 greedy codes: {frame0}")

# ----------------------------------------------------------------------------- full greedy generation
# Autoregressive greedy loop; identical structure to Rust generate_codes with temperature 0.
with torch.no_grad():
    embeds = prefill.clone()
    all_frames = []
    for step in range(MAX_FRAMES):
        h, lg = talker_forward(embeds)
        ll = lg[0, -1, :].clone()
        ll[suppress] = float("-inf")
        c0 = int(ll.argmax(-1).item())
        if c0 == codec_eos:
            break
        frame = sub_talker_groups(h[:, -1:, :], c0)
        all_frames.append(frame)
        # next input = sum of all group embeddings + trailing_pad
        summed = codec_emb(torch.tensor([[c0]], device=dev))
        for group in range(1, G):
            summed = summed + cp.model.get_input_embeddings()[group - 1](
                torch.tensor([[frame[group]]], device=dev)
            )
        nxt = summed + trailing_pad
        embeds = torch.cat([embeds, nxt], dim=1)

T_gen = len(all_frames)
assert T_gen > 0, "greedy generation produced 0 frames"
codes_TQ = torch.tensor(all_frames, dtype=torch.int64)  # (T_gen, G)
save_i64("greedy_TQ.i64", codes_TQ)  # row-major (T, Q); codebook-0 == column 0
codes_QT = codes_TQ.t().contiguous()  # (Q, T_gen)
save_i64("codes_QT.i64", codes_QT)
print(f"[tts-ref] greedy generated T_gen={T_gen} frames; codebook-0[:8]={codes_TQ[:8, 0].tolist()}")

# ----------------------------------------------------------------------------- codec decode stages
# codes for the codec: (b, Q, T). Mirrors Qwen3TTSTokenizerV2Decoder.forward / Rust decode_debug.
dec = codec.decoder
codes_in = codes_QT.unsqueeze(0).to(dev)  # (1, Q, T)
codes_in = torch.clamp(codes_in, min=0)

with torch.no_grad():
    quant = dec.quantizer.decode(codes_in)  # (1, codebook_dim, T)
    pc = dec.pre_conv(quant)  # (1, latent_dim, T)
    pc_t = pc.transpose(1, 2)  # (1, T, latent_dim)
    pt = dec.pre_transformer(inputs_embeds=pc_t).last_hidden_state  # (1, T, latent_dim)
    h = pt.permute(0, 2, 1)  # (1, latent_dim, T)
    for blocks in dec.upsample:
        for block in blocks:
            h = block(h)
    up = h.clone()  # (1, latent_dim, T*up)
    for block in dec.decoder:
        h = block(h)
    wav = h.clamp(min=-1, max=1)  # (1, 1, samples)

# NOTE on the Rust env mapping: decode_debug returns (quant, pc_t, pt, up, wav) but the test binds
#   let (quant, _pc, pretrans, up, wav) = decode_debug(...)
# i.e. it DISCARDS pc_t and checks ZEN3_REF_PRETRANS against `pt` (the pre_transformer OUTPUT).
# So ref_pretrans.f32 == pt, and pc_t is dumped separately for debugging only.
save_f32("ref_quant.f32", quant)
save_f32("ref_pretrans.f32", pt)
save_f32("ref_pretrans_out.f32", pt)  # alias; == pt
save_f32("ref_preconv.f32", pc_t)  # debug only (Rust discards pc_t / `_pc`)
save_f32("ref_upsample.f32", up)
save_f32("ref_wav.f32", wav)

# sanity: full-decoder path should equal the stage chain
with torch.no_grad():
    wav_full = dec(codes_in)
max_wav_diff = float((wav - wav_full).abs().max().item())
print(f"[tts-ref] stagewise-vs-forward wav max_abs_diff={max_wav_diff:.3e}")

# ----------------------------------------------------------------------------- meta + env
codebook_dim = codec_config.decoder_config.codebook_dim
latent_dim = codec_config.decoder_config.latent_dim
total_up = int(np.prod(codec_config.decoder_config.upsampling_ratios))
samples = wav.shape[-1]
meta = f"""# zen-3-tts reference dump
model={MODEL}
text={TEXT!r}
language={LANGUAGE} lang_id={lang_id}
seed={SEED} device={DEVICE}
T_ids={T_ids}
talker T={T} H={H} V={tc.vocab_size}
num_code_groups G={G}
T_gen={T_gen} Q={G}
codebook_dim={codebook_dim} latent_dim={latent_dim} upsample_ratio_product={total_up}
wav_samples={samples} sample_rate={codec_config.output_sample_rate}
stagewise_vs_forward_wav_max_abs_diff={max_wav_diff:.6e}

# ---- export this to enable the engine codec_validation cargo tests ----
export ZEN3_MAIN_WEIGHTS={MODEL}/model.safetensors
export ZEN3_MAIN_CONFIG={MODEL}/config.json
export ZEN3_CODEC_WEIGHTS={CODEC_DIR}/model.safetensors
export ZEN3_CODEC_CONFIG={CODEC_DIR}/config.json
export ZEN3_INPUT_IDS={OUT}/input_ids.i64
export ZEN3_LANG_ID={lang_id}
export ZEN3_TK_PREFILL={OUT}/tk_prefill.f32
export ZEN3_TK_T={T}
export ZEN3_TK_HIDDEN={OUT}/tk_hidden.f32
export ZEN3_TK_LOGITS={OUT}/tk_logits.f32
export ZEN3_TK_FRAME0={OUT}/tk_frame0_codes.i64
export ZEN3_GREEDY_TQ={OUT}/greedy_TQ.i64
export ZEN3_GREEDY_T={T_gen}
export ZEN3_GREEDY_Q={G}
export ZEN3_CODES_QT={OUT}/codes_QT.i64
export ZEN3_T={T_gen}
export ZEN3_Q={G}
export ZEN3_REF_QUANT={OUT}/ref_quant.f32
export ZEN3_REF_PRETRANS={OUT}/ref_pretrans.f32
export ZEN3_REF_UPSAMPLE={OUT}/ref_upsample.f32
export ZEN3_REF_WAV={OUT}/ref_wav.f32
export ZEN3_PREFILL_CHECK=1
export ZEN3_FULLGEN_CHECK=1
"""
with open(os.path.join(OUT, "meta.txt"), "w") as f:
    f.write(meta)
print("[tts-ref] meta.txt + env written ->", os.path.join(OUT, "meta.txt"))
print("[tts-ref] DONE ->", OUT)
