//! Whisper-tiny audio-feature stage for MuseTalk, ported to Rust (hanzo-ml).
//!
//! Reproduces MuseTalk's exact audio pipeline (utils/audio_processor.py + the HF whisper-tiny
//! ENCODER) with no Python dependency:
//!   wav -> log-mel (HF WhisperFeatureExtractor) -> whisper-tiny encoder (5 hidden states stacked)
//!   -> per-frame 5-window x 10-context chunk -> PositionalEncoding -> [num_frames, 50, 384].
//!
//! The encoder is the standard (non-causal) OpenAI/HF Whisper audio encoder:
//!   conv1(k3,s1,p1)+GELU, conv2(k3,s2,p1)+GELU, +learned absolute position embedding,
//!   N pre-LayerNorm self-attention + GELU-MLP blocks, final LayerNorm.
//! For whisper-tiny: d_model=384, 4 layers, 6 heads, ffn=1536, 80 mel bins.
//!
//! Hidden-states semantics (matches `WhisperModel.encoder(..., output_hidden_states=True)`):
//!   hs[0] = tensor after conv stack + position embedding (the block *input*),
//!   hs[i] = output of encoder block i (i=1..=N), i.e. N+1 = 5 tensors for tiny.
//!   These are stacked on a new axis-2 -> [B, T, 5, 384].

use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_nn::ops::softmax_last_dim;
use hanzo_nn::{Conv1d, Conv1dConfig, LayerNorm, Linear, Module};
use hanzo_quant::ShardedVarBuilder;

// Self-contained Linear builders over a ShardedVarBuilder (mirrors crate::layers, kept local so
// this module has no coupling to the rest of the bench / can be lifted into hanzo-transformers).
fn linear(in_dim: usize, out_dim: usize, vb: ShardedVarBuilder) -> Result<Linear> {
    let ws = vb.get((out_dim, in_dim), "weight")?;
    let bs = vb.get(out_dim, "bias")?;
    Ok(Linear::new(ws, Some(bs)))
}
fn linear_no_bias(in_dim: usize, out_dim: usize, vb: ShardedVarBuilder) -> Result<Linear> {
    let ws = vb.get((out_dim, in_dim), "weight")?;
    Ok(Linear::new(ws, None))
}

/// Whisper-tiny encoder config (the only variant MuseTalk uses).
#[derive(Debug, Clone)]
pub struct WhisperConfig {
    pub num_mel_bins: usize,
    pub max_source_positions: usize,
    pub d_model: usize,
    pub encoder_attention_heads: usize,
    pub encoder_layers: usize,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        // openai/whisper-tiny
        Self {
            num_mel_bins: 80,
            max_source_positions: 1500,
            d_model: 384,
            encoder_attention_heads: 6,
            encoder_layers: 4,
        }
    }
}

// ---- mel constants (HF WhisperFeatureExtractor for whisper-tiny) ----
pub const SAMPLE_RATE: usize = 16000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const CHUNK_LENGTH: usize = 30;
pub const N_SAMPLES: usize = CHUNK_LENGTH * SAMPLE_RATE; // 480000
pub const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH; // 3000
pub const N_FREQ: usize = N_FFT / 2 + 1; // 201

// ===========================================================================
// Mel spectrogram (numerically identical to HF WhisperFeatureExtractor)
// ===========================================================================

/// Periodic Hann window of length `n` (== torch.hann_window(n), periodic=True).
fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (std::f64::consts::PI * i as f64 / n as f64).sin();
            (x * x) as f32
        })
        .collect()
}

/// Reflect-pad a 1-D signal by `pad` on each side (numpy/torch "reflect": edge sample not repeated).
fn reflect_pad(x: &[f32], pad: usize) -> Vec<f32> {
    let n = x.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    for i in 0..pad {
        out.push(x[pad - i]); // reflect without repeating x[0]
    }
    out.extend_from_slice(x);
    for i in 0..pad {
        out.push(x[n - 2 - i]); // reflect without repeating x[n-1]
    }
    out
}

/// Compute the HF log-mel spectrogram for one 30s segment.
///
/// `mel_filters` is the HF filterbank flattened row-major as [N_FREQ, num_mel] (i.e. shape
/// [201, 80] in PyTorch). Returns a `[num_mel, N_FRAMES]` tensor on `device` (f32), matching
/// `WhisperFeatureExtractor(...).input_features[0]`.
pub fn log_mel_spectrogram(
    samples: &[f32],
    mel_filters: &Tensor, // [N_FREQ, num_mel] f32
    cfg: &WhisperConfig,
    device: &Device,
) -> Result<Tensor> {
    // 1) pad/truncate to exactly N_SAMPLES (30s)
    let mut seg = vec![0f32; N_SAMPLES];
    let take = samples.len().min(N_SAMPLES);
    seg[..take].copy_from_slice(&samples[..take]);

    // 2) center reflect-pad by N_FFT/2, periodic Hann
    let pad = N_FFT / 2;
    let padded = reflect_pad(&seg, pad);
    let win = hann_window(N_FFT);

    // number of STFT frames before dropping the last (torch.stft center=True)
    let n_frames_full = 1 + (padded.len() - N_FFT) / HOP_LENGTH;
    // We keep exactly N_FRAMES (HF drops the final frame: stft[..., :-1]).
    let n_keep = N_FRAMES.min(n_frames_full);

    // 3) windowed frames -> [n_keep, N_FFT]
    let mut framed = vec![0f32; n_keep * N_FFT];
    for f in 0..n_keep {
        let off = f * HOP_LENGTH;
        for j in 0..N_FFT {
            framed[f * N_FFT + j] = padded[off + j] * win[j];
        }
    }
    let framed = Tensor::from_vec(framed, (n_keep, N_FFT), device)?;

    // 4) real DFT by matmul: power[f, bin] = (frame . cos)^2 + (frame . sin)^2
    //    cos/sin: [N_FFT, N_FREQ]. exp(-2*pi*i*bin*n/N_FFT).
    let (cosm, sinm) = dft_matrices(device)?; // [N_FFT, N_FREQ]
    let re = framed.matmul(&cosm)?; // [n_keep, N_FREQ]
    let im = framed.matmul(&sinm)?; // [n_keep, N_FREQ]
    let power = (re.sqr()? + im.sqr()?)?; // [n_keep, N_FREQ]

    // 5) mel = mel_filters^T @ power^T  -> [num_mel, n_keep]
    //    power: [n_keep, N_FREQ], mel_filters: [N_FREQ, num_mel]
    let mel = power.matmul(mel_filters)?.t()?.contiguous()?; // [num_mel, n_keep]

    // 6) log10(clamp(mel, 1e-10)); max(., max-8); (.+4)/4
    let mel = mel.clamp(1e-10f32, f32::INFINITY)?;
    let log_spec = mel.log()?.affine(1.0 / std::f64::consts::LN_10, 0.0)?; // ln -> log10
    let mx = log_spec.flatten_all()?.max(0)?.to_scalar::<f32>()? - 8.0; // global max - 8
    let log_spec = log_spec.clamp(mx, f32::INFINITY)?;
    let log_spec = log_spec.affine(0.25, 1.0)?; // (x+4)/4 == 0.25x + 1

    // pad time to N_FRAMES if the (sub-N_SAMPLES) audio gave fewer frames -- HF always emits 3000
    let log_spec = if n_keep < N_FRAMES {
        log_spec.pad_with_zeros(D::Minus1, 0, N_FRAMES - n_keep)?
    } else {
        log_spec
    };
    let _ = cfg;
    Ok(log_spec) // [num_mel, N_FRAMES]
}

/// cos/sin DFT matrices [N_FFT, N_FREQ] for a real FFT (rfft) of length N_FFT.
/// re = x @ cos, im = x @ sin, where sin already carries the -sign of exp(-i theta).
fn dft_matrices(device: &Device) -> Result<(Tensor, Tensor)> {
    let mut cosm = vec![0f32; N_FFT * N_FREQ];
    let mut sinm = vec![0f32; N_FFT * N_FREQ];
    let two_pi = 2.0 * std::f64::consts::PI;
    for n in 0..N_FFT {
        for bin in 0..N_FREQ {
            let theta = two_pi * (bin as f64) * (n as f64) / (N_FFT as f64);
            cosm[n * N_FREQ + bin] = theta.cos() as f32;
            sinm[n * N_FREQ + bin] = (-theta.sin()) as f32;
        }
    }
    Ok((
        Tensor::from_vec(cosm, (N_FFT, N_FREQ), device)?,
        Tensor::from_vec(sinm, (N_FFT, N_FREQ), device)?,
    ))
}

// ===========================================================================
// Whisper encoder (standard HF/OpenAI audio encoder, with hidden-state capture)
// ===========================================================================

struct MultiHeadAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    n_head: usize,
}

impl MultiHeadAttention {
    fn load(n_state: usize, n_head: usize, vb: ShardedVarBuilder) -> Result<Self> {
        // HF whisper: q,v,out have bias; k has no bias.
        let q = linear(n_state, n_state, vb.pp("q_proj"))?;
        let k = linear_no_bias(n_state, n_state, vb.pp("k_proj"))?;
        let v = linear(n_state, n_state, vb.pp("v_proj"))?;
        let out = linear(n_state, n_state, vb.pp("out_proj"))?;
        Ok(Self { q, k, v, out, n_head })
    }

    fn reshape_head(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        x.reshape((b, t, self.n_head, c / self.n_head))?.transpose(1, 2)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let q = self.q.forward(x)?;
        let k = self.k.forward(x)?;
        let v = self.v.forward(x)?;
        let (_b, t, c) = q.dims3()?;
        // OpenAI scaling: split sqrt(d_head) across q and k as ^-0.25 each.
        let scale = ((c / self.n_head) as f64).powf(-0.25);
        let q = (self.reshape_head(&q)? * scale)?;
        let k = (self.reshape_head(&k)?.transpose(2, 3)? * scale)?;
        let v = self.reshape_head(&v)?.contiguous()?;
        let qk = q.contiguous()?.matmul(&k.contiguous()?)?; // [b,h,t,t]
        let w = softmax_last_dim(&qk)?;
        let wv = w.matmul(&v)?.transpose(1, 2)?.reshape((qk.dim(0)?, t, c))?;
        self.out.forward(&wv)
    }
}

struct EncoderBlock {
    attn: MultiHeadAttention,
    attn_ln: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    mlp_ln: LayerNorm,
}

impl EncoderBlock {
    fn load(n_state: usize, n_head: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let attn = MultiHeadAttention::load(n_state, n_head, vb.pp("self_attn"))?;
        let attn_ln = layer_norm(n_state, vb.pp("self_attn_layer_norm"))?;
        let n_mlp = n_state * 4;
        let fc1 = linear(n_state, n_mlp, vb.pp("fc1"))?;
        let fc2 = linear(n_mlp, n_state, vb.pp("fc2"))?;
        let mlp_ln = layer_norm(n_state, vb.pp("final_layer_norm"))?;
        Ok(Self { attn, attn_ln, fc1, fc2, mlp_ln })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (x + self.attn.forward(&self.attn_ln.forward(x)?)?)?;
        let mlp = self
            .fc2
            .forward(&self.fc1.forward(&self.mlp_ln.forward(&x)?)?.gelu_erf()?)?;
        x + mlp
    }
}

fn layer_norm(size: usize, vb: ShardedVarBuilder) -> Result<LayerNorm> {
    let weight = vb.get(size, "weight")?;
    let bias = vb.get(size, "bias")?;
    Ok(LayerNorm::new(weight, bias, 1e-5))
}

fn conv1d(
    in_c: usize,
    out_c: usize,
    k: usize,
    cfg: Conv1dConfig,
    vb: ShardedVarBuilder,
) -> Result<Conv1d> {
    let w = vb.get((out_c, in_c, k), "weight")?;
    let b = vb.get(out_c, "bias")?;
    Ok(Conv1d::new(w, Some(b), cfg))
}

/// Per-stage encoder outputs (for numerical verification against PyTorch).
pub struct EncoderDebug {
    pub conv1: Tensor,        // [B, d, 3000]
    pub conv2: Tensor,        // [B, d, 1500]
    pub input: Tensor,        // [B, 1500, d]  (conv stack + position embedding == hs[0])
    pub hiddens: Vec<Tensor>, // L+1 tensors, each [B, 1500, d]
}

/// Whisper-tiny audio encoder with hidden-state capture.
pub struct WhisperEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    position_embedding: Tensor, // [max_source_positions, d_model], the LEARNED embed_positions
    blocks: Vec<EncoderBlock>,
    ln_post: LayerNorm,
    dtype: DType,
}

impl WhisperEncoder {
    /// `vb` should already be scoped to `model.encoder` (matching the HF safetensors layout).
    pub fn load(vb: ShardedVarBuilder, cfg: &WhisperConfig) -> Result<Self> {
        let n_state = cfg.d_model;
        let n_head = cfg.encoder_attention_heads;
        let c1 = Conv1dConfig { padding: 1, stride: 1, groups: 1, dilation: 1, cudnn_fwd_algo: None };
        let c2 = Conv1dConfig { padding: 1, stride: 2, groups: 1, dilation: 1, cudnn_fwd_algo: None };
        let conv1 = conv1d(cfg.num_mel_bins, n_state, 3, c1, vb.pp("conv1"))?;
        let conv2 = conv1d(n_state, n_state, 3, c2, vb.pp("conv2"))?;
        // Load the trained absolute position embedding (whisper-tiny: a fixed sinusoidal table).
        let position_embedding =
            vb.get((cfg.max_source_positions, n_state), "embed_positions.weight")?;
        let blocks = (0..cfg.encoder_layers)
            .map(|i| EncoderBlock::load(n_state, n_head, vb.pp(format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let ln_post = layer_norm(n_state, vb.pp("layer_norm"))?;
        Ok(Self { conv1, conv2, position_embedding, blocks, ln_post, dtype: vb.dtype() })
    }

    /// Returns all encoder hidden states stacked: `[B, T, n_layers+1, d_model]`.
    /// hs[0] = post-conv + position embedding (block input); hs[i] = block-i output for i<L;
    /// hs[L] = `layer_norm(block_{L-1} output)` -- i.e. the FINAL hidden state has the encoder's
    /// post LayerNorm applied (this is exactly what HF `WhisperEncoder` appends last, == its
    /// `last_hidden_state`). MuseTalk stacks these 5 states as its audio context.
    pub fn forward_hidden_states(&self, mel: &Tensor) -> Result<Tensor> {
        let hiddens = self.hidden_states_vec(mel)?;
        // stack on a new dim=2 -> [B, T, L+1, d_model]
        Tensor::stack(&hiddens, 2)
    }

    fn hidden_states_vec(&self, mel: &Tensor) -> Result<Vec<Tensor>> {
        // mel: [B, num_mel, N_FRAMES] (Conv1d wants channels in dim-1)
        let mel = mel.to_dtype(self.dtype)?;
        let x = self.conv1.forward(&mel)?.gelu_erf()?;
        let x = self.conv2.forward(&x)?.gelu_erf()?;
        let x = x.transpose(1, 2)?.contiguous()?; // [B, T, d_model]
        let t = x.dim(1)?;
        let pos = self.position_embedding.narrow(0, 0, t)?.to_dtype(self.dtype)?;
        let mut x = x.broadcast_add(&pos)?;

        let n = self.blocks.len();
        let mut hiddens: Vec<Tensor> = Vec::with_capacity(n + 1);
        hiddens.push(x.clone());
        for (i, blk) in self.blocks.iter().enumerate() {
            x = blk.forward(&x)?;
            if i + 1 == n {
                // HF appends layer_norm(last_block_output) as the final hidden state.
                hiddens.push(self.ln_post.forward(&x)?);
            } else {
                hiddens.push(x.clone());
            }
        }
        Ok(hiddens)
    }

    /// Per-stage encoder outputs for numerical debugging (mirrors `forward_hidden_states`).
    pub fn forward_debug(&self, mel: &Tensor) -> Result<EncoderDebug> {
        let mel = mel.to_dtype(self.dtype)?;
        let conv1 = self.conv1.forward(&mel)?.gelu_erf()?; // [B, d, 3000]
        let conv2 = self.conv2.forward(&conv1)?.gelu_erf()?; // [B, d, 1500]
        let x = conv2.transpose(1, 2)?.contiguous()?; // [B, 1500, d]
        let t = x.dim(1)?;
        let pos = self.position_embedding.narrow(0, 0, t)?.to_dtype(self.dtype)?;
        let input = x.broadcast_add(&pos)?;
        let n = self.blocks.len();
        let mut hiddens: Vec<Tensor> = Vec::with_capacity(n + 1);
        let mut h = input.clone();
        hiddens.push(h.clone());
        for (i, blk) in self.blocks.iter().enumerate() {
            h = blk.forward(&h)?;
            if i + 1 == n {
                hiddens.push(self.ln_post.forward(&h)?);
            } else {
                hiddens.push(h.clone());
            }
        }
        Ok(EncoderDebug { conv1, conv2, input, hiddens })
    }

    /// Standard last-hidden-state encoder output (post final LayerNorm). Provided for completeness.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let mel = mel.to_dtype(self.dtype)?;
        let x = self.conv1.forward(&mel)?.gelu_erf()?;
        let x = self.conv2.forward(&x)?.gelu_erf()?;
        let x = x.transpose(1, 2)?.contiguous()?;
        let t = x.dim(1)?;
        let pos = self.position_embedding.narrow(0, 0, t)?.to_dtype(self.dtype)?;
        let mut x = x.broadcast_add(&pos)?;
        for blk in self.blocks.iter() {
            x = blk.forward(&x)?;
        }
        self.ln_post.forward(&x)
    }
}

// ===========================================================================
// PositionalEncoding (MuseTalk UNet) -- standard sinusoidal, max_len 5000
// ===========================================================================

pub struct PositionalEncoding {
    pe: Tensor, // [1, max_len, d_model]
}

impl PositionalEncoding {
    pub fn new(d_model: usize, max_len: usize, device: &Device, dtype: DType) -> Result<Self> {
        let mut data = vec![0f32; max_len * d_model];
        let ln10000 = (10000.0f64).ln();
        for pos in 0..max_len {
            for i in 0..(d_model / 2) {
                let div = (-(ln10000) * (2.0 * i as f64) / d_model as f64).exp();
                let angle = pos as f64 * div;
                data[pos * d_model + 2 * i] = angle.sin() as f32;
                data[pos * d_model + 2 * i + 1] = angle.cos() as f32;
            }
        }
        let pe = Tensor::from_vec(data, (1, max_len, d_model), device)?.to_dtype(dtype)?;
        Ok(Self { pe })
    }

    /// x: [B, T, d_model] -> x + pe[:, :T, :]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let t = x.dim(1)?;
        let pe = self.pe.narrow(1, 0, t)?.to_dtype(x.dtype())?;
        x.broadcast_add(&pe)
    }
}

// ===========================================================================
// Full MuseTalk audio2feature pipeline
// ===========================================================================

pub struct AudioFeatureExtractor {
    encoder: WhisperEncoder,
    pe: PositionalEncoding,
    mel_filters: Tensor, // [N_FREQ, num_mel] f32
    cfg: WhisperConfig,
    device: Device,
    dtype: DType,
}

impl AudioFeatureExtractor {
    /// `whisper_vb` scoped to `model.encoder`. `mel_filters` is [N_FREQ, num_mel] f32 (HF order).
    pub fn new(
        whisper_vb: ShardedVarBuilder,
        mel_filters: Tensor,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let cfg = WhisperConfig::default();
        let encoder = WhisperEncoder::load(whisper_vb, &cfg)?;
        let pe = PositionalEncoding::new(cfg.d_model, 5000, device, dtype)?;
        Ok(Self {
            encoder,
            pe,
            mel_filters: mel_filters.to_device(device)?.to_dtype(DType::F32)?,
            cfg,
            device: device.clone(),
            dtype,
        })
    }

    /// mel for one 30s segment of pcm -> [1, num_mel, N_FRAMES].
    pub fn mel(&self, pcm: &[f32]) -> Result<Tensor> {
        let m = log_mel_spectrogram(pcm, &self.mel_filters, &self.cfg, &self.device)?;
        m.unsqueeze(0) // [1, num_mel, N_FRAMES]
    }

    /// Whisper hidden-state stack for one mel segment -> [1, T, L+1, d_model].
    pub fn encode_hidden(&self, mel: &Tensor) -> Result<Tensor> {
        self.encoder.forward_hidden_states(mel)
    }

    /// Per-stage encoder outputs for verification.
    pub fn encode_hidden_debug(&self, mel: &Tensor) -> Result<EncoderDebug> {
        self.encoder.forward_debug(mel)
    }

    /// MuseTalk get_whisper_chunk geometry over the full pcm.
    ///
    /// Returns `[num_frames, 50, d_model]` (the per-video-frame chunk, pre-PositionalEncoding),
    /// matching `AudioProcessor.get_whisper_chunk(..., fps, pad_left=2, pad_right=2)`.
    pub fn whisper_chunks(
        &self,
        pcm: &[f32],
        fps: usize,
        pad_left: usize,
        pad_right: usize,
    ) -> Result<Tensor> {
        let lib_len = pcm.len();
        let feat_len_per_frame = 2 * (pad_left + pad_right + 1); // 10

        // 30s segments; stack the 5 hidden states for each; concat on time.
        let seg_len = CHUNK_LENGTH * SAMPLE_RATE;
        let mut segs: Vec<Tensor> = Vec::new();
        let mut start = 0usize;
        if pcm.is_empty() {
            // single empty segment -> still produce 3000 frames of mel
            segs.push(self.encode_hidden(&self.mel(&[])?)?);
        }
        while start < pcm.len() {
            let end = (start + seg_len).min(pcm.len());
            let mel = self.mel(&pcm[start..end])?; // [1, num_mel, 3000]
            segs.push(self.encode_hidden(&mel)?); // [1, T, L+1, d]
            start += seg_len;
        }
        let whisper_feature = Tensor::cat(&segs, 1)?; // [1, T_total, L+1, d]

        // trim to actual_length, then pad
        let audio_fps = 50usize;
        let whisper_idx_multiplier = audio_fps as f64 / fps as f64;
        let num_frames = ((lib_len as f64 / SAMPLE_RATE as f64) * fps as f64).floor() as usize;
        let actual_length =
            ((lib_len as f64 / SAMPLE_RATE as f64) * audio_fps as f64).floor() as usize;
        let t_total = whisper_feature.dim(1)?;
        let actual_length = actual_length.min(t_total);
        let wf = whisper_feature.narrow(1, 0, actual_length)?; // [1, actual, L+1, d]

        let padding_nums = whisper_idx_multiplier.ceil() as usize;
        let left = padding_nums * pad_left;
        let right = padding_nums * 3 * pad_right;
        let (_b, _a, l, d) = wf.dims4()?;
        let zl = Tensor::zeros((1, left, l, d), wf.dtype(), &self.device)?;
        let zr = Tensor::zeros((1, right, l, d), wf.dtype(), &self.device)?;
        let wf = Tensor::cat(&[&zl, &wf, &zr], 1)?; // [1, actual+left+right, L+1, d]
        let wf_t = wf.dim(1)?;

        // per-frame gather + rearrange 'b c h w -> b (c h) w' where c=10 (time), h=L+1=5
        let mut prompts: Vec<Tensor> = Vec::with_capacity(num_frames);
        for frame in 0..num_frames {
            let audio_index = (frame as f64 * whisper_idx_multiplier).floor() as usize;
            if audio_index + feat_len_per_frame > wf_t {
                // matches Python: padding guarantees this never trips for valid num_frames.
                break;
            }
            let clip = wf.narrow(1, audio_index, feat_len_per_frame)?; // [1, 10, L+1, d]
            // rearrange to [1, 10*(L+1), d] = [1, 50, d]
            let clip = clip.reshape((1, feat_len_per_frame * l, d))?;
            prompts.push(clip);
        }
        let chunks = Tensor::cat(&prompts, 0)?; // [num_frames, 50, d]
        Ok(chunks)
    }

    /// Apply PositionalEncoding to a batch of chunks `[B, 50, d]` -> `[B, 50, d]`.
    pub fn positional_encoding(&self, chunks: &Tensor) -> Result<Tensor> {
        self.pe.forward(&chunks.to_dtype(self.dtype)?)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}
