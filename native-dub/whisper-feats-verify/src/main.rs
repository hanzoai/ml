//! Standalone verifier for the native MuseTalk whisper-tiny audio-feature stage.
//!
//! Reuses the EXACT bench module (`musetalk-bench/src/musetalk/whisper.rs`) via #[path] include,
//! so there is a single source of truth. Compares each Rust stage to the dumped Python reference
//! (refdump/wf/*.npy) and prints cosine / PSNR. Decoupled from the rest of the bench so it builds
//! regardless of unrelated WIP in the harness.
//!
//! Run (on spark): MUSETALK_DEV=cuda MUSETALK_DTYPE=f32 ./target/release/whisper-feats-verify

// In-tree copy of the ShardedVarBuilder whisper module (the live build links the identical
// file at musetalk-bench/src/musetalk/whisper.rs; see native-dub/README.md).
#[path = "whisper_shardedvb.rs"]
mod whisper;

use std::sync::Arc;

use hanzo_ml::{DType, Device, Result, Tensor};
use hanzo_quant::{ShardedSafeTensors, ShardedVarBuilder};

use whisper::{AudioFeatureExtractor, N_SAMPLES};

fn real_vb(path: &str, dtype: DType, dev: &Device) -> Result<ShardedVarBuilder> {
    let paths = [std::path::PathBuf::from(path)];
    unsafe { ShardedSafeTensors::sharded(&paths, dtype, dev, None, Arc::new(|_| true)) }
}

fn pick_dtype() -> DType {
    match std::env::var("MUSETALK_DTYPE").as_deref() {
        Ok("f16") => DType::F16,
        Ok("bf16") => DType::BF16,
        _ => DType::F32,
    }
}

fn psnr_cosine(a: &Tensor, b: &Tensor) -> Result<(f64, f64)> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(a.len(), b.len());
    let mut mse = 0f64;
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (x, y) = (x as f64, y as f64);
        mse += (x - y) * (x - y);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    mse /= a.len() as f64;
    let psnr = if mse <= 1e-12 { 120.0 } else { 10.0 * (1.0 / mse).log10() };
    let cosine = dot / (na.sqrt() * nb.sqrt() + 1e-12);
    Ok((psnr, cosine))
}

/// Minimal 16-bit PCM mono WAV reader -> f32 in [-1,1] (matches librosa.load / soundfile).
fn read_wav_16k_mono(path: &str) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).map_err(hanzo_ml::Error::wrap)?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        hanzo_ml::bail!("not a RIFF/WAVE file: {path}");
    }
    let mut pos = 12usize;
    let (mut channels, mut bits, mut sample_rate) = (1u16, 16u16, 16000u32);
    let mut data: Option<(usize, usize)> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let sz = u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]])
            as usize;
        let body = pos + 8;
        if id == b"fmt " {
            channels = u16::from_le_bytes([bytes[body + 2], bytes[body + 3]]);
            sample_rate =
                u32::from_le_bytes([bytes[body + 4], bytes[body + 5], bytes[body + 6], bytes[body + 7]]);
            bits = u16::from_le_bytes([bytes[body + 14], bytes[body + 15]]);
        } else if id == b"data" {
            data = Some((body, sz.min(bytes.len() - body)));
        }
        pos = body + sz + (sz & 1);
    }
    let (dstart, dlen) = data.ok_or_else(|| hanzo_ml::Error::msg("no data chunk"))?;
    if bits != 16 {
        hanzo_ml::bail!("only 16-bit PCM WAV supported (got {bits}-bit)");
    }
    if sample_rate != 16000 {
        hanzo_ml::bail!("expected 16kHz WAV (got {sample_rate})");
    }
    let ch = channels.max(1) as usize;
    let mut samples = Vec::with_capacity(dlen / (2 * ch));
    let mut i = 0usize;
    while i + 2 * ch <= dlen {
        let mut acc = 0f32;
        for c in 0..ch {
            let o = dstart + i + 2 * c;
            acc += i16::from_le_bytes([bytes[o], bytes[o + 1]]) as f32 / 32768.0;
        }
        samples.push(acc / ch as f32);
        i += 2 * ch;
    }
    Ok(samples)
}

fn build_audio_feats(wdir: &str, dev: &Device, dtype: DType) -> Result<AudioFeatureExtractor> {
    let vb = real_vb(&format!("{wdir}/whisper.safetensors"), dtype, dev)?
        .pp("model")
        .pp("encoder");
    let mel_filters = Tensor::read_npy(format!("{wdir}/mel_filters.npy"))?; // [201, 80] f32
    AudioFeatureExtractor::new(vb, mel_filters, dev, dtype)
}

/// Native audio2feature dump: wav -> per-video-frame post-PositionalEncoding whisper feature,
/// written as `{outdir}/audio_{i:06}.npy` ([1,50,384]) -- a drop-in replacement for the Python
/// whisper stage in musetalk_dub_native.py.
///   WF_AUDIO=<wav> WF_OUTDIR=<dir> [WF_FPS=25] [MUSETALK_DEV=cuda] [MUSETALK_DTYPE=f16]
fn run_dump() -> Result<()> {
    let wdir = std::env::var("MUSETALK_WDIR")
        .unwrap_or_else(|_| "/home/z/work/zen-dub-run/rustweights".to_string());
    let audio = std::env::var("WF_AUDIO").expect("set WF_AUDIO");
    let outdir = std::env::var("WF_OUTDIR").expect("set WF_OUTDIR");
    let fps: usize = std::env::var("WF_FPS").ok().and_then(|s| s.parse().ok()).unwrap_or(25);
    let dev = match std::env::var("MUSETALK_DEV").as_deref() {
        Ok("cuda") => Device::new_cuda(0)?,
        _ => Device::Cpu,
    };
    let dtype = pick_dtype();
    std::fs::create_dir_all(&outdir).ok();
    let fx = build_audio_feats(&wdir, &dev, dtype)?;
    let pcm = read_wav_16k_mono(&audio)?;
    let chunks = fx.whisper_chunks(&pcm, fps, 2, 2)?; // [num_frames, 50, 384]
    let n = chunks.dim(0)?;
    println!(
        "[whisperfeat] {audio} -> {n} frames @ {fps}fps  dev={:?} dtype={dtype:?}",
        dev.location()
    );
    for i in 0..n {
        let raw = chunks.narrow(0, i, 1)?; // [1,50,384]
        let pe = fx.positional_encoding(&raw)?; // post-PE, the UNet cross-attn context
        pe.to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .write_npy(format!("{outdir}/audio_{i:06}.npy"))?;
    }
    println!("[whisperfeat] wrote {n} audio_*.npy to {outdir}");
    Ok(())
}

fn main() -> Result<()> {
    if std::env::args().nth(1).as_deref() == Some("dump") {
        return run_dump();
    }
    let refdir = std::env::var("WF_REFDIR")
        .unwrap_or_else(|_| "/home/z/work/zen-dub-run/refdump/wf".to_string());
    let wdir = std::env::var("MUSETALK_WDIR")
        .unwrap_or_else(|_| "/home/z/work/zen-dub-run/rustweights".to_string());
    let audio = std::env::var("WF_AUDIO")
        .unwrap_or_else(|_| "/home/z/work/zen-dub-run/clip/tts_en.wav".to_string());
    let dev = match std::env::var("MUSETALK_DEV").as_deref() {
        Ok("cuda") => Device::new_cuda(0)?,
        _ => Device::Cpu,
    };
    let dtype = pick_dtype();
    let load = |n: &str| -> Result<Tensor> {
        Tensor::read_npy(format!("{refdir}/{n}.npy"))?.to_device(&Device::Cpu)
    };
    let cmp = |label: &str, rust: &Tensor, refn: &str| -> Result<f64> {
        let r = load(refn)?;
        let rust_cpu = rust.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        if r.dims() != rust_cpu.dims() {
            println!("{label:24} SHAPE MISMATCH ref={:?} rust={:?}", r.dims(), rust_cpu.dims());
            return Ok(0.0);
        }
        let (p, c) = psnr_cosine(&r, &rust_cpu)?;
        println!("{label:24} PSNR {p:8.3} dB  cosine {c:.6}   shape {:?}", rust_cpu.dims());
        Ok(c)
    };

    println!(
        "\n==== Whisper-tiny audio features: RUST vs Python  dev={:?} dtype={:?} ====",
        dev.location(),
        dtype
    );
    let fx = build_audio_feats(&wdir, &dev, dtype)?;
    let pcm = read_wav_16k_mono(&audio)?;
    println!("pcm: {} samples ({:.3}s)  audio={audio}", pcm.len(), pcm.len() as f64 / 16000.0);

    // also cross-check our WAV decode against the python pcm dump (if present)
    if let Ok(refpcm) = Tensor::read_npy(format!("{refdir}/pcm16k.npy")) {
        let refpcm = refpcm.flatten_all()?.to_vec1::<f32>()?;
        let n = refpcm.len().min(pcm.len());
        let mut md = 0f32;
        for i in 0..n {
            md = md.max((refpcm[i] - pcm[i]).abs());
        }
        println!("wav-decode vs librosa pcm: len {} vs {}, max|diff|={md:.3e}", pcm.len(), refpcm.len());
    }

    let mel = fx.mel(&pcm[..pcm.len().min(N_SAMPLES)])?; // [1,80,3000]
    cmp("mel", &mel, "mel")?;

    // per-stage encoder debug (gated): compares conv1/conv2/input + each of the 5 hidden states.
    if std::env::var("WF_DEBUG").is_ok() {
        let dbg = fx.encode_hidden_debug(&mel)?;
        cmp("enc_conv1", &dbg.conv1, "enc_conv1")?;
        cmp("enc_conv2", &dbg.conv2, "enc_conv2")?;
        cmp("enc_input", &dbg.input, "enc_input")?;
        for (i, h) in dbg.hiddens.iter().enumerate() {
            cmp(&format!("enc_hs_{i}"), h, &format!("enc_hs_{i}"))?;
        }
    }

    let hs = fx.encode_hidden(&mel)?; // [1, T, 5, 384]
    cmp("hidden_stack", &hs, "hidden_stack")?;

    let chunks = fx.whisper_chunks(&pcm, 25, 2, 2)?; // [num_frames, 50, 384]
    cmp("chunks_all", &chunks, "chunks_all")?;

    let nf = chunks.dim(0)?;
    let mut worst = f64::MAX;
    for i in [0usize, 1, 5, 30, 204] {
        if i >= nf {
            continue;
        }
        let raw = chunks.narrow(0, i, 1)?;
        worst = worst.min(cmp(&format!("chunk_raw[{i}]"), &raw, &format!("chunk_raw_{i}"))?);
        let pe = fx.positional_encoding(&raw)?;
        worst = worst.min(cmp(&format!("chunk_pe[{i}]"), &pe, &format!("chunk_pe_{i}"))?);
    }
    println!("\nworst per-frame cosine: {worst:.6}  (>0.9999 = matching; these feed UNet cross-attn)");
    Ok(())
}
