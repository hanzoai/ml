//! `hanzo-train` CLI: train a DSpark draft on the DeepSpec target cache and save an engine-loadable
//! checkpoint. CPU / f32 MVP. The goal is a real, decreasing loss curve on real data + a checkpoint
//! whose keys and shapes exactly match the engine loader.

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use hanzo_ml::{Device, Result, Tensor};
use hanzo_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use rand::{rngs::StdRng, Rng, SeedableRng};

use hanzo_train::cache::Cache;
use hanzo_train::model::{verify_checkpoint, Dspark, DsparkCfg};

#[derive(Parser, Debug)]
#[command(name = "hanzo-train", about = "Native-Rust DSpark draft trainer (CPU/f32 MVP)")]
struct Args {
    /// DeepSpec target-cache v2 directory (manifest.json + samples.idx + shard-*.bin).
    #[arg(long)]
    cache_dir: PathBuf,

    /// Frozen embed_tokens + lm_head init safetensors (keys embed_tokens.weight, lm_head.weight).
    #[arg(long, default_value = "/home/z/work/zen/hf/v4-dspark-init/embed_head.safetensors")]
    init: PathBuf,

    /// Output checkpoint directory (writes model.safetensors + config.json).
    #[arg(long, default_value = "./dspark-mvp-ckpt")]
    out: PathBuf,

    #[arg(long, default_value_t = 100)]
    steps: usize,

    /// Decoder layers (MVP default 2; full model uses 5).
    #[arg(long, default_value_t = 2)]
    layers: usize,

    #[arg(long, default_value_t = 4096)]
    intermediate: usize,

    #[arg(long, default_value_t = 32)]
    heads: usize,

    #[arg(long, default_value_t = 8)]
    kv_heads: usize,

    #[arg(long, default_value_t = 128)]
    head_dim: usize,

    #[arg(long, default_value_t = 129280)]
    vocab: usize,

    #[arg(long, default_value_t = 256)]
    markov_rank: usize,

    #[arg(long, default_value_t = 7)]
    block: usize,

    #[arg(long, default_value_t = 129279)]
    mask_token_id: u32,

    /// Anchors sampled per training sample.
    #[arg(long, default_value_t = 4)]
    num_anchors: usize,

    /// Samples accumulated per optimizer step (micro-batch).
    #[arg(long, default_value_t = 4)]
    micro_batch: usize,

    #[arg(long, default_value_t = 6e-4)]
    lr: f64,

    /// Init scale for the output-side RMSNorm (see DsparkCfg::final_norm_init). `1.0` = faithful;
    /// a small value (e.g. 0.1) starts the loss near ln(vocab) and well-conditions training.
    #[arg(long, default_value_t = 0.1)]
    final_norm_init: f64,

    /// Only train on samples with seq_len <= this (0 = no cap). Bounds per-step cost/memory and
    /// keeps sequence lengths uniform for a smoother curve. The `fc` fuse dominates cost on long seqs.
    #[arg(long, default_value_t = 0)]
    max_seq: usize,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value_t = 5)]
    log_every: usize,

    /// Safety floor: if host MemAvailable drops below this (GB) at a step boundary, save a partial
    /// checkpoint and stop — so this trainer can never starve a co-resident job into the OOM killer.
    #[arg(long, default_value_t = 7.0)]
    min_avail_gb: f64,

    /// Freeze the vocab×rank Markov head (exclude it from AdamW + detach its bias). Drops ~0.8GB of
    /// optimizer/grad memory; use on memory-constrained hosts. The head is still saved at its init.
    #[arg(long, default_value_t = false)]
    freeze_markov: bool,
}

/// Host MemAvailable in GB from /proc/meminfo; `+inf` if it can't be read (never blocks then).
fn mem_available_gb() -> f64 {
    let Ok(s) = std::fs::read_to_string("/proc/meminfo") else {
        return f64::INFINITY;
    };
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            if let Some(kb) = rest.trim().split_whitespace().next() {
                if let Ok(kb) = kb.parse::<f64>() {
                    return kb / 1024.0 / 1024.0;
                }
            }
        }
    }
    f64::INFINITY
}

fn pick_anchors(loss_mask: &[u8], seq: usize, block: usize, n: usize, rng: &mut StdRng) -> Vec<usize> {
    // Candidate anchors: loss_mask[a] && loss_mask[a+1], and a+block <= seq-1 so all block targets fit.
    if seq < block + 2 {
        return Vec::new();
    }
    let mut cands: Vec<usize> = (1..=seq - block - 1)
        .filter(|&a| loss_mask[a] != 0 && loss_mask[a + 1] != 0)
        .collect();
    let take = n.min(cands.len());
    // Partial Fisher-Yates for `take` uniform picks without replacement.
    for i in 0..take {
        let j = i + rng.random_range(0..(cands.len() - i));
        cands.swap(i, j);
    }
    cands.truncate(take);
    cands
}

fn main() -> Result<()> {
    let args = Args::parse();
    let dev = Device::Cpu;

    let cache = Cache::open(&args.cache_dir)?;
    let hidden = cache.manifest.hidden_size;
    let n_fused = cache.n_fused;
    let target_layer_ids = cache.manifest.target_layer_ids.clone();
    println!(
        "cache: {} samples, hidden={hidden}, n_fused={n_fused}, layer_ids={:?}",
        cache.len(),
        target_layer_ids
    );

    // RoPE table must cover the longest sequence (+ block) in the cache.
    let max_seq = cache.records.iter().map(|r| r.seq_len as usize).max().unwrap_or(0);
    let max_pos = max_seq + args.block + 1;

    let cfg = DsparkCfg {
        vocab: args.vocab,
        hidden,
        intermediate: args.intermediate,
        layers: args.layers,
        heads: args.heads,
        kv_heads: args.kv_heads,
        head_dim: args.head_dim,
        rms_eps: 1e-6,
        rope_theta: 1e6,
        max_pos,
        block: args.block,
        mask_token_id: args.mask_token_id,
        markov_rank: args.markov_rank,
        target_layer_ids,
        final_norm_init: args.final_norm_init,
    };
    assert_eq!(cfg.heads * cfg.head_dim, cfg.hidden, "heads*head_dim must equal hidden");

    println!(
        "model: layers={} heads={}/{} head_dim={} intermediate={} vocab={} block={} markov_rank={}",
        cfg.layers, cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.intermediate, cfg.vocab, cfg.block, cfg.markov_rank
    );

    let model = Dspark::new(cfg.clone(), &args.init, &dev, !args.freeze_markov)?;
    let vars = model.trainable_vars();
    let n_params: usize = vars.iter().map(|v| v.as_tensor().elem_count()).sum();
    println!(
        "trainable vars: {} ({} params){}",
        vars.len(),
        n_params,
        if args.freeze_markov { " [markov frozen]" } else { "" }
    );

    let mut opt = AdamW::new(
        vars,
        ParamsAdamW {
            lr: args.lr,
            ..Default::default()
        },
    )?;

    // Eligible sample pool: long enough to host a full block, and within the optional seq cap.
    let eligible: Vec<usize> = (0..cache.len())
        .filter(|&i| {
            let s = cache.records[i].seq_len as usize;
            s >= cfg.block + 2 && (args.max_seq == 0 || s <= args.max_seq)
        })
        .collect();
    if eligible.is_empty() {
        hanzo_ml::bail!("no eligible samples (seq_len in [{}, {}])", cfg.block + 2, args.max_seq);
    }
    println!("eligible samples: {}/{} (max_seq={})", eligible.len(), cache.len(), args.max_seq);

    let mut rng = StdRng::seed_from_u64(args.seed);
    let ln_vocab = (cfg.vocab as f64).ln();
    println!("baseline CE = ln(vocab) = {:.4}\n--- training ---", ln_vocab);

    let t_start = Instant::now();
    let mut tok_total = 0usize;
    for step in 0..args.steps {
        // Never risk the co-resident dump: bail (and save) before memory gets dangerous.
        let avail = mem_available_gb();
        if avail < args.min_avail_gb {
            println!(
                "step {step}: MemAvailable {avail:.1}GB < floor {:.1}GB — stopping to protect the host; saving partial checkpoint",
                args.min_avail_gb
            );
            break;
        }

        let mut block_chunks: Vec<Tensor> = Vec::new();
        let mut bias_chunks: Vec<Tensor> = Vec::new();
        let mut targets: Vec<u32> = Vec::new();

        for _ in 0..args.micro_batch {
            let si = eligible[rng.random_range(0..eligible.len())];
            let mut s = cache.read_sample(si)?;
            let seq = s.seq_len;
            let anchors = pick_anchors(&s.loss_mask, seq, cfg.block, args.num_anchors, &mut rng);
            if anchors.is_empty() {
                continue;
            }
            let th = Tensor::from_vec(std::mem::take(&mut s.target_hidden), (seq, n_fused * hidden), &dev)?;
            let fused = model.fuse(&th)?;
            for a in anchors {
                if let Some((block, bias, tgt)) = model.draft_anchor(&fused, &s.input_ids, &s.loss_mask, a)? {
                    targets.extend_from_slice(&tgt);
                    block_chunks.push(block);
                    bias_chunks.push(bias);
                }
            }
        }

        if targets.is_empty() {
            continue;
        }
        let block_refs: Vec<&Tensor> = block_chunks.iter().collect();
        let bias_refs: Vec<&Tensor> = bias_chunks.iter().collect();
        let block_cat = Tensor::cat(&block_refs, 0)?; // [N, hidden]
        let bias_cat = Tensor::cat(&bias_refs, 0)?; // [N, vocab]
        let tgt = Tensor::from_vec(targets.clone(), (targets.len(),), &dev)?;
        // CE through the FROZEN head via surrogate backward (no [hidden, vocab] head grad formed).
        let (loss, surrogate) = model.head_ce(&block_cat, &bias_cat, &tgt)?;
        opt.step(&surrogate.backward()?)?;

        tok_total += targets.len();
        if step % args.log_every == 0 || step == args.steps - 1 {
            let l = loss.to_scalar::<f32>()?;
            let tps = tok_total as f64 / t_start.elapsed().as_secs_f64();
            println!(
                "step {step:>4}  loss {l:8.4}  tokens {:>5}  {:.0} tok/s  avail {:.1}GB",
                targets.len(),
                tps,
                mem_available_gb()
            );
        }
    }
    let elapsed = t_start.elapsed().as_secs_f64();
    println!(
        "--- done: {} steps in {:.1}s, {:.0} supervised tok/s ---",
        args.steps,
        elapsed,
        tok_total as f64 / elapsed
    );

    model.save(&args.out)?;
    let ckpt = args.out.join("model.safetensors");
    println!("saved checkpoint -> {}", ckpt.display());
    verify_checkpoint(&ckpt, &cfg)?;
    println!("checkpoint load-check PASSED: every engine key present with matching shape");
    Ok(())
}
