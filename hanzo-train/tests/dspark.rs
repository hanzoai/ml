//! The DSpark draft on the cluster runtime, end to end on the CPU over a synthetic target cache
//! (each sequence counts up by one, and its hidden states name its tokens): two workers learn
//! it, land on their rounds replayed in one process bit for bit (with the Markov head trained
//! and frozen), and the coordinator writes a checkpoint in the engine's layout; a worker whose
//! cache differs is refused.

use anyhow::Result;
use hanzo_ml::{DType, Device, Tensor};
use hanzo_train::cluster::{self, net, replay, start, Adam, Model, Params, Round, Summary, Worker};
use hanzo_train::dspark::{self, Draft, Engine, Local, Spec, CODE};
use std::collections::HashMap;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier};

/// Vocabulary, hidden width, fused target layers, samples.
const V: usize = 32;
const H: usize = 16;
const FUSED: usize = 2;
const N: usize = 24;

/// A fixed stream: splitmix64, uniform in [−1, 1).
fn stream(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

/// A target cache of `n` sequences in `dir`, each counting up from its own start, every
/// position's hidden states a fixed vector of its token; and the frozen embedding and head.
fn cache(dir: &Path, n: usize) -> Result<Local> {
    std::fs::create_dir_all(dir)?;
    let mut next = stream(3);
    let token: Vec<Vec<f32>> = (0..V).map(|_| (0..H).map(|_| next()).collect()).collect();
    let (mut idx, mut shard) = (Vec::new(), Vec::new());
    for i in 0..n {
        let seq = 12 + i % 9;
        let ids: Vec<i32> = (0..seq).map(|p| ((i * 5 + p) % (V - 1)) as i32).collect();
        let mut at = |bytes: &[u8]| {
            let off = shard.len() as u64;
            shard.extend_from_slice(bytes);
            off
        };
        let ids_at = at(&ids.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<_>>());
        let attn_at = at(&vec![1u8; seq]);
        let loss_at = at(&(0..seq).map(|p| (p > 0) as u8).collect::<Vec<_>>());
        let half = |layers: usize| -> Vec<u8> {
            ids.iter()
                .flat_map(|&t| {
                    (0..layers)
                        .flat_map(|l| token[t as usize].iter().map(move |x| x * (1.0 + l as f32)))
                        .flat_map(|x| net::bf16(x).to_le_bytes())
                        .collect::<Vec<_>>()
                })
                .collect()
        };
        let hidden_at = at(&half(FUSED));
        let last_at = at(&half(1));
        idx.extend((i as u64).to_le_bytes());
        idx.extend(0u32.to_le_bytes());
        idx.extend((seq as u32).to_le_bytes());
        for off in [ids_at, attn_at, loss_at, hidden_at, last_at] {
            idx.extend(off.to_le_bytes());
        }
    }
    std::fs::write(dir.join("samples.idx"), idx)?;
    std::fs::write(dir.join("shard-0.bin"), shard)?;
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_vec(&serde_json::json!({
            "hidden_size": H, "target_layer_ids": [1, 2], "num_samples": n,
            "shards": [{"file_name": "shard-0.bin", "shard_id": 0}],
        }))?,
    )?;
    let init = dir.join("embed_head.safetensors");
    let mut next = stream(5);
    let table = |next: &mut dyn FnMut() -> f32| -> Result<Tensor> {
        let xs: Vec<f32> = (0..V * H).map(|_| next()).collect();
        Ok(Tensor::from_vec(xs, (V, H), &Device::Cpu)?.to_dtype(DType::F16)?)
    };
    let tensors = HashMap::from([
        ("embed_tokens.weight".to_string(), table(&mut next)?),
        ("lm_head.weight".to_string(), table(&mut next)?),
    ]);
    hanzo_ml::safetensors::save(&tensors, &init)?;
    Ok(Local {
        cache_dir: dir.to_path_buf(),
        init,
    })
}

fn spec(freeze_markov: bool) -> Spec {
    Spec {
        steps: 80,
        layers: 1,
        intermediate: 32,
        heads: 2,
        kv_heads: 1,
        head_dim: 8,
        vocab: V,
        markov_rank: 4,
        block: 3,
        mask_token_id: (V - 1) as u32,
        num_anchors: 2,
        micro_batch: 2,
        lr: 5e-2,
        warmup: 0.0,
        cool: 0.0,
        final_norm_init: 0.1,
        max_seq: 0,
        seed: 7,
        freeze_markov,
        fixed_batch: false,
        round: 0.0,
        grace: 60.0,
        outer: 0.7,
        momentum: 0.5,
        every: 0,
    }
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("hanzo-train-dspark-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// Mean loss over the plan's first `k` steps with the trained parameters at `theta`.
fn loss(spec: &Spec, local: &Local, theta: &[f32], k: usize) -> Result<f64> {
    let mut d = Draft::new(spec, local)?;
    let params = Params::of(d.vars(), |n| d.rate(n).is_some());
    params.write(d.vars(), theta)?;
    // a zero rate: the step reports its loss and moves nothing
    let mut still = Adam::new(d.vars(), |_| Some(0.0), spec.adam())?;
    let plan = spec.plan(&d.cache)?;
    let at = Round {
        round: 0,
        params: &params,
        theta,
    };
    let mut sum = 0.0;
    for b in 0..k {
        sum += d.step(&mut still, &plan.batch(b), &at)?.sums[0];
    }
    Ok(sum / k as f64)
}

/// Two workers, members before either takes a step: θ_global after the plan, and the plan.
fn two(spec: &Spec, local: &Local, out: &Path) -> Result<(cluster::Report, Vec<f32>)> {
    let (run, begin, _) = dspark::run(spec, local, Some(out.to_path_buf()))?;
    let theta = begin.theta.clone();
    let engine = Engine {
        spec: spec.clone(),
        local: local.clone(),
    };
    let c = start(TcpListener::bind("127.0.0.1:0")?, run, begin, engine)?;
    let gate = Arc::new(Barrier::new(2));
    let ws: Vec<_> = ["a", "b"]
        .into_iter()
        .map(|name| {
            let (addr, local, gate) = (c.addr(), local.clone(), gate.clone());
            std::thread::spawn(move || -> Result<Summary> {
                let joined = cluster::join(addr, name, CODE, |s| dspark::load(s, &local));
                gate.wait();
                joined?.map_or(Ok(Summary::default()), Worker::run)
            })
        })
        .collect();
    let report = c.join()?;
    for w in ws {
        w.join().expect("worker")?;
    }
    Ok((report, theta))
}

#[test]
fn two_workers_learn_the_draft_as_their_rounds_replay() -> Result<()> {
    let root = scratch("learn");
    let local = cache(&root.join("cache"), N)?;
    for frozen in [false, true] {
        let s = spec(frozen);
        let out = root.join(format!("out-{frozen}"));
        let (report, start0) = two(&s, &local, &out)?;
        let plan = s.plan(&dspark::Draft::new(&s, &local)?.cache)?;
        let same = replay(
            &report.log,
            &plan,
            s.adam(),
            s.outer,
            s.momentum,
            start0.clone(),
            |_| Draft::new(&s, &local),
        )?;
        let sizes: Vec<usize> = report.log.iter().map(Vec::len).collect();
        let (before, after) = (
            loss(&s, &local, &start0, 6)?,
            loss(&s, &local, &report.theta, 6)?,
        );
        eprintln!(
            "markov frozen {frozen}: rounds of {sizes:?}; loss {before:.4} → {after:.4}; \
             ln V {:.4}",
            (V as f64).ln()
        );
        assert_eq!(sizes, vec![2; s.steps / 2]);
        assert_eq!(
            report.theta, same,
            "the cluster against its rounds replayed"
        );
        // measured: 3.44 → 0.52 with the Markov head, 3.44 → 2.76 with it frozen
        let bound = if frozen { 0.9 } else { 0.5 };
        assert!(after < bound * before, "{before} → {after}");

        // the checkpoint: the engine's layout, θ_global's trained weights, a frozen Markov head
        // at the init every worker drew
        let saved = hanzo_ml::safetensors::load(out.join("model.safetensors"), &Device::Cpu)?;
        assert!(out.join("config.json").exists());
        let fresh = Draft::new(&s, &local)?;
        let params = Params::of(fresh.vars(), |n| fresh.rate(n).is_some());
        for (i, name) in params.names.iter().enumerate() {
            let t = saved[name].flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(t, report.theta[params.span(i)], "{name}");
        }
        let markov = "markov_head.markov_w1.weight";
        assert_eq!(params.names.iter().any(|n| n == markov), !frozen);
        if frozen {
            let init = fresh.vars().data().lock().unwrap()[markov]
                .as_tensor()
                .clone();
            assert_eq!(
                saved[markov].flatten_all()?.to_vec1::<f32>()?,
                init.flatten_all()?.to_vec1::<f32>()?
            );
        }
    }
    std::fs::remove_dir_all(&root)?;
    Ok(())
}

#[test]
fn a_worker_with_another_cache_is_refused() -> Result<()> {
    let root = scratch("refuse");
    let (ours, theirs) = (cache(&root.join("a"), N)?, cache(&root.join("b"), N - 1)?);
    let s = spec(false);
    let (run, begin, _) = dspark::run(&s, &ours, None)?;
    let engine = Engine {
        spec: s.clone(),
        local: ours.clone(),
    };
    let c = start(TcpListener::bind("127.0.0.1:0")?, run, begin, engine)?;
    let err = cluster::join(c.addr(), "x", CODE, |setup| dspark::load(setup, &theirs))
        .err()
        .expect("refused");
    assert!(err.to_string().contains("cache"), "{err}");
    std::fs::remove_dir_all(&root)?;
    Ok(())
}
