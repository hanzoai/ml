//! AdaCheck redundancy-aware training checkpoint tests and reproducible benchmarks (HIP-1349).

use anyhow::Result;
use hanzo_train::cluster::net::{self, Params};
use hanzo_train::cluster::{AdaCheckManifest, AdaCheckStore, Inner};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Barrier};
use std::time::Instant;

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "hanzo_adacheck_{tag}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn synthetic_params(len: usize) -> Params {
    Params {
        names: vec!["layers.0.weight".into()],
        shapes: vec![vec![len]],
        offsets: vec![0],
        len,
    }
}

fn assert_f32_bits_eq(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (idx, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "bit mismatch at element {idx}: {x} vs {y}"
        );
    }
}

fn splitmix_f32(seed: u64) -> f32 {
    let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32 - 0.5
}

#[test]
fn bit_identical_roundtrip_including_signed_zeros_and_subnormals() -> Result<()> {
    let root = temp_dir("roundtrip");
    let store = AdaCheckStore::new(root.join("store"), 1_024);
    let n = 4_096;
    let params = synthetic_params(n);

    let mut theta: Vec<f32> = (0..n).map(|i| splitmix_f32(i as u64)).collect();
    theta[0] = -0.0f32;
    theta[1] = 0.0f32;
    theta[2] = f32::from_bits(1);

    let momentum: Vec<f32> = vec![0.0; n];
    let carry: Vec<f32> = vec![0.0; n];
    let merged: Vec<bool> = (0..64).map(|i| i % 3 == 0).collect();

    let mut w1_flat: Vec<f32> = (0..3 * n).map(|i| splitmix_f32(0xabcd_0000 + i as u64)).collect();
    w1_flat[7] = -0.0f32;
    let w2_flat = w1_flat.clone();

    let mut inner = HashMap::new();
    inner.insert(
        "worker-spark".to_string(),
        Inner {
            t: 12,
            data: Arc::new(net::pack(&w1_flat)),
        },
    );
    inner.insert(
        "worker-evo".to_string(),
        Inner {
            t: 12,
            data: Arc::new(net::pack(&w2_flat)),
        },
    );

    let ckpt_dir = root.join("ckpt_1");
    let save_stats =
        store.save_checkpoint(&ckpt_dir, &theta, &momentum, &carry, &merged, 5, &inner)?;
    assert!(
        save_stats.reused_bytes_intra > 0,
        "zero carry/momentum and identical worker states must deduplicate intra-checkpoint"
    );

    let (restored, _rstats) = store.restore_checkpoint(&ckpt_dir, &params, 128)?;
    assert_eq!(restored.round, 5);
    assert_f32_bits_eq(&restored.theta, &theta);
    assert_f32_bits_eq(&restored.momentum, &momentum);
    assert_f32_bits_eq(&restored.carry, &carry);
    assert_eq!(&restored.merged[..64], &merged[..]);
    assert_eq!(&restored.merged[64..], &vec![false; 64][..]);
    assert_eq!(restored.inner["worker-spark"].t, 12);
    assert_eq!(
        *restored.inner["worker-spark"].data,
        *inner["worker-spark"].data
    );
    assert_eq!(
        *restored.inner["worker-evo"].data,
        *inner["worker-evo"].data
    );

    let _ = fs::remove_dir_all(&root);
    Ok(())
}

#[test]
fn detects_corruption_rejects_path_traversal_and_supports_concurrent_saves() -> Result<()> {
    let root = temp_dir("security");
    let store = Arc::new(AdaCheckStore::new(root.join("store"), 1_024));
    let n = 2_048;
    let params = synthetic_params(n);

    // 1. Path traversal worker name is rejected.
    let mut bad_inner = HashMap::new();
    bad_inner.insert(
        "../escape".to_string(),
        Inner {
            t: 1,
            data: Arc::new(vec![0u8; n * 12]),
        },
    );
    let err = store.save_checkpoint(
        &root.join("bad"),
        &vec![0.0; n],
        &vec![0.0; n],
        &vec![0.0; n],
        &[true, false],
        1,
        &bad_inner,
    );
    assert!(err.is_err(), "path traversal worker name must be rejected");

    // 2. Concurrent saves sharing one blob store succeed without races.
    let barrier = Arc::new(Barrier::new(8));
    let mut handles = Vec::new();
    for tid in 0..8usize {
        let (st, b, r) = (store.clone(), barrier.clone(), root.clone());
        handles.push(std::thread::spawn(move || -> Result<()> {
            b.wait();
            let theta: Vec<f32> = (0..n)
                .map(|i| splitmix_f32(((tid % 2) as u64) << 20 | i as u64))
                .collect();
            let mom = vec![0.0f32; n];
            let carry = vec![0.0f32; n];
            let mut inner = HashMap::new();
            inner.insert(
                format!("worker-{tid}"),
                Inner {
                    t: tid as u64,
                    data: Arc::new(vec![0u8; n * 12]),
                },
            );
            let dir = r.join(format!("concurrent_{tid}"));
            st.save_checkpoint(&dir, &theta, &mom, &carry, &[true; 8], tid as u64, &inner)?;
            Ok(())
        }));
    }
    for h in handles {
        h.join().unwrap()?;
    }
    for tid in 0..8usize {
        let (start, _) =
            store.restore_checkpoint(&root.join(format!("concurrent_{tid}")), &params, 8)?;
        assert_eq!(start.round, tid as u64);
    }

    // 3. Silent bit-flip in concurrent_0's first theta blob fails SHA-256 verification.
    let m0: AdaCheckManifest = serde_json::from_slice(&fs::read(
        root.join("concurrent_0").join("adacheck_manifest.json"),
    )?)?;
    let d0 = &m0.theta.chunks[0];
    let bp = root
        .join("store")
        .join("blobs")
        .join(&d0[..2])
        .join(format!("{d0}.bin"));
    let mut data = fs::read(&bp)?;
    data[0] ^= 0x01;
    fs::write(&bp, &data)?;

    let restore_corrupt = store.restore_checkpoint(&root.join("concurrent_0"), &params, 8);
    assert!(
        restore_corrupt.is_err(),
        "corrupted blob must fail SHA-256 verification on restore"
    );

    let _ = fs::remove_dir_all(&root);
    Ok(())
}

#[test]
fn reproducible_adacheck_benchmark_heterogeneous_vs_dense_control() -> Result<()> {
    // 262,144 parameters (~1.00 MiB f32 weights, 3.00 MiB per worker AdamW state, 4 workers =>
    // 15.00 MiB logical state per checkpoint, 75.00 MiB across 5 checkpoints).
    let n = 262_144usize;
    let params = synthetic_params(n);
    let rounds = 5usize;

    let run_regime =
        |tag: &str, active_workers: usize, active_param_frac: f64| -> Result<(u64, u64, f64, f64)> {
            let root = temp_dir(tag);
            let store = AdaCheckStore::new(root.join("adacheck_store"), 4_096);

            let mut theta: Vec<f32> = (0..n).map(|i| splitmix_f32(i as u64)).collect();
            let mut momentum: Vec<f32> = if active_param_frac >= 0.99 {
                (0..n).map(|i| splitmix_f32(0x1000_0000 + i as u64)).collect()
            } else {
                vec![0.0; n]
            };
            let mut carry: Vec<f32> = if active_param_frac >= 0.99 {
                (0..n).map(|i| splitmix_f32(0x2000_0000 + i as u64)).collect()
            } else {
                vec![0.0; n]
            };
            let merged = vec![true; 128];

            let mut worker_states: Vec<Vec<f32>> = (0..4)
                .map(|w| {
                    (0..3 * n)
                        .map(|i| splitmix_f32(((w as u64 + 1) << 32) ^ (i as u64)))
                        .collect()
                })
                .collect();

            let mut total_logical = 0u64;
            let mut save_ms_sum = 0.0f64;
            let mut restore_ms_sum = 0.0f64;
            let active_elems = ((n as f64) * active_param_frac) as usize;

            for r in 1..=rounds {
                for i in 0..active_elems {
                    let d = (r as f32) * 1e-5 + (i as f32) * 1e-9;
                    theta[i] += d;
                    momentum[i] = 0.9 * momentum[i] + d;
                    if active_param_frac >= 0.99 {
                        carry[i] += d * 0.01;
                    }
                }
                for w in 0..active_workers {
                    for i in 0..(active_elems * 3) {
                        worker_states[w][i] += (r as f32) * 1e-5 + (w as f32) * 1e-6;
                    }
                }

                let mut inner = HashMap::new();
                for w in 0..4 {
                    inner.insert(
                        format!("worker-{w}"),
                        Inner {
                            t: if w < active_workers { r as u64 } else { 1 },
                            data: Arc::new(net::pack(&worker_states[w])),
                        },
                    );
                }

                let ckpt_dir = root.join(format!("round_{r}"));
                let t_save = Instant::now();
                let st = store.save_checkpoint(
                    &ckpt_dir,
                    &theta,
                    &momentum,
                    &carry,
                    &merged,
                    r as u64,
                    &inner,
                )?;
                save_ms_sum += t_save.elapsed().as_secs_f64() * 1_000.0;
                total_logical += st.logical_bytes;

                let (restored, rst) = store.restore_checkpoint(&ckpt_dir, &params, 128)?;
                restore_ms_sum += rst.restore_ms;
                assert_f32_bits_eq(&restored.theta, &theta);
            }

            let physical = store.physical_blob_bytes()?;
            let _ = fs::remove_dir_all(&root);
            Ok((
                total_logical,
                physical,
                save_ms_sum / rounds as f64,
                restore_ms_sum / rounds as f64,
            ))
        };

    let (het_log, het_phys, het_save_ms, het_rest_ms) = run_regime("het", 2, 0.5)?;
    let (den_log, den_phys, den_save_ms, den_rest_ms) = run_regime("dense", 4, 1.0)?;

    let het_ratio = het_log as f64 / het_phys as f64;
    let den_ratio = den_log as f64 / den_phys as f64;
    println!(
        "\n[AdaCheck HIP-1349 5-checkpoint x 4-worker benchmark (N=262,144 params)]\n  \
         Heterogeneous/staggered (2/4 active workers, 50% active params): logical={:.2}MiB  physical={:.2}MiB  dedup={het_ratio:.2}x  avg_save={het_save_ms:.1}ms  avg_restore={het_rest_ms:.1}ms\n  \
         Dense all-worker full mutation (negative control):               logical={:.2}MiB  physical={:.2}MiB  dedup={den_ratio:.2}x  avg_save={den_save_ms:.1}ms  avg_restore={den_rest_ms:.1}ms\n",
        het_log as f64 / (1024.0 * 1024.0),
        het_phys as f64 / (1024.0 * 1024.0),
        den_log as f64 / (1024.0 * 1024.0),
        den_phys as f64 / (1024.0 * 1024.0),
    );
    assert!(
        het_ratio > 2.2,
        "heterogeneous DiLoCo checkpoint deduplication ratio {het_ratio:.2}x below expected >2.2x"
    );
    assert!(
        den_ratio < 1.05,
        "dense full-mutation control should show ~1.00x ratio ({den_ratio:.2}x)"
    );
    Ok(())
}
