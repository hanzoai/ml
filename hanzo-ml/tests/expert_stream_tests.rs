// Disk-streaming LRU expert cache: bit-exact vs resident + cache mechanics.
//
// The load-bearing guarantee: a streamed expert is the identical file bytes the resident slice
// would hold, so `indexed_moe_forward` produces bit-identical output; only the fetch path differs.

use hanzo_ml::quantized::expert_stream::ExpertStreamBank;
use hanzo_ml::quantized::gguf_file::TensorInfo;
use hanzo_ml::quantized::{GgmlDType, QStorage, QTensor};
use hanzo_ml::{Device, Result, Tensor};

const E: usize = 6; // experts in the bank
const N: usize = 8; // expert out dim
const K: usize = 64; // expert in dim (2 Q8_0 blocks)
const T: usize = 4; // tokens
const TOPK: usize = 2;

/// Build a resident Q8_0 expert bank `[E, N, K]`, plus its raw bytes written to a temp file so a
/// streaming bank can be opened over the same bytes.
fn fixture() -> Result<(Device, QTensor, std::path::PathBuf, usize)> {
    let dev = Device::Cpu;
    let src = Tensor::randn(0f32, 1f32, (E, N, K), &dev)?;
    let resident = QTensor::quantize(&src, GgmlDType::Q8_0)?;
    let bytes = resident.data()?.to_vec();
    let expert_bytes = bytes.len() / E;
    let path = std::env::temp_dir().join(format!(
        "hanzo_stream_{}_{:p}.bin",
        std::process::id(),
        &bytes as *const _
    ));
    std::fs::write(&path, &bytes)?;
    Ok((dev, resident, path, expert_bytes))
}

fn routing(dev: &Device) -> Result<(Tensor, Tensor)> {
    // Shared gate/up-style input: x is [T, 1, K]; router picks TOPK of E experts per token.
    let x = Tensor::randn(0f32, 1f32, (T, 1, K), dev)?;
    let ids: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 1, 0]; // T*TOPK, spans all experts
    let ids = Tensor::from_vec(ids, (T, TOPK), dev)?;
    Ok((x, ids))
}

fn stream_qtensor(path: &std::path::Path, expert_bytes: usize, name: &str) -> Result<(QTensor, std::sync::Arc<ExpertStreamBank>)> {
    let bank = ExpertStreamBank::open(
        name.to_string(),
        path,
        0,
        expert_bytes,
        E,
        GgmlDType::Q8_0,
        N,
        K,
    )?;
    let qt = QTensor::new(QStorage::Stream(bank.clone()), (E, N, K))?;
    Ok((qt, bank))
}

#[test]
fn streamed_moe_is_bit_exact_vs_resident() -> Result<()> {
    let (dev, resident, path, _expert_bytes) = fixture()?;
    let (x, ids) = routing(&dev)?;

    // Resident path via QStorage::Cpu -> the generic `_` arm.
    let yr = resident.indexed_moe_forward(&x, &ids)?.flatten_all()?.to_vec1::<f32>()?;

    // Streaming path via read_stream (the loader seam) -> the `Stream` arm.
    let info = TensorInfo {
        ggml_dtype: GgmlDType::Q8_0,
        shape: (E, N, K).into(),
        offset: 0,
    };
    let streamed = info.read_stream(&path, 0, "blk.0.ffn_gate_exps.weight")?;
    let ys = streamed.indexed_moe_forward(&x, &ids)?.flatten_all()?.to_vec1::<f32>()?;

    assert_eq!(yr.len(), ys.len());
    for (i, (a, b)) in yr.iter().zip(ys.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "streamed vs resident differ at {i}: {a} != {b}"
        );
    }
    let _ = std::fs::remove_file(&path);
    Ok(())
}

#[test]
fn lru_cap_bounds_resident_and_stays_bit_exact() -> Result<()> {
    let (dev, resident, path, expert_bytes) = fixture()?;
    let (x, ids) = routing(&dev)?;
    let yr = resident.indexed_moe_forward(&x, &ids)?.flatten_all()?.to_vec1::<f32>()?;

    let (qt, bank) = stream_qtensor(&path, expert_bytes, "cap.bank")?;
    bank.set_cap(1); // pathologically small: at most 1 non-pinned expert resident

    // Two forwards touch all E experts repeatedly; the cache must never exceed the cap...
    for _ in 0..3 {
        let ys = qt.indexed_moe_forward(&x, &ids)?.flatten_all()?.to_vec1::<f32>()?;
        // ...and correctness must be independent of the cap (streamed == resident every time).
        for (a, b) in yr.iter().zip(ys.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
    let (pinned, cached, cap, _hits, _misses) = bank.stats();
    assert_eq!(pinned, 0);
    assert_eq!(cap, 1);
    assert!(cached <= 1, "LRU held {cached} experts, cap was 1");
    assert!(bank.resident_bytes() <= expert_bytes, "resident exceeded 1 expert");
    let _ = std::fs::remove_file(&path);
    Ok(())
}

#[test]
fn pinned_expert_is_a_hit_and_survives_eviction() -> Result<()> {
    let (_dev, _resident, path, expert_bytes) = fixture()?;
    let (qt, bank) = stream_qtensor(&path, expert_bytes, "pin.bank")?;
    bank.set_cap(1);
    bank.pin(3)?; // pin expert 3 into the hot-set

    let dev = Device::Cpu;
    // Drive many forwards that route through OTHER experts, thrashing the size-1 LRU.
    let x = Tensor::randn(0f32, 1f32, (2, 1, K), &dev)?;
    let ids = Tensor::from_vec(vec![0u32, 1, 2, 4], (2, 2), &dev)?;
    for _ in 0..5 {
        let _ = qt.indexed_moe_forward(&x, &ids)?;
    }
    let (pinned_before, _c, _cap, hits_before, _m) = bank.stats();
    assert_eq!(pinned_before, 1, "expert 3 should stay pinned through eviction");

    // Fetching the pinned expert is a cache HIT (no disk read), regardless of LRU pressure.
    let _ = bank.fetch(3)?;
    let (_p, _c, _cap, hits_after, _m2) = bank.stats();
    assert_eq!(hits_after, hits_before + 1, "pinned fetch must be a hit");
    let _ = std::fs::remove_file(&path);
    Ok(())
}

#[test]
fn usage_sidecar_learns_the_hot_expert() -> Result<()> {
    use hanzo_ml::quantized::expert_stream;
    let (_dev, _resident, path, expert_bytes) = fixture()?;
    let (_qt, bank) = stream_qtensor(&path, expert_bytes, "learn.bank")?;

    // Route expert 4 much more than the rest, then persist the learned usage.
    for _ in 0..10 {
        let _ = bank.fetch(4)?;
    }
    let _ = bank.fetch(0)?;
    let _ = bank.fetch(1)?;

    let sidecar = std::env::temp_dir().join(format!("hanzo_usage_{}.txt", std::process::id()));
    let _ = std::fs::remove_file(&sidecar);
    expert_stream::set_usage_sidecar(sidecar.clone());
    expert_stream::save_usage()?;

    let text = std::fs::read_to_string(&sidecar)?;
    // Among this bank's lines, expert 4 must carry the highest count.
    let mut best: Option<(u32, u64)> = None;
    for line in text.lines() {
        let mut it = line.split_whitespace();
        if it.next() == Some("learn.bank") {
            if let (Some(e), Some(c)) = (it.next(), it.next()) {
                let (e, c) = (e.parse::<u32>().unwrap(), c.parse::<u64>().unwrap());
                if best.map(|(_, bc)| c > bc).unwrap_or(true) {
                    best = Some((e, c));
                }
            }
        }
    }
    assert_eq!(best.map(|(e, _)| e), Some(4), "hottest expert should be 4");
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&sidecar);
    Ok(())
}
