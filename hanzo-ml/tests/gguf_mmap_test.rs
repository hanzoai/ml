//! mmap weight-streaming for GGUF (antirez ds4_ssd, "RAM as a speed spectrum").
//!
//! The default loader (`Content::read` + `Content::tensor`) `read_exact`s every tensor's bytes into
//! an owned `Vec` -- an 86 GB model needs 86 GB resident. The mmap loader (`Content::read_mmap` +
//! `Content::tensor_mmap`) instead references the quantized blocks *in place* inside the mapped
//! file (`QStorage::Cpu(QMmap)`), so the OS pages weights in/out of the page cache and a model
//! larger than free RAM runs.
//!
//! This test builds a small multi-dtype GGUF on disk and asserts, for each tensor, that the mmap
//! path is (a) bit-identical to the resident path after dequantization, and (b) genuinely no-copy
//! -- its bytes alias the live mapping rather than a freshly allocated `Vec`. (b) is checked via
//! the data path (pointer-in-mapping), not a process-RSS probe, so it is deterministic.

use hanzo_ml::quantized::gguf_file::{self, Content};
use hanzo_ml::quantized::{GgmlDType, QTensor};
use hanzo_ml::{Device, Result, Tensor};
use std::path::{Path, PathBuf};

fn tmp_path() -> PathBuf {
    // Monotonic per-call counter: SystemTime has microsecond resolution on macOS, so two
    // parallel test threads can collide on a timestamp-derived name -- one test's cleanup
    // then deletes the file under the other (ENOENT in read_mmap).
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "hanzo_gguf_mmap_{}_{}.gguf",
        std::process::id(),
        seq
    ))
}

/// A deterministic f32 tensor with enough spread that quantization is non-trivial.
fn sample(rows: usize, cols: usize, dev: &Device) -> Result<Tensor> {
    let n = rows * cols;
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.017).sin() * 1.3 - 0.2)
        .collect();
    Tensor::from_vec(data, (rows, cols), dev)
}

/// Write a GGUF with a quant block type (Q4_0, block align 2), an 8-bit type (Q8_0, align 2), and
/// a plain F32 tensor (align 4 -- exercises the wider-alignment mmap arm). Last dim is a multiple
/// of the QK block size (32) so quantization is valid.
fn write_fixture(path: &Path) -> Result<Vec<&'static str>> {
    let dev = Device::Cpu;
    let q4_0 = QTensor::quantize(&sample(8, 64, &dev)?, GgmlDType::Q4_0)?;
    let q8_0 = QTensor::quantize(&sample(6, 96, &dev)?, GgmlDType::Q8_0)?;
    let f32t = QTensor::quantize(&sample(4, 32, &dev)?, GgmlDType::F32)?;
    let mut file = std::fs::File::create(path)?;
    gguf_file::write(
        &mut file,
        &[],
        &[("blk.q4_0", &q4_0), ("blk.q8_0", &q8_0), ("blk.f32", &f32t)],
    )?;
    file.sync_all()?;
    Ok(vec!["blk.q4_0", "blk.q8_0", "blk.f32"])
}

fn run() -> Result<()> {
    let dev = Device::Cpu;
    let path = tmp_path();
    let names = write_fixture(&path)?;

    // Resident path: header + per-tensor read_exact into owned Vecs.
    let mut file = std::fs::File::open(&path)?;
    let resident = Content::read(&mut file)?;

    // mmap path: header parsed from the mapping; tensor data referenced in place.
    let (mapped, mmap) = Content::read_mmap(&path)?;
    let base = mmap.as_ptr() as usize;
    let map_end = base + mmap.len();

    for name in names {
        let t_res = resident.tensor(&mut file, name, &dev)?;
        let t_mm = mapped.tensor_mmap(&mmap, name, &dev)?;

        assert_eq!(t_res.dtype(), t_mm.dtype(), "{name}: dtype mismatch");
        assert_eq!(t_res.shape(), t_mm.shape(), "{name}: shape mismatch");
        assert_eq!(
            t_res.storage_size_in_bytes(),
            t_mm.storage_size_in_bytes(),
            "{name}: byte size mismatch"
        );

        // (a) Bit-identical dequantization: both paths read the same on-disk bytes through the same
        // decode routine, so the f32 outputs must be equal exactly (not just approximately).
        let v_res = t_res.dequantize(&dev)?.flatten_all()?.to_vec1::<f32>()?;
        let v_mm = t_mm.dequantize(&dev)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(
            v_res, v_mm,
            "{name}: dequantized values differ between resident and mmap"
        );

        // (b) No-copy: the mmap tensor's bytes alias the live mapping. If `from_mmap` had fallen
        // back to `to_vec` (or copied), this pointer would be outside [base, map_end).
        let d_mm = t_mm.data()?;
        let p_mm = d_mm.as_ptr() as usize;
        assert!(
            p_mm >= base && p_mm + d_mm.len() <= map_end,
            "{name}: mmap tensor data is NOT inside the mapping (it was copied resident): \
             ptr={p_mm:#x}, len={}, mapping=[{base:#x},{map_end:#x})",
            d_mm.len()
        );

        // Contrast: the resident tensor owns a fresh buffer outside the mapping.
        let d_res = t_res.data()?;
        let p_res = d_res.as_ptr() as usize;
        assert!(
            p_res < base || p_res >= map_end,
            "{name}: resident tensor unexpectedly aliased the mapping"
        );
    }

    std::fs::remove_file(&path).ok();
    Ok(())
}

#[test]
fn gguf_mmap_no_copy_matches_resident() {
    run().expect("mmap GGUF load must match the resident load bit-for-bit and stay no-copy");
}

/// The whole point of the feature: only the header is materialized up front. After `read_mmap`,
/// the resident byte cost is the (KB-sized) parsed metadata, not the tensor data -- the tensor
/// blocks remain in the mapping until a forward pass touches them. We assert this structurally:
/// every CPU tensor's storage points into the mapping (proved per-tensor above) and constructing
/// all of them allocates no owned weight buffers.
#[test]
fn gguf_mmap_tensors_reference_mapping_not_heap() {
    let dev = Device::Cpu;
    let path = tmp_path();
    let names = write_fixture(&path).expect("write fixture");

    let (mapped, mmap) = Content::read_mmap(&path).expect("read_mmap");
    let base = mmap.as_ptr() as usize;
    let map_end = base + mmap.len();

    let mut total_mapped_bytes = 0usize;
    for name in names {
        let t = mapped.tensor_mmap(&mmap, name, &dev).expect("tensor_mmap");
        let d = t.data().expect("data");
        let p = d.as_ptr() as usize;
        assert!(
            p >= base && p + d.len() <= map_end,
            "{name}: tensor not backed by the mapping"
        );
        total_mapped_bytes += d.len();
    }
    assert!(
        total_mapped_bytes > 0,
        "expected non-empty mapped tensor data"
    );

    std::fs::remove_file(&path).ok();
}
