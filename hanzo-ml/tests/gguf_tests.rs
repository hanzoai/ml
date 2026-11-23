//! Regression tests for the allocation caps added in huggingface/candle#3533.

use hanzo_ml::quantized::gguf_file::Content;
use hanzo_ml::Device;
use std::io::Cursor;

const GGUF_MAGIC: [u8; 4] = *b"GGUF";
const GGUF_V3: u32 = 3;

fn header(tensor_count: u64, metadata_kv_count: u64) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC);
    buf.extend_from_slice(&GGUF_V3.to_le_bytes());
    buf.extend_from_slice(&tensor_count.to_le_bytes());
    buf.extend_from_slice(&metadata_kv_count.to_le_bytes());
    buf
}

fn length_prefixed(s: &[u8]) -> Vec<u8> {
    let mut buf = (s.len() as u64).to_le_bytes().to_vec();
    buf.extend_from_slice(s);
    buf
}

/// Pad past the upfront `count * min_per_entry` sufficient-bytes check so
/// the inner per-field cap is what fires.
fn pad(buf: &mut Vec<u8>, n: usize) {
    buf.resize(buf.len() + n, 0);
}

fn assert_rejects(buf: Vec<u8>, msg_contains: &str) {
    let mut cursor = Cursor::new(buf);
    let err = Content::read(&mut cursor).expect_err("expected Err");
    let msg = format!("{err}");
    assert!(msg.contains(msg_contains), "unexpected error: {msg}");
}

#[test]
fn empty_header_loads() {
    let mut cursor = Cursor::new(header(0, 0));
    let content = Content::read(&mut cursor).expect("empty header should parse");
    assert!(content.metadata.is_empty());
    assert!(content.tensor_infos.is_empty());
}

#[test]
fn rejects_oversized_metadata_kv_count() {
    assert_rejects(header(0, 1u64 << 31), "metadata_kv_count");
}

#[test]
fn rejects_oversized_tensor_count() {
    assert_rejects(header(1u64 << 31, 0), "tensor_count");
}

#[test]
fn rejects_oversized_string_length() {
    let mut buf = header(1, 0);
    buf.extend_from_slice(&(1u64 << 31).to_le_bytes());
    pad(&mut buf, 64);
    assert_rejects(buf, "string length");
}

#[test]
fn rejects_oversized_array_length() {
    let mut buf = header(0, 1);
    buf.extend(length_prefixed(b"k"));
    buf.extend_from_slice(&9u32.to_le_bytes()); // value_type = Array
    buf.extend_from_slice(&0u32.to_le_bytes()); // inner type = U8
    buf.extend_from_slice(&(1u64 << 31).to_le_bytes()); // element count
    pad(&mut buf, 64);
    assert_rejects(buf, "array length");
}

#[test]
fn rejects_oversized_n_dimensions() {
    let mut buf = header(1, 0);
    buf.extend(length_prefixed(b"t"));
    buf.extend_from_slice(&u32::MAX.to_le_bytes());
    pad(&mut buf, 64);
    assert_rejects(buf, "dimensions");
}

#[test]
fn rejects_deeply_nested_arrays() {
    // 256 nested arrays of arrays, well above GGUF_MAX_VALUE_DEPTH (64).
    // Each layer is type=Array(9) + inner_type(u32) + len=1(u64) = 16 bytes
    // except the innermost which terminates with a U8 (type=0) + len=0(u64).
    let mut buf = header(0, 1);
    buf.extend(length_prefixed(b"k"));
    buf.extend_from_slice(&9u32.to_le_bytes()); // outer value_type = Array
    for _ in 0..256 {
        buf.extend_from_slice(&9u32.to_le_bytes()); // inner type = Array
        buf.extend_from_slice(&1u64.to_le_bytes()); // len = 1
    }
    buf.extend_from_slice(&0u32.to_le_bytes()); // terminal inner type = U8
    buf.extend_from_slice(&0u64.to_le_bytes()); // terminal len = 0
    assert_rejects(buf, "nesting depth");
}

#[test]
fn empty_v1_header_loads() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC);
    buf.extend_from_slice(&1u32.to_le_bytes()); // version 1: u32 length prefixes
    buf.extend_from_slice(&0u32.to_le_bytes()); // tensor_count
    buf.extend_from_slice(&0u32.to_le_bytes()); // metadata_kv_count
    let mut cursor = Cursor::new(buf);
    Content::read(&mut cursor).expect("empty v1 header should parse");
}

#[test]
fn rejects_tensor_size_exceeding_file() {
    // Create a gguf tensor that claims shape [1_073_741_824] F32 (4 GiB), but has no actual tensor data.
    let mut buf = header(1, 0);
    buf.extend(length_prefixed(b"t"));
    buf.extend_from_slice(&1u32.to_le_bytes()); // n dims
    buf.extend_from_slice(&1_073_741_824u64.to_le_bytes()); // 1 GiB elements
    buf.extend_from_slice(&0u32.to_le_bytes()); // F32
    buf.extend_from_slice(&0u64.to_le_bytes()); // no offset
    let mut cursor = Cursor::new(buf);
    let content = Content::read(&mut cursor).expect("header should parse");
    let err = content
        .tensor(&mut cursor, "t", &Device::Cpu)
        .expect_err("expected Err from oversized tensor load");
    let msg = format!("{err}");
    assert!(msg.contains("remaining"), "unexpected error: {msg}");
}

#[test]
fn rejects_string_length_above_remaining_file_bytes() {
    let mut buf = header(1, 0);
    buf.extend_from_slice(&(1u64 << 20).to_le_bytes()); // 1 MB, below cap, above file size
    pad(&mut buf, 64);
    assert_rejects(buf, "string length");
}

#[test]
fn legacy_type42_pq2_0_remaps_while_modern_type42_loads_as_q2_0() {
    use hanzo_ml::quantized::{BlockPQ2_0, BlockQ2_0, GgmlDType, GgmlType};

    fn append_tensor_info(buf: &mut Vec<u8>, name: &[u8], ne0: u64, ggml_type: u32, offset: u64) {
        buf.extend(length_prefixed(name));
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&ne0.to_le_bytes());
        buf.extend_from_slice(&ggml_type.to_le_bytes());
        buf.extend_from_slice(&offset.to_le_bytes());
    }

    let pq_vals: Vec<f32> = (0..128)
        .map(|i| (((i * 3 + 1) % 3) - 1) as f32 * 0.5)
        .collect();
    let mut pq_blk = [BlockPQ2_0::zeros(); 1];
    BlockPQ2_0::from_float(&pq_vals, &mut pq_blk);
    let pq_bytes = unsafe { std::slice::from_raw_parts(pq_blk.as_ptr() as *const u8, 34) }.to_vec();

    let q2_vals: Vec<f32> = (0..128)
        .map(|i| (((i * 5 + 2) % 3) - 1) as f32 * 0.25)
        .collect();
    let mut q2_blk = [BlockQ2_0::zeros(); 2];
    BlockQ2_0::from_float(&q2_vals, &mut q2_blk);
    let q2_bytes = unsafe { std::slice::from_raw_parts(q2_blk.as_ptr() as *const u8, 36) }.to_vec();

    // Case 1: Single legacy tensor at EOF (34 bytes < 36 bytes needed for Q2_0).
    let mut legacy_eof = header(1, 0);
    append_tensor_info(&mut legacy_eof, b"w_legacy", 128, 42, 0);
    let align_pad = (32 - (legacy_eof.len() % 32)) % 32;
    pad(&mut legacy_eof, align_pad);
    legacy_eof.extend_from_slice(&pq_bytes);

    let mut cur1 = Cursor::new(legacy_eof);
    let c1 = Content::read(&mut cur1).expect("legacy GGUF header should parse");
    assert_eq!(c1.tensor_infos["w_legacy"].ggml_dtype, GgmlDType::PQ2_0);
    let qt1 = c1
        .tensor(&mut cur1, "w_legacy", &Device::Cpu)
        .expect("legacy PQ2_0 tensor should load");
    assert_eq!(qt1.dtype(), GgmlDType::PQ2_0);
    let deq1 = qt1
        .dequantize(&Device::Cpu)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(deq1, pq_vals);

    // Case 2: Legacy GGUF with prism.hadamard.version = 6 and 32-byte aligned neighbor offset (64 bytes span).
    let mut legacy_meta = header(2, 1);
    legacy_meta.extend(length_prefixed(b"prism.hadamard.version"));
    legacy_meta.extend_from_slice(&4u32.to_le_bytes()); // U32
    legacy_meta.extend_from_slice(&6u32.to_le_bytes());
    append_tensor_info(&mut legacy_meta, b"w0", 128, 42, 0);
    append_tensor_info(&mut legacy_meta, b"w1", 128, 42, 64);
    let align_pad2 = (32 - (legacy_meta.len() % 32)) % 32;
    pad(&mut legacy_meta, align_pad2);
    legacy_meta.extend_from_slice(&pq_bytes);
    pad(&mut legacy_meta, 64 - 34);
    legacy_meta.extend_from_slice(&pq_bytes);

    let mut cur2 = Cursor::new(legacy_meta);
    let c2 = Content::read(&mut cur2).expect("legacy v6 GGUF should parse");
    assert_eq!(c2.tensor_infos["w0"].ggml_dtype, GgmlDType::PQ2_0);
    assert_eq!(c2.tensor_infos["w1"].ggml_dtype, GgmlDType::PQ2_0);

    // Case 3: Modern type = 42 tensor (36 bytes for 128 elements) loads as Q2_0.
    let mut modern = header(1, 0);
    append_tensor_info(&mut modern, b"w_q2", 128, 42, 0);
    let align_pad3 = (32 - (modern.len() % 32)) % 32;
    pad(&mut modern, align_pad3);
    modern.extend_from_slice(&q2_bytes);

    let mut cur3 = Cursor::new(modern);
    let c3 = Content::read(&mut cur3).expect("modern Q2_0 GGUF should parse");
    assert_eq!(c3.tensor_infos["w_q2"].ggml_dtype, GgmlDType::Q2_0);
    let qt3 = c3
        .tensor(&mut cur3, "w_q2", &Device::Cpu)
        .expect("modern Q2_0 tensor should load");
    assert_eq!(qt3.dtype(), GgmlDType::Q2_0);
    let deq3 = qt3
        .dequantize(&Device::Cpu)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(deq3, q2_vals);
}
