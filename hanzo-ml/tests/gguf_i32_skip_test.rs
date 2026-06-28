//! Validates that the GGUF loader skips the integer auxiliary tensors (I8/I16/I32/I64) -- e.g.
//! deepseek4's `ffn_gate_tid2eid` expert-routing maps (GGML_TYPE_I32 = 26) -- instead of hard-failing
//! the whole model load with "unknown dtype for tensor 26".
//!
//! Run against a real model by setting HANZO_TEST_GGUF to its path (only the headers are read, not the
//! 86 GB of tensor data, so it's fast). Skips cleanly when the env var / file is absent.

use hanzo_ml::quantized::gguf_file::Content;

#[test]
fn gguf_header_loads_skipping_integer_tensors() {
    let path = match std::env::var("HANZO_TEST_GGUF") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("skip: set HANZO_TEST_GGUF to a model path to exercise the I32-skip loader");
            return;
        }
    };
    let mut f = std::fs::File::open(&path).expect("open gguf");
    // Pre-fix this returned Err("unknown dtype for tensor 26") on deepseek4 GGUFs; post-fix the I32
    // routing tensors are skipped and the header parses.
    let content = Content::read(&mut f).expect("gguf header must parse (I32 tensors skipped)");
    let n = content.tensor_infos.len();
    assert!(n > 0, "expected weight tensors after skipping integer auxiliaries");
    // None of the kept tensors are the skipped integer routing maps.
    let leaked_int = content
        .tensor_infos
        .keys()
        .find(|k| k.contains("tid2eid"))
        .cloned();
    assert!(
        leaked_int.is_none(),
        "integer routing tensor was not skipped: {leaked_int:?}"
    );
    println!("OK: {path} header parsed, {n} weight tensors kept, integer auxiliaries skipped");
}
