//! The crate's canonical JSON is the Python package's, byte for byte, and a WAL that package
//! wrote verifies and resumes here. `tests/vectors/python.json` and `tests/vectors/wal.jsonl`
//! are hanzo-research (Python) 0.2.1's own output (`tests/vectors/gen.py`), where 0.2.1 of this
//! crate wrote -0.0 as `-0`, 1e21 as 22 digits, 1e-7 as `0.0000001` and 2^53+1 as itself.

use std::path::Path;

use hanzo_research::lineage::ResearchEvent;
use hanzo_research::{rfc8785_canonicalize, rfc8785_digest, ResearchWal};
use serde_json::{json, Value};

const WAL: &str = include_str!("vectors/wal.jsonl");

fn vectors() -> Value {
    serde_json::from_str(include_str!("vectors/python.json")).unwrap()
}

fn canonical<T: serde::Serialize>(v: &T) -> String {
    String::from_utf8(rfc8785_canonicalize(v).unwrap()).unwrap()
}

#[test]
fn numbers_are_the_python_packages_bytes() {
    let doc = vectors();
    let numbers = doc["numbers"].as_array().unwrap();
    assert_eq!(numbers.len(), 32);
    for n in numbers {
        let text = n["text"].as_str().unwrap();
        let v: Value = serde_json::from_str(text).unwrap();
        assert_eq!(canonical(&v), n["canonical"].as_str().unwrap(), "{text}");
        assert_eq!(
            rfc8785_digest(&v).unwrap(),
            n["digest"].as_str().unwrap(),
            "{text}"
        );
    }
}

#[test]
fn typed_numbers_write_as_their_text_does() {
    let doc = vectors();
    let want = |text: &str| {
        doc["numbers"]
            .as_array()
            .unwrap()
            .iter()
            .find(|n| n["text"] == text)
            .unwrap()["canonical"]
            .as_str()
            .unwrap()
            .to_string()
    };
    assert_eq!(canonical(&-0.0f64), want("-0.0"));
    assert_eq!(canonical(&1e21f64), want("1e21"));
    assert_eq!(canonical(&1e-7f64), want("1e-7"));
    assert_eq!(canonical(&9007199254740993u64), want("9007199254740993"));
    assert_eq!(canonical(&-9007199254740993i64), want("-9007199254740993"));
    assert_eq!(canonical(&u64::MAX), want("18446744073709551615"));
    assert_eq!(canonical(&100.0f64), want("100.0"));
    assert_eq!(canonical(&-3.0f32), want("-3.0"));
    assert_eq!(canonical(&5e-324f64), want("5e-324"));
}

#[test]
fn payloads_digest_as_the_python_package_digests_them() {
    for p in vectors()["payloads"].as_array().unwrap() {
        assert_eq!(canonical(&p["value"]), p["canonical"].as_str().unwrap());
        assert_eq!(
            rfc8785_digest(&p["value"]).unwrap(),
            p["digest"].as_str().unwrap()
        );
    }
}

#[test]
fn a_wal_the_python_package_wrote_verifies() {
    // Each line chains to the digest of the line before and its payload hashes to its
    // payload_hash: the Python package's `verify`, line by line.
    let mut last = "0".repeat(64);
    for (i, line) in WAL.lines().enumerate() {
        let e: ResearchEvent = serde_json::from_str(line).unwrap();
        assert_eq!(e.sequence, i as u64 + 1);
        assert_eq!(e.previous_event_hash, last, "line {}", i + 1);
        let again = ResearchEvent::new(
            e.run_id.clone(),
            e.worker_id.clone(),
            e.event_id.clone(),
            e.sequence,
            last.clone(),
            e.payload.clone(),
            e.created_at.clone(),
        )
        .unwrap();
        assert_eq!(again.payload_hash, e.payload_hash, "line {}", i + 1);
        last = e.digest().unwrap();
    }
    assert_eq!(last, vectors()["wal"]["last"].as_str().unwrap());
}

#[test]
fn a_wal_the_python_package_wrote_resumes() {
    let dir =
        std::env::temp_dir().join(format!("hanzo-research-python-wal-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("wal.jsonl"), WAL).unwrap();

    let mut wal = ResearchWal::open(Path::new(&dir), "run-1").unwrap();
    assert_eq!(wal.sequence(), 3);
    assert_eq!(
        wal.last_event_hash(),
        vectors()["wal"]["last"].as_str().unwrap()
    );
    // Line 1's payload, built here from Rust values, hashes as the Python package hashed it.
    let e = wal
        .append(
            "step",
            &json!({"step": 1, "loss": -0.0, "tokens": 9007199254740993u64}),
        )
        .unwrap();
    assert_eq!(e.sequence, 4);
    assert_eq!(
        e.previous_event_hash,
        vectors()["wal"]["last"].as_str().unwrap()
    );
    let first: ResearchEvent = serde_json::from_str(WAL.lines().next().unwrap()).unwrap();
    assert_eq!(e.payload_hash, first.payload_hash);
    let _ = std::fs::remove_dir_all(&dir);
}
