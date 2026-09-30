//! RFC 8785: JSON Canonicalization Scheme (JCS) implementation and digest utilities.
//!
//! Provides deterministic canonical formatting for JSON payloads and calculates SHA-256 digests
//! across canonical bytes to guarantee reproducible hashes for artifacts, environment specs,
//! gates, and lineage objects.

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::io::Write;

/// Canonicalize a `serde_json::Value` according to RFC 8785 (JSON Canonicalization Scheme).
pub fn canonicalize_value(val: &Value, out: &mut Vec<u8>) {
    match val {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(b) => {
            if *b {
                out.extend_from_slice(b"true");
            } else {
                out.extend_from_slice(b"false");
            }
        }
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                let _ = write!(out, "{i}");
            } else if let Some(u) = n.as_u64() {
                let _ = write!(out, "{u}");
            } else if let Some(f) = n.as_f64() {
                let _ = write!(out, "{f}");
            }
        }
        Value::String(s) => {
            canonicalize_string(s, out);
        }
        Value::Array(arr) => {
            out.push(b'[');
            for (i, v) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                canonicalize_value(v, out);
            }
            out.push(b']');
        }
        Value::Object(map) => {
            out.push(b'{');
            // RFC 8785 requires sorting keys by UTF-16 code units
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|(k1, _), (k2, _)| {
                let u1 = k1.encode_utf16();
                let u2 = k2.encode_utf16();
                u1.cmp(u2)
            });
            for (i, (k, v)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                canonicalize_string(k, out);
                out.push(b':');
                canonicalize_value(v, out);
            }
            out.push(b'}');
        }
    }
}

/// Serialize a string with RFC 8785 escaping rules.
pub fn canonicalize_string(s: &str, out: &mut Vec<u8>) {
    out.push(b'"');
    for c in s.chars() {
        match c {
            '"' => out.extend_from_slice(b"\\\""),
            '\\' => out.extend_from_slice(b"\\\\"),
            '\x08' => out.extend_from_slice(b"\\b"),
            '\x0C' => out.extend_from_slice(b"\\f"),
            '\n' => out.extend_from_slice(b"\\n"),
            '\r' => out.extend_from_slice(b"\\r"),
            '\t' => out.extend_from_slice(b"\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => {
                let mut buf = [0u8; 4];
                let enc = c.encode_utf8(&mut buf);
                out.extend_from_slice(enc.as_bytes());
            }
        }
    }
    out.push(b'"');
}

/// Convert any serializable value into RFC 8785 canonical JSON bytes.
pub fn rfc8785_canonicalize<T: Serialize>(val: &T) -> Result<Vec<u8>, serde_json::Error> {
    let json_val = serde_json::to_value(val)?;
    let mut out = Vec::new();
    canonicalize_value(&json_val, &mut out);
    Ok(out)
}

/// Compute SHA-256 over RFC 8785 canonical JSON bytes and return lowercase hex.
pub fn rfc8785_digest<T: Serialize>(val: &T) -> Result<String, serde_json::Error> {
    let bytes = rfc8785_canonicalize(val)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let hash = hasher.finalize();
    Ok(hash.iter().map(|b| format!("{b:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_rfc8785_sorting() {
        let v = json!({
            "b": 2,
            "a": 1,
            "c": {
                "z": true,
                "x": false
            }
        });
        let bytes = rfc8785_canonicalize(&v).unwrap();
        assert_eq!(
            String::from_utf8(bytes).unwrap(),
            r#"{"a":1,"b":2,"c":{"x":false,"z":true}}"#
        );
    }

    #[test]
    fn test_rfc8785_digest_deterministic() {
        let v1 = json!({"foo": "bar", "num": 42});
        let v2 = json!({"num": 42, "foo": "bar"});
        assert_eq!(rfc8785_digest(&v1).unwrap(), rfc8785_digest(&v2).unwrap());
    }
}
