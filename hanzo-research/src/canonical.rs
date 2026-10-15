//! RFC 8785: JSON Canonicalization Scheme (JCS) implementation and digest utilities.
//!
//! Provides deterministic canonical formatting for JSON payloads and calculates SHA-256 digests
//! across canonical bytes to guarantee reproducible hashes for artifacts, environment specs,
//! gates, and lineage objects. Keys sort by UTF-16 code units, strings escape by RFC 8785's
//! table, and every number is written as ECMAScript writes the IEEE 754 double it denotes, so
//! these bytes are the Python package's (`hanzo_research.canonical`): `tests/canonical.rs` holds
//! them to the bytes that package wrote (`tests/vectors/python.json`).

use serde::ser::Error as _;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::io::Write;

/// Canonicalize a `serde_json::Value` according to RFC 8785 (JSON Canonicalization Scheme).
/// Refuses a number that denotes no double (only `serde_json`'s `arbitrary_precision` holds one).
pub fn canonicalize_value(val: &Value, out: &mut Vec<u8>) -> Result<(), serde_json::Error> {
    match val {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(b) => {
            if *b {
                out.extend_from_slice(b"true");
            } else {
                out.extend_from_slice(b"false");
            }
        }
        Value::Number(n) => match n.as_f64() {
            Some(x) => number(x, out),
            None => {
                return Err(serde_json::Error::custom(format!(
                    "{n} denotes no double, so it has no RFC 8785 form"
                )))
            }
        },
        Value::String(s) => {
            canonicalize_string(s, out);
        }
        Value::Array(arr) => {
            out.push(b'[');
            for (i, v) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                canonicalize_value(v, out)?;
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
                canonicalize_value(v, out)?;
            }
            out.push(b'}');
        }
    }
    Ok(())
}

/// A number as ECMAScript's `Number.prototype.toString` writes it (RFC 8785 section 3.2.2.3).
/// Every JSON number is the IEEE 754 double it denotes, so an integer past 2^53 is written as
/// the double it rounds to, -0 as `0`, 1e21 as `1e+21` and 1e-7 as `1e-7`.
fn number(x: f64, out: &mut Vec<u8>) {
    if x == 0.0 {
        out.push(b'0');
        return;
    }
    if x < 0.0 {
        out.push(b'-');
    }
    // The shortest digits that read back as x, the nearer of two when both do and the even
    // one at a tie, as ECMAScript chooses them: Ryu's (std's LowerExp breaks a tie upward,
    // 683210194955870.3 where ECMAScript writes 683210194955870.2). Ryu writes `d.ddd`,
    // `0.000ddd` or `d.ddde<exp>`; x = 0.<digits> * 10^n.
    let mut buf = ryu::Buffer::new();
    let text = buf.format_finite(x.abs());
    let (mantissa, exp) = text.split_once('e').unwrap_or((text, "0"));
    let (whole, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let every: Vec<u8> = whole.bytes().chain(frac.bytes()).collect();
    let lead = every.iter().take_while(|&&d| d == b'0').count();
    let mut digits = &every[lead..];
    while digits.last() == Some(&b'0') {
        digits = &digits[..digits.len() - 1];
    }
    let exp: i32 = exp.parse().expect("Ryu writes a decimal exponent");
    let n = whole.len() as i32 + exp - lead as i32;
    let k = digits.len() as i32;
    if k <= n && n <= 21 {
        out.extend_from_slice(digits);
        out.resize(out.len() + (n - k) as usize, b'0');
    } else if 0 < n && n <= 21 {
        out.extend_from_slice(&digits[..n as usize]);
        out.push(b'.');
        out.extend_from_slice(&digits[n as usize..]);
    } else if -6 < n && n <= 0 {
        out.extend_from_slice(b"0.");
        out.resize(out.len() + (-n) as usize, b'0');
        out.extend_from_slice(digits);
    } else {
        out.push(digits[0]);
        if k > 1 {
            out.push(b'.');
            out.extend_from_slice(&digits[1..]);
        }
        let _ = write!(out, "e{}{}", if n > 1 { '+' } else { '-' }, (n - 1).abs());
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
    canonicalize_value(&json_val, &mut out)?;
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
    fn rfc8785_sample() {
        // RFC 8785 section 3.2.2's input and its canonical form, as the RFC prints them.
        let v: Value = serde_json::from_str(
            r#"{"numbers":[333333333.33333329,1E30,4.50,2e-3,0.000000000000000000000000001],"string":"\u20ac$\u000F\u000aA'\u0042\u0022\u005c\\\"\/","literals":[null,true,false]}"#,
        )
        .unwrap();
        assert_eq!(
            String::from_utf8(rfc8785_canonicalize(&v).unwrap()).unwrap(),
            r#"{"literals":[null,true,false],"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],"string":"€$\u000f\nA'B\"\\\\\"/"}"#
        );
    }

    #[test]
    fn test_rfc8785_digest_deterministic() {
        let v1 = json!({"foo": "bar", "num": 42});
        let v2 = json!({"num": 42, "foo": "bar"});
        assert_eq!(rfc8785_digest(&v1).unwrap(), rfc8785_digest(&v2).unwrap());
    }
}
