//! Canonical JSON: RFC 8785 (JCS). Keys sorted by UTF-16 code units, no whitespace, strings
//! escaped as ECMAScript's `JSON.stringify`, numbers as ECMAScript's `Number.prototype.toString`.
//! Input must be I-JSON (RFC 7493): no duplicate names, no NaN or infinity, no integer an IEEE
//! 754 double cannot hold. Anything else is refused, never rewritten.

use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{self, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::fmt;

/// Why a value has no canonical form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "canonical json: {}", self.0)
    }
}

impl std::error::Error for Error {}

impl ser::Error for Error {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        Error(msg.to_string())
    }
}

/// The canonical bytes of `v`.
pub fn to_vec<T: Serialize + ?Sized>(v: &T) -> Result<Vec<u8>, Error> {
    v.serialize(Finite)?;
    let v = serde_json::to_value(v).map_err(|e| Error(e.to_string()))?;
    let mut out = Vec::new();
    write(&v, &mut out)?;
    Ok(out)
}

/// The canonical bytes of JSON text `bytes`, refusing duplicate names.
pub fn bytes(bytes: &[u8]) -> Result<Vec<u8>, Error> {
    let v = parse(bytes)?;
    let mut out = Vec::new();
    write(&v, &mut out)?;
    Ok(out)
}

/// JSON text parsed as I-JSON: a duplicate name anywhere is refused.
pub fn parse(bytes: &[u8]) -> Result<Value, Error> {
    let mut d = serde_json::Deserializer::from_slice(bytes);
    let v = de::DeserializeSeed::deserialize(Strict, &mut d).map_err(|e| Error(e.to_string()))?;
    d.end().map_err(|e| Error(e.to_string()))?;
    Ok(v.0)
}

/// Lowercase hex SHA-256 of `bytes`: an artifact's address.
pub fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// `sha256:` and the hex SHA-256 of `v`'s canonical bytes: a value's `_hash`.
pub fn hash<T: Serialize + ?Sized>(v: &T) -> Result<String, Error> {
    Ok(format!("sha256:{}", sha256(&to_vec(v)?)))
}

fn write(v: &Value, out: &mut Vec<u8>) -> Result<(), Error> {
    match v {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(true) => out.extend_from_slice(b"true"),
        Value::Bool(false) => out.extend_from_slice(b"false"),
        Value::Number(n) => {
            let x = if let Some(i) = n.as_i64() {
                exact(i as f64, i as f64 as i64 == i && i != i64::MAX, n)?
            } else if let Some(u) = n.as_u64() {
                exact(u as f64, u as f64 as u64 == u && u != u64::MAX, n)?
            } else {
                n.as_f64().ok_or_else(|| Error(format!("number {n}")))?
            };
            number(x, out)?;
        }
        Value::String(s) => string(s, out),
        Value::Array(a) => {
            out.push(b'[');
            for (i, v) in a.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write(v, out)?;
            }
            out.push(b']');
        }
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
            out.push(b'{');
            for (i, k) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                string(k, out);
                out.push(b':');
                write(&m[k], out)?;
            }
            out.push(b'}');
        }
    }
    Ok(())
}

/// `x` when the integer `n` converted to it exactly.
fn exact(x: f64, ok: bool, n: &serde_json::Number) -> Result<f64, Error> {
    if ok {
        Ok(x)
    } else {
        Err(Error(format!("integer {n} is not an IEEE 754 double")))
    }
}

/// ECMAScript's `Number.prototype.toString` of a finite `x` (ECMA-262 §6.1.6.1.20).
fn number(x: f64, out: &mut Vec<u8>) -> Result<(), Error> {
    if !x.is_finite() {
        return Err(Error(format!("{x} is not a JSON number")));
    }
    if x == 0.0 {
        out.push(b'0');
        return Ok(());
    }
    if x < 0.0 {
        out.push(b'-');
    }
    // Ryū's shortest round-trip digits, the closest of them and even on a tie, as ECMAScript
    // chooses them; ryu writes them as `d.ddd`, `ddd.d` or with `e<exp>`.
    let mut buf = ryu::Buffer::new();
    let s = buf.format_finite(x.abs());
    let (mantissa, exp) = s.split_once('e').unwrap_or((s, "0"));
    let exp: i32 = exp.parse().expect("ryu writes an integer exponent");
    let point = mantissa.find('.').unwrap_or(mantissa.len()) as i32;
    let all: Vec<u8> = mantissa.bytes().filter(|&b| b != b'.').collect();
    let lead = all.iter().take_while(|&&b| b == b'0').count();
    let trail = all.iter().rev().take_while(|&&b| b == b'0').count();
    let digits = &all[lead..all.len() - trail];
    // x = 0.d₁d₂…dₖ × 10ⁿ
    let k = digits.len() as i32;
    let n = point - lead as i32 + exp;
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
        let e = n - 1;
        out.extend_from_slice(format!("e{}{}", if e < 0 { '-' } else { '+' }, e.abs()).as_bytes());
    }
    Ok(())
}

fn string(s: &str, out: &mut Vec<u8>) {
    out.push(b'"');
    for c in s.chars() {
        match c {
            '"' => out.extend_from_slice(b"\\\""),
            '\\' => out.extend_from_slice(b"\\\\"),
            '\u{8}' => out.extend_from_slice(b"\\b"),
            '\u{c}' => out.extend_from_slice(b"\\f"),
            '\n' => out.extend_from_slice(b"\\n"),
            '\r' => out.extend_from_slice(b"\\r"),
            '\t' => out.extend_from_slice(b"\\t"),
            c if (c as u32) < 0x20 => {
                out.extend_from_slice(format!("\\u{:04x}", c as u32).as_bytes())
            }
            c => {
                let mut buf = [0u8; 4];
                out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    out.push(b'"');
}

/// A `Value` read with duplicate names refused.
struct Strict;
struct Parsed(Value);

impl<'de> de::DeserializeSeed<'de> for Strict {
    type Value = Parsed;

    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<Parsed, D::Error> {
        d.deserialize_any(Strict)
    }
}

impl<'de> Visitor<'de> for Strict {
    type Value = Parsed;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("JSON")
    }

    fn visit_bool<E>(self, v: bool) -> Result<Parsed, E> {
        Ok(Parsed(Value::Bool(v)))
    }

    fn visit_i64<E>(self, v: i64) -> Result<Parsed, E> {
        Ok(Parsed(Value::from(v)))
    }

    fn visit_u64<E>(self, v: u64) -> Result<Parsed, E> {
        Ok(Parsed(Value::from(v)))
    }

    fn visit_f64<E: de::Error>(self, v: f64) -> Result<Parsed, E> {
        serde_json::Number::from_f64(v)
            .map(|n| Parsed(Value::Number(n)))
            .ok_or_else(|| E::custom(format!("{v} is not a JSON number")))
    }

    fn visit_str<E>(self, v: &str) -> Result<Parsed, E> {
        Ok(Parsed(Value::String(v.to_string())))
    }

    fn visit_string<E>(self, v: String) -> Result<Parsed, E> {
        Ok(Parsed(Value::String(v)))
    }

    fn visit_unit<E>(self) -> Result<Parsed, E> {
        Ok(Parsed(Value::Null))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Parsed, A::Error> {
        let mut out = Vec::new();
        while let Some(v) = a.next_element_seed(Strict)? {
            out.push(v.0);
        }
        Ok(Parsed(Value::Array(out)))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<Parsed, A::Error> {
        let mut out = Map::new();
        while let Some(k) = a.next_key::<String>()? {
            let v = a.next_value_seed(Strict)?;
            if out.insert(k.clone(), v.0).is_some() {
                return Err(de::Error::custom(format!("duplicate name {k:?}")));
            }
        }
        Ok(Parsed(Value::Object(out)))
    }
}

/// A serializer that writes nothing and refuses a non-finite float, which `serde_json` would
/// otherwise turn into `null`.
struct Finite;

impl ser::Serializer for Finite {
    type Ok = ();
    type Error = Error;
    type SerializeSeq = Finite;
    type SerializeTuple = Finite;
    type SerializeTupleStruct = Finite;
    type SerializeTupleVariant = Finite;
    type SerializeMap = Finite;
    type SerializeStruct = Finite;
    type SerializeStructVariant = Finite;

    fn serialize_bool(self, _: bool) -> Result<(), Error> {
        Ok(())
    }
    fn serialize_i8(self, _: i8) -> Result<(), Error> {
        Ok(())
    }
    fn serialize_i16(self, _: i16) -> Result<(), Error> {
        Ok(())
    }
    fn serialize_i32(self, _: i32) -> Result<(), Error> {
        Ok(())
    }
    fn serialize_i64(self, _: i64) -> Result<(), Error> {
        Ok(())
    }
    fn serialize_u8(self, _: u8) -> Result<(), Error> {
        Ok(())
    }
    fn serialize_u16(self, _: u16) -> Result<(), Error> {
        Ok(())
    }
    fn serialize_u32(self, _: u32) -> Result<(), Error> {
        Ok(())
    }
    fn serialize_u64(self, _: u64) -> Result<(), Error> {
        Ok(())
    }
    fn serialize_f32(self, v: f32) -> Result<(), Error> {
        self.serialize_f64(v as f64)
    }
    fn serialize_f64(self, v: f64) -> Result<(), Error> {
        if v.is_finite() {
            Ok(())
        } else {
            Err(Error(format!("{v} is not a JSON number")))
        }
    }
    fn serialize_char(self, _: char) -> Result<(), Error> {
        Ok(())
    }
    fn serialize_str(self, _: &str) -> Result<(), Error> {
        Ok(())
    }
    fn serialize_bytes(self, _: &[u8]) -> Result<(), Error> {
        Ok(())
    }
    fn serialize_none(self) -> Result<(), Error> {
        Ok(())
    }
    fn serialize_some<T: Serialize + ?Sized>(self, v: &T) -> Result<(), Error> {
        v.serialize(Finite)
    }
    fn serialize_unit(self) -> Result<(), Error> {
        Ok(())
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<(), Error> {
        Ok(())
    }
    fn serialize_unit_variant(self, _: &'static str, _: u32, _: &'static str) -> Result<(), Error> {
        Ok(())
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        v: &T,
    ) -> Result<(), Error> {
        v.serialize(Finite)
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        v: &T,
    ) -> Result<(), Error> {
        v.serialize(Finite)
    }
    fn serialize_seq(self, _: Option<usize>) -> Result<Finite, Error> {
        Ok(Finite)
    }
    fn serialize_tuple(self, _: usize) -> Result<Finite, Error> {
        Ok(Finite)
    }
    fn serialize_tuple_struct(self, _: &'static str, _: usize) -> Result<Finite, Error> {
        Ok(Finite)
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Finite, Error> {
        Ok(Finite)
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Finite, Error> {
        Ok(Finite)
    }
    fn serialize_struct(self, _: &'static str, _: usize) -> Result<Finite, Error> {
        Ok(Finite)
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Finite, Error> {
        Ok(Finite)
    }
}

macro_rules! each {
    ($($t:ident :: $f:ident),*) => {$(
        impl ser::$t for Finite {
            type Ok = ();
            type Error = Error;
            fn $f<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), Error> {
                v.serialize(Finite)
            }
            fn end(self) -> Result<(), Error> {
                Ok(())
            }
        }
    )*};
}

each!(
    SerializeSeq::serialize_element,
    SerializeTuple::serialize_element,
    SerializeTupleStruct::serialize_field,
    SerializeTupleVariant::serialize_field
);

impl ser::SerializeMap for Finite {
    type Ok = ();
    type Error = Error;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, k: &T) -> Result<(), Error> {
        k.serialize(Finite)
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), Error> {
        v.serialize(Finite)
    }
    fn end(self) -> Result<(), Error> {
        Ok(())
    }
}

impl ser::SerializeStruct for Finite {
    type Ok = ();
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        _: &'static str,
        v: &T,
    ) -> Result<(), Error> {
        v.serialize(Finite)
    }
    fn end(self) -> Result<(), Error> {
        Ok(())
    }
}

impl ser::SerializeStructVariant for Finite {
    type Ok = ();
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        _: &'static str,
        v: &T,
    ) -> Result<(), Error> {
        v.serialize(Finite)
    }
    fn end(self) -> Result<(), Error> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn text(v: &Value) -> String {
        String::from_utf8(to_vec(v).unwrap()).unwrap()
    }

    /// RFC 8785 Appendix B: IEEE 754 bit patterns and their serializations.
    #[test]
    fn rfc8785_numbers() {
        let cases = [
            (0x0000000000000000u64, "0"),
            (0x8000000000000000, "0"),
            (0x0000000000000001, "5e-324"),
            (0x8000000000000001, "-5e-324"),
            (0x7fefffffffffffff, "1.7976931348623157e+308"),
            (0xffefffffffffffff, "-1.7976931348623157e+308"),
            (0x4340000000000000, "9007199254740992"),
            (0xc340000000000000, "-9007199254740992"),
            (0x4430000000000000, "295147905179352830000"),
            (0x44b52d02c7e14af5, "9.999999999999997e+22"),
            (0x44b52d02c7e14af6, "1e+23"),
            (0x44b52d02c7e14af7, "1.0000000000000001e+23"),
            (0x444b1ae4d6e2ef4e, "999999999999999700000"),
            (0x444b1ae4d6e2ef4f, "999999999999999900000"),
            (0x444b1ae4d6e2ef50, "1e+21"),
            (0x3eb0c6f7a0b5ed8c, "9.999999999999997e-7"),
            (0x3eb0c6f7a0b5ed8d, "0.000001"),
            (0x41b3de4355555553, "333333333.3333332"),
            (0x41b3de4355555554, "333333333.33333325"),
            (0x41b3de4355555555, "333333333.3333333"),
            (0x41b3de4355555556, "333333333.3333334"),
            (0x41b3de4355555557, "333333333.33333343"),
            (0xbecbf647612f3696, "-0.0000033333333333333333"),
            (0x43143ff3c1cb0959, "1424953923781206.2"),
        ];
        for (bits, want) in cases {
            let mut out = Vec::new();
            number(f64::from_bits(bits), &mut out).unwrap();
            assert_eq!(String::from_utf8(out).unwrap(), want, "{bits:016x}");
        }
        for bits in [0x7fffffffffffffffu64, 0x7ff0000000000000] {
            assert!(
                number(f64::from_bits(bits), &mut Vec::new()).is_err(),
                "{bits:016x}"
            );
        }
    }

    /// RFC 8785 §3.2.2: the example input and its canonical form.
    #[test]
    fn rfc8785_example() {
        let input = br#"{
            "numbers": [333333333.33333329, 1E30, 4.50, 2e-3, 0.000000000000000000000000001],
            "string": "\u20ac$\u000F\u000aA'\u0042\u0022\u005c\\\"\/",
            "literals": [null, true, false]
        }"#;
        assert_eq!(
            String::from_utf8(bytes(input).unwrap()).unwrap(),
            "{\"literals\":[null,true,false],\"numbers\":[333333333.3333333,1e+30,4.5,0.002,1e-27],\
             \"string\":\"€$\\u000f\\nA'B\\\"\\\\\\\\\\\"/\"}"
        );
    }

    /// RFC 8785 §3.2.3: names sort by UTF-16 code units, so U+1F600 (a surrogate pair) sorts
    /// before U+FB33.
    #[test]
    fn rfc8785_sorting() {
        let input = br#"{
            "\u20ac": "Euro Sign",
            "\r": "Carriage Return",
            "\ufb33": "Hebrew Letter Dalet With Dagesh",
            "1": "One",
            "\ud83d\ude00": "Emoji: Grinning Face",
            "\u0080": "Control",
            "\u00f6": "Latin Small Letter O With Diaeresis"
        }"#;
        assert_eq!(
            String::from_utf8(bytes(input).unwrap()).unwrap(),
            "{\"\\r\":\"Carriage Return\",\"1\":\"One\",\"\u{80}\":\"Control\",\
             \"ö\":\"Latin Small Letter O With Diaeresis\",\"€\":\"Euro Sign\",\
             \"😀\":\"Emoji: Grinning Face\",\"\u{fb33}\":\"Hebrew Letter Dalet With Dagesh\"}"
        );
    }

    #[test]
    fn integers_and_whitespace() {
        let v = json!({"b": 1, "a": [1.5, -0.0, 1e21, 2.0, 1e-7, true, null, "é\"x"], "A": {"z": 0, "y": -3}});
        assert_eq!(
            text(&v),
            r#"{"A":{"y":-3,"z":0},"a":[1.5,0,1e+21,2,1e-7,true,null,"é\"x"],"b":1}"#
        );
        assert_eq!(text(&json!(9007199254740993u64 - 1)), "9007199254740992");
        assert!(to_vec(&json!(9007199254740993u64)).is_err());
        assert!(to_vec(&json!(u64::MAX)).is_err());
        assert!(to_vec(&json!(i64::MIN)).is_ok());
    }

    #[test]
    fn refusals() {
        assert!(parse(br#"{"a":1,"a":2}"#).is_err());
        assert!(parse(br#"{"x":{"a":1,"a":1}}"#).is_err());
        assert!(parse(br#"{"a":1} x"#).is_err());
        #[derive(serde::Serialize)]
        struct M {
            v: f64,
        }
        assert!(to_vec(&M { v: f64::NAN }).is_err());
        assert!(to_vec(&vec![Some(f64::INFINITY)]).is_err());
        assert_eq!(to_vec(&M { v: 0.5 }).unwrap(), br#"{"v":0.5}"#);
    }

    #[test]
    fn hashes() {
        assert_eq!(
            sha256(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hash(&json!({"b": 2, "a": 1})).unwrap(),
            hash(&json!({"a": 1, "b": 2})).unwrap()
        );
        assert!(hash(&json!({})).unwrap().starts_with("sha256:"));
    }
}
