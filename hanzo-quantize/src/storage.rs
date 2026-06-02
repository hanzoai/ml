//! On-disk format for hanzo-quantize adapters.
//!
//! ```text
//! [0..8]   : b"HZQUANT0"                   (8-byte magic)
//! [8..12]  : u32 LE version                (currently 1)
//! [12..13] : u8 kind                       (1=BitDelta, 2=DeltaQuant)
//! [13..21] : u64 LE header_len
//! [21..29] : u64 LE payload_len
//! [29..29+header_len]      : bincode header
//! [...+payload_len]        : raw payload bytes (sign bits / packed quants)
//! ```
//!
//! The fixed-position header sizes mean the file is memmap-friendly: you can
//! mmap, read the 29-byte preamble, then slice the header and payload without
//! a copy. [`read_adapter_mmap`] uses this path.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use memmap2::MmapOptions;

use crate::{
    bitdelta::{BitDeltaAdapter, BitDeltaHeader},
    deltaquant::{DeltaQuantAdapter, DeltaQuantHeader},
    Error, Result,
};

pub const MAGIC: [u8; 8] = *b"HZQUANT0";
pub const VERSION: u32 = 1;

const PREAMBLE_LEN: usize = 8 + 4 + 1 + 8 + 8; // 29 bytes

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterKind {
    BitDelta,
    DeltaQuant,
}

impl AdapterKind {
    pub fn tag(self) -> u8 {
        match self {
            AdapterKind::BitDelta => 1,
            AdapterKind::DeltaQuant => 2,
        }
    }
    pub fn from_tag(b: u8) -> Result<Self> {
        match b {
            1 => Ok(AdapterKind::BitDelta),
            2 => Ok(AdapterKind::DeltaQuant),
            _ => Err(Error::InvalidKind(b)),
        }
    }
}

/// Either kind of adapter, read or written through [`read_adapter`] /
/// [`write_adapter`].
#[derive(Debug, Clone)]
pub enum Adapter {
    BitDelta(BitDeltaAdapter),
    DeltaQuant(DeltaQuantAdapter),
}

impl Adapter {
    pub fn kind(&self) -> AdapterKind {
        match self {
            Adapter::BitDelta(_) => AdapterKind::BitDelta,
            Adapter::DeltaQuant(_) => AdapterKind::DeltaQuant,
        }
    }
}

impl From<BitDeltaAdapter> for Adapter {
    fn from(a: BitDeltaAdapter) -> Self {
        Adapter::BitDelta(a)
    }
}
impl From<DeltaQuantAdapter> for Adapter {
    fn from(a: DeltaQuantAdapter) -> Self {
        Adapter::DeltaQuant(a)
    }
}

/// Write an adapter to disk in the `HZQUANT0` format.
pub fn write_adapter<P: AsRef<Path>>(path: P, adapter: &Adapter) -> Result<()> {
    let mut f = File::create(path)?;
    let bytes = encode(adapter)?;
    f.write_all(&bytes)?;
    Ok(())
}

/// Read an adapter from disk. Loads the whole file into memory (the usual
/// case for small adapters). For huge adapters use [`read_adapter_mmap`].
pub fn read_adapter<P: AsRef<Path>>(path: P) -> Result<Adapter> {
    let mut f = File::open(path)?;
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes)?;
    decode(&bytes)
}

/// Memmap a file and decode without copying the payload (the bincode header
/// still gets copied into a `Vec<u8>` for deserialization).
pub fn read_adapter_mmap<P: AsRef<Path>>(path: P) -> Result<Adapter> {
    let f = File::open(path)?;
    let m = unsafe { MmapOptions::new().map(&f)? };
    decode(&m[..])
}

/// Encode an adapter to a contiguous byte vector. Used by [`write_adapter`]
/// and by anyone shipping adapters over the wire.
pub fn encode(adapter: &Adapter) -> Result<Vec<u8>> {
    let (header_bytes, payload_bytes, kind) = match adapter {
        Adapter::BitDelta(b) => {
            let hb = bincode::serialize(&b.header)?;
            (hb, b.sign_bits.clone(), AdapterKind::BitDelta)
        }
        Adapter::DeltaQuant(d) => {
            let hb = bincode::serialize(&d.header)?;
            (hb, d.packed.clone(), AdapterKind::DeltaQuant)
        }
    };
    let mut out = Vec::with_capacity(PREAMBLE_LEN + header_bytes.len() + payload_bytes.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.push(kind.tag());
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&(payload_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&payload_bytes);
    Ok(out)
}

/// Decode bytes produced by [`encode`].
pub fn decode(bytes: &[u8]) -> Result<Adapter> {
    if bytes.len() < PREAMBLE_LEN {
        return Err(Error::Truncated { want: PREAMBLE_LEN, have: bytes.len() });
    }
    let magic: [u8; 8] = bytes[0..8].try_into().unwrap();
    if magic != MAGIC {
        return Err(Error::BadMagic { expected: MAGIC, got: magic });
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if version != VERSION {
        return Err(Error::UnknownVersion(version));
    }
    let kind = AdapterKind::from_tag(bytes[12])?;
    let header_len = u64::from_le_bytes(bytes[13..21].try_into().unwrap()) as usize;
    let payload_len = u64::from_le_bytes(bytes[21..29].try_into().unwrap()) as usize;
    let total = PREAMBLE_LEN + header_len + payload_len;
    if bytes.len() < total {
        return Err(Error::Truncated { want: total, have: bytes.len() });
    }
    let header_bytes = &bytes[PREAMBLE_LEN..PREAMBLE_LEN + header_len];
    let payload_bytes = &bytes[PREAMBLE_LEN + header_len..total];

    match kind {
        AdapterKind::BitDelta => {
            let header: BitDeltaHeader = bincode::deserialize(header_bytes)?;
            Ok(Adapter::BitDelta(BitDeltaAdapter {
                header,
                sign_bits: payload_bytes.to_vec(),
            }))
        }
        AdapterKind::DeltaQuant => {
            let header: DeltaQuantHeader = bincode::deserialize(header_bytes)?;
            Ok(Adapter::DeltaQuant(DeltaQuantAdapter {
                header,
                packed: payload_bytes.to_vec(),
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitdelta::BitDeltaAdapter;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn roundtrip_bitdelta_through_bytes() {
        let dev = Device::Cpu;
        let v: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.01).collect();
        let full = Tensor::from_vec(v, (8, 8), &dev).unwrap();
        let base = Tensor::zeros((8, 8), DType::F32, &dev).unwrap();
        let a = BitDeltaAdapter::compress_against_full(&full, &base).unwrap();
        let bytes = encode(&Adapter::BitDelta(a.clone())).unwrap();
        let dec = decode(&bytes).unwrap();
        let Adapter::BitDelta(b) = dec else { panic!("kind") };
        assert_eq!(b.sign_bits, a.sign_bits);
        assert_eq!(b.header.scales, a.header.scales);
        assert_eq!(b.header.shape, a.header.shape);
        assert_eq!(b.header.numel, a.header.numel);
    }

    #[test]
    fn bad_magic_errors() {
        let mut bytes = vec![0u8; PREAMBLE_LEN];
        bytes[0] = b'X';
        assert!(matches!(decode(&bytes), Err(Error::BadMagic { .. })));
    }

    #[test]
    fn truncated_errors() {
        let bytes = vec![0u8; 4];
        assert!(matches!(decode(&bytes), Err(Error::Truncated { .. })));
    }

    #[test]
    fn write_then_read_file() {
        let dev = Device::Cpu;
        let v: Vec<f32> = (0..32).map(|i| (i as f32) * 0.01).collect();
        let full = Tensor::from_vec(v, (4, 8), &dev).unwrap();
        let base = Tensor::zeros((4, 8), DType::F32, &dev).unwrap();
        let a = BitDeltaAdapter::compress_against_full(&full, &base).unwrap();

        let tmp = std::env::temp_dir().join("hzquantize_test.bin");
        write_adapter(&tmp, &Adapter::BitDelta(a.clone())).unwrap();
        let back = read_adapter(&tmp).unwrap();
        let Adapter::BitDelta(b) = back else { panic!() };
        assert_eq!(b.sign_bits, a.sign_bits);
        let _ = std::fs::remove_file(tmp);
    }
}
