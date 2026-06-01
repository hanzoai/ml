//! Native-candle quantization for fine-tune deltas.
//!
//! This crate is the runtime path for federation: workers push BitDelta-
//! compressed adapters (~32× smaller than f32 weights), the coordinator
//! aggregates them with [`soup`], and applies them on top of an existing
//! GGUF-quantized base via [`bitdelta::BitDeltaAdapter::apply_to`].
//!
//! Three layers:
//!
//! - [`bitdelta`]    — 1-bit deltas with **per-output-channel** scale,
//!   MSB-first sign packing (matches GGUF convention), candle-native
//!   `QTensor` in/out.
//! - [`deltaquant`]  — INT2/INT4/INT8 grouped symmetric quant aware of
//!   candle's existing scale convention.
//! - [`soup`]        — coordinate-wise aggregation (mean / trimmed-mean /
//!   krum) of multiple compressed adapters.
//!
//! The on-disk format ([`storage`]) is memmap-friendly: `b"HZQUANT0"` (8
//! bytes) + LE u32 version + bincode header + raw payload bytes.
//!
//! The [`ggml_bridge`] turns an existing GGUF base + a full-precision tensor
//! into `(base_qtensor, BitDeltaAdapter)` so an existing GGUF model can be
//! "split" into a base ship + a delta adapter.

pub mod bitdelta;
pub mod deltaquant;
pub mod ggml_bridge;
pub mod soup;
pub mod storage;

pub use bitdelta::{BitDeltaAdapter, BitDeltaHeader};
pub use deltaquant::{DeltaQuantAdapter, DeltaQuantHeader, QuantBits};
pub use soup::{aggregate, AggregateMethod};
pub use storage::{read_adapter, write_adapter, AdapterKind, MAGIC, VERSION};

/// Crate-level error type. Wraps candle errors and storage errors uniformly.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("candle: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("bincode: {0}")]
    Bincode(#[from] bincode::Error),
    #[error("shape mismatch: full={full:?} base={base:?}")]
    ShapeMismatch { full: Vec<usize>, base: Vec<usize> },
    #[error("empty tensor: {0}")]
    Empty(&'static str),
    #[error("bad magic: expected {expected:x?} got {got:x?}")]
    BadMagic { expected: [u8; 8], got: [u8; 8] },
    #[error("unknown version: {0}")]
    UnknownVersion(u32),
    #[error("aggregation: need >= {needed}, got {got}")]
    NotEnough { needed: usize, got: usize },
    #[error("payload truncated: want {want} bytes, have {have}")]
    Truncated { want: usize, have: usize },
    #[error("invalid bits: {0}")]
    InvalidBits(u8),
    #[error("invalid kind tag: {0}")]
    InvalidKind(u8),
}

pub type Result<T> = std::result::Result<T, Error>;
