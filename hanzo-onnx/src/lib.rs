use hanzo_ml::Result;
use prost::Message;

pub mod onnx {
    // prost-build writes this module from onnx.proto3; the doc comments are the
    // upstream proto's, reflowed by the generator. The lint is about source we do
    // not author and cannot edit, so it is allowed here and nowhere wider.
    #![allow(clippy::doc_overindented_list_items)]
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub mod eval;
mod ml;
pub mod value;
pub use eval::{dtype, simple_eval, Domain};
pub use value::{Key, Labels, Table, Text, Value};

pub fn read_file<P: AsRef<std::path::Path>>(p: P) -> Result<onnx::ModelProto> {
    let buf = std::fs::read(p)?;
    onnx::ModelProto::decode(buf.as_slice()).map_err(hanzo_ml::Error::wrap)
}
