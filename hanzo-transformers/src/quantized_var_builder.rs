//! Varbuilder for Loading gguf files
//!
//! VarBuilder is a utility to store quantized tensors from a [GGUF model file](https://huggingface.co/docs/hub/gguf).
//! These tensors can be loaded from disk using `from_gguf` or from an in-memory
//! buffer using `from_gguf_buffer`.

use hanzo_ml::quantized::QTensor;
use hanzo_ml::{Device, Result, Shape};
use std::sync::Arc;

// VarBuilder specialized for QTensors
#[derive(Clone)]
pub struct VarBuilder {
    data: Arc<std::collections::HashMap<String, Arc<QTensor>>>,
    path: Vec<String>,
    device: Device,
}

impl VarBuilder {
    pub fn from_gguf<P: AsRef<std::path::Path>>(p: P, device: &Device) -> Result<Self> {
        // Opt-in mmap weight streaming: reference weights in the page cache instead of
        // materializing them resident, so a model larger than RAM runs (`HANZO_GGUF_MMAP=1`;
        // `=0`/empty keeps the default resident path). See `VarBuilder::from_gguf_mmap`.
        if std::env::var("HANZO_GGUF_MMAP")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false)
        {
            return Self::from_gguf_mmap(p, device);
        }
        let mut file = std::fs::File::open(p)?;
        let content = hanzo_ml::quantized::gguf_file::Content::read(&mut file)?;
        let mut data = std::collections::HashMap::new();
        for tensor_name in content.tensor_infos.keys() {
            let tensor = content.tensor(&mut file, tensor_name, device)?;
            data.insert(tensor_name.to_string(), Arc::new(tensor));
        }
        Ok(Self {
            data: Arc::new(data),
            path: Vec::new(),
            device: device.clone(),
        })
    }

    /// mmap-backed twin of [`VarBuilder::from_gguf`]: the GGUF stays memory-mapped and each weight
    /// is referenced in place. On the CPU the quantized blocks are never copied resident -- the OS
    /// demand-pages them from the file and reclaims them under pressure -- so a model whose weights
    /// exceed free RAM loads and runs (antirez ds4_ssd, "RAM as a speed spectrum"). GPU backends
    /// stage each weight into their own device buffer directly from the mapping (no resident Vec).
    /// Explicit constructor for the `HANZO_GGUF_MMAP=1` toggle.
    pub fn from_gguf_mmap<P: AsRef<std::path::Path>>(p: P, device: &Device) -> Result<Self> {
        let (content, mmap) = hanzo_ml::quantized::gguf_file::Content::read_mmap(p)?;
        let mut data = std::collections::HashMap::new();
        for tensor_name in content.tensor_infos.keys() {
            let tensor = content.tensor_mmap(&mmap, tensor_name, device)?;
            data.insert(tensor_name.to_string(), Arc::new(tensor));
        }
        Ok(Self {
            data: Arc::new(data),
            path: Vec::new(),
            device: device.clone(),
        })
    }

    pub fn from_gguf_buffer(buffer: &[u8], device: &Device) -> Result<Self> {
        let mut cursor = std::io::Cursor::new(buffer);
        let content = hanzo_ml::quantized::gguf_file::Content::read(&mut cursor)?;
        let mut data = std::collections::HashMap::new();
        for tensor_name in content.tensor_infos.keys() {
            let tensor = content.tensor(&mut cursor, tensor_name, device)?;
            data.insert(tensor_name.to_string(), Arc::new(tensor));
        }
        Ok(Self {
            data: Arc::new(data),
            path: Vec::new(),
            device: device.clone(),
        })
    }

    pub fn pp<S: ToString>(&self, s: S) -> Self {
        let mut path = self.path.clone();
        path.push(s.to_string());
        Self {
            data: self.data.clone(),
            path,
            device: self.device.clone(),
        }
    }

    fn path(&self, tensor_name: &str) -> String {
        if self.path.is_empty() {
            tensor_name.to_string()
        } else {
            [&self.path.join("."), tensor_name].join(".")
        }
    }

    pub fn get<S: Into<Shape>>(&self, s: S, name: &str) -> Result<Arc<QTensor>> {
        let path = self.path(name);
        match self.data.get(&path) {
            None => {
                hanzo_ml::bail!("cannot find tensor {path}")
            }
            Some(qtensor) => {
                let shape = s.into();
                if qtensor.shape() != &shape {
                    hanzo_ml::bail!(
                        "shape mismatch for {name}, got {:?}, expected {shape:?}",
                        qtensor.shape()
                    )
                }
                Ok(qtensor.clone())
            }
        }
    }

    pub fn get_no_shape(&self, name: &str) -> Result<Arc<QTensor>> {
        let path = self.path(name);
        match self.data.get(&path) {
            None => {
                hanzo_ml::bail!("cannot find tensor {name}")
            }
            Some(qtensor) => Ok(qtensor.clone()),
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }
}
