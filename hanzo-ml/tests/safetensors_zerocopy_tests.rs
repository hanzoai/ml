use hanzo_ml::safetensors;
use hanzo_ml::{DType, Device, Result, Tensor};
use std::collections::HashMap;

struct TmpFile(std::path::PathBuf);

impl TmpFile {
    fn create(base: &str) -> TmpFile {
        let filename = std::env::temp_dir().join(format!(
            "hanzo-test-{}-{}-{:?}.safetensors",
            base,
            std::process::id(),
            std::thread::current().id(),
        ));
        TmpFile(filename)
    }
}

impl AsRef<std::path::Path> for TmpFile {
    fn as_ref(&self) -> &std::path::Path {
        self.0.as_path()
    }
}

impl Drop for TmpFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn test_safetensors_view_is_borrowed() -> Result<()> {
    let dev = Device::Cpu;

    // Test multiple dtypes: F32, F16, BF16, I32, I64, U8, U32
    let t_f32 = Tensor::arange(0.0f32, 100.0f32, &dev)?;
    let view_f32 = t_f32.to_safetensors_view()?;
    assert!(
        view_f32.is_borrowed(),
        "Contiguous CPU F32 tensor must be borrowed without allocation"
    );
    assert_eq!(view_f32.data_len(), 100 * 4);

    let t_u8 = Tensor::from_slice(&[1u8, 2, 3, 4, 5], 5, &dev)?;
    let view_u8 = t_u8.to_safetensors_view()?;
    assert!(
        view_u8.is_borrowed(),
        "Contiguous CPU U8 tensor must be borrowed without allocation"
    );
    assert_eq!(view_u8.data_len(), 5);

    let t_i32 = Tensor::from_slice(&[10i32, 20, 30, 40], (2, 2), &dev)?;
    let view_i32 = t_i32.to_safetensors_view()?;
    assert!(
        view_i32.is_borrowed(),
        "Contiguous CPU I32 tensor must be borrowed without allocation"
    );
    assert_eq!(view_i32.data_len(), 4 * 4);

    let t_i64 = Tensor::from_slice(&[100i64, 200, 300], 3, &dev)?;
    let view_i64 = t_i64.to_safetensors_view()?;
    assert!(
        view_i64.is_borrowed(),
        "Contiguous CPU I64 tensor must be borrowed without allocation"
    );
    assert_eq!(view_i64.data_len(), 3 * 8);

    let t_f16 = Tensor::from_slice(
        &[
            half::f16::from_f32(1.0),
            half::f16::from_f32(2.0),
            half::f16::from_f32(3.0),
        ],
        3,
        &dev,
    )?;
    let view_f16 = t_f16.to_safetensors_view()?;
    assert!(
        view_f16.is_borrowed(),
        "Contiguous CPU F16 tensor must be borrowed without allocation"
    );
    assert_eq!(view_f16.data_len(), 3 * 2);

    let t_bf16 = Tensor::from_slice(
        &[
            half::bf16::from_f32(4.0),
            half::bf16::from_f32(5.0),
            half::bf16::from_f32(6.0),
        ],
        3,
        &dev,
    )?;
    let view_bf16 = t_bf16.to_safetensors_view()?;
    assert!(
        view_bf16.is_borrowed(),
        "Contiguous CPU BF16 tensor must be borrowed without allocation"
    );
    assert_eq!(view_bf16.data_len(), 3 * 2);

    Ok(())
}

#[test]
fn test_safetensors_view_fallback_for_non_contiguous() -> Result<()> {
    let dev = Device::Cpu;
    // 2x3 matrix
    let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &dev)?;
    assert!(t.is_contiguous());

    // Transpose is non-contiguous
    let t_t = t.t()?;
    assert!(!t_t.is_contiguous());

    let view = t_t.to_safetensors_view()?;
    assert!(
        !view.is_borrowed(),
        "Non-contiguous tensor must use fallback contiguous copy"
    );
    assert_eq!(view.shape(), &[3, 2]);

    // Save and verify data is correctly laid out
    let tmp = TmpFile::create("transposed");
    t_t.save_safetensors("t_t", &tmp)?;

    let loaded = safetensors::load(&tmp, &dev)?;
    let loaded_t = loaded.get("t_t").unwrap();
    assert_eq!(loaded_t.dims(), &[3, 2]);
    assert_eq!(
        loaded_t.to_vec2::<f32>()?,
        vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]
    );

    Ok(())
}

#[test]
fn test_safetensors_save_roundtrip_all_dtypes() -> Result<()> {
    let dev = Device::Cpu;
    let mut map = HashMap::new();

    let u8_data = vec![10u8, 20, 30, 40, 50];
    map.insert("u8", Tensor::from_slice(&u8_data, 5, &dev)?);

    let u32_data = vec![1000u32, 2000, 3000];
    map.insert("u32", Tensor::from_slice(&u32_data, 3, &dev)?);

    let i16_data = vec![-10i16, 20, -30, 40];
    map.insert("i16", Tensor::from_slice(&i16_data, (2, 2), &dev)?);

    let i32_data = vec![-100i32, 200, -300];
    map.insert("i32", Tensor::from_slice(&i32_data, 3, &dev)?);

    let i64_data = vec![-1000i64, 2000, -3000, 4000];
    map.insert("i64", Tensor::from_slice(&i64_data, (2, 2), &dev)?);

    let f32_data = vec![1.5f32, -2.5, 3.25, -4.125];
    map.insert("f32", Tensor::from_slice(&f32_data, (2, 2), &dev)?);

    let f64_data = vec![1.123456789f64, -2.987654321];
    map.insert("f64", Tensor::from_slice(&f64_data, 2, &dev)?);

    let f16_data = vec![
        half::f16::from_f32(1.5),
        half::f16::from_f32(-2.5),
        half::f16::from_f32(3.0),
    ];
    map.insert("f16", Tensor::from_slice(&f16_data, 3, &dev)?);

    let bf16_data = vec![
        half::bf16::from_f32(1.5),
        half::bf16::from_f32(-2.5),
        half::bf16::from_f32(3.0),
    ];
    map.insert("bf16", Tensor::from_slice(&bf16_data, 3, &dev)?);

    let tmp = TmpFile::create("all_dtypes");
    safetensors::save(&map, &tmp)?;

    let loaded = safetensors::load(&tmp, &dev)?;
    assert_eq!(loaded.len(), map.len());

    assert_eq!(loaded.get("u8").unwrap().to_vec1::<u8>()?, u8_data);
    assert_eq!(loaded.get("u32").unwrap().to_vec1::<u32>()?, u32_data);
    assert_eq!(
        loaded.get("i16").unwrap().to_vec2::<i16>()?,
        vec![vec![-10, 20], vec![-30, 40]]
    );
    assert_eq!(loaded.get("i32").unwrap().to_vec1::<i32>()?, i32_data);
    assert_eq!(
        loaded.get("i64").unwrap().to_vec2::<i64>()?,
        vec![vec![-1000, 2000], vec![-3000, 4000]]
    );
    assert_eq!(
        loaded.get("f32").unwrap().to_vec2::<f32>()?,
        vec![vec![1.5, -2.5], vec![3.25, -4.125]]
    );
    assert_eq!(loaded.get("f64").unwrap().to_vec1::<f64>()?, f64_data);
    assert_eq!(loaded.get("f16").unwrap().to_vec1::<half::f16>()?, f16_data);
    assert_eq!(
        loaded.get("bf16").unwrap().to_vec1::<half::bf16>()?,
        bf16_data
    );

    Ok(())
}

#[test]
fn test_safetensors_serialize_in_memory() -> Result<()> {
    let dev = Device::Cpu;
    let mut map = HashMap::new();
    map.insert("weight", Tensor::arange(0.0f32, 64.0f32, &dev)?);
    map.insert("bias", Tensor::zeros(64, DType::F32, &dev)?);

    let bytes = safetensors::serialize(&map)?;
    assert!(!bytes.is_empty());

    let loaded = safetensors::load_buffer(&bytes, &dev)?;
    assert_eq!(loaded.len(), 2);
    assert_eq!(
        loaded.get("weight").unwrap().to_vec1::<f32>()?,
        (0..64).map(|i| i as f32).collect::<Vec<_>>()
    );
    assert_eq!(
        loaded.get("bias").unwrap().to_vec1::<f32>()?,
        vec![0.0f32; 64]
    );

    Ok(())
}

#[test]
fn test_safetensors_sliced_contiguous_tensor() -> Result<()> {
    let dev = Device::Cpu;
    let full = Tensor::arange(0.0f32, 100.0f32, &dev)?;

    // Slicing 10 elements starting from index 20
    let sliced = full.narrow(0, 20, 10)?;
    assert!(sliced.is_contiguous());

    let view = sliced.to_safetensors_view()?;
    assert!(
        view.is_borrowed(),
        "Contiguous sub-slice must be zero-copy borrowed"
    );
    assert_eq!(view.data_len(), 10 * 4);

    let tmp = TmpFile::create("slice");
    sliced.save_safetensors("sliced", &tmp)?;

    let loaded = safetensors::load(&tmp, &dev)?;
    let loaded_slice = loaded.get("sliced").unwrap();
    assert_eq!(
        loaded_slice.to_vec1::<f32>()?,
        (20..30).map(|i| i as f32).collect::<Vec<_>>()
    );

    Ok(())
}

#[test]
fn test_safetensors_empty_tensor() -> Result<()> {
    let dev = Device::Cpu;
    let empty = Tensor::zeros(0, DType::F32, &dev)?;
    let view = empty.to_safetensors_view()?;
    assert_eq!(view.data_len(), 0);

    let tmp = TmpFile::create("empty");
    empty.save_safetensors("empty", &tmp)?;

    let loaded = safetensors::load(&tmp, &dev)?;
    let loaded_empty = loaded.get("empty").unwrap();
    assert_eq!(loaded_empty.elem_count(), 0);
    assert_eq!(loaded_empty.to_vec1::<f32>()?, Vec::<f32>::new());

    Ok(())
}

#[test]
fn test_safetensors_packed_sub_byte_dtypes() -> Result<()> {
    use ::safetensors::tensor as st;

    struct RawView<'a> {
        dtype: st::Dtype,
        shape: &'a [usize],
        data: &'a [u8],
    }

    impl<'a> st::View for RawView<'a> {
        fn dtype(&self) -> st::Dtype {
            self.dtype
        }
        fn shape(&self) -> &[usize] {
            self.shape
        }
        fn data(&self) -> std::borrow::Cow<'_, [u8]> {
            std::borrow::Cow::Borrowed(self.data)
        }
        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    let dev = Device::Cpu;

    // 1. Test F4 (MX4) packed format
    let f4_bytes = [0x12u8, 0x34, 0x56, 0x78];
    let f4_shape = [8];
    let f4_view = RawView {
        dtype: st::Dtype::F4,
        shape: &f4_shape,
        data: &f4_bytes,
    };
    let initial_buf = st::serialize([("f4_test", f4_view)], None).map_err(hanzo_ml::Error::from)?;

    let loaded = safetensors::load_buffer(&initial_buf, &dev)?;
    let f4_tensor = loaded.get("f4_test").expect("f4_test loaded");
    assert_eq!(f4_tensor.dtype(), DType::F4);

    let view = f4_tensor.to_safetensors_view()?;
    assert!(
        view.is_borrowed(),
        "Contiguous CPU F4 tensor view must be borrowed zero-copy"
    );
    assert_eq!(
        view.data_len(),
        4,
        "F4 view data_len must match packed byte count (4 bytes), not 0"
    );
    assert_eq!(view.data(), &f4_bytes);

    // Save and re-load roundtrip
    let tmp_f4 = TmpFile::create("f4");
    f4_tensor.save_safetensors("f4_test", &tmp_f4)?;
    let reloaded = safetensors::load(&tmp_f4, &dev)?;
    let reloaded_f4 = reloaded.get("f4_test").expect("f4_test reloaded");
    let reloaded_view = reloaded_f4.to_safetensors_view()?;
    assert_eq!(reloaded_view.data(), &f4_bytes);

    // Narrowed F4 slice test: byte-aligned offset
    let f4_aligned = f4_tensor.narrow(0, 0, 4)?;
    let view_aligned = f4_aligned.to_safetensors_view()?;
    assert!(
        view_aligned.is_borrowed(),
        "Aligned narrowed F4 tensor must be borrowed zero-copy"
    );
    assert_eq!(view_aligned.data_len(), 2);
    assert_eq!(view_aligned.data(), &f4_bytes[0..2]);

    let tmp_aligned = TmpFile::create("f4_aligned");
    f4_aligned.save_safetensors("f4_aligned", &tmp_aligned)?;
    let reloaded_aligned = safetensors::load(&tmp_aligned, &dev)?;
    let reloaded_al_t = reloaded_aligned
        .get("f4_aligned")
        .expect("f4_aligned loaded");
    assert_eq!(reloaded_al_t.to_safetensors_view()?.data(), &f4_bytes[0..2]);

    // Narrowed F4 slice test: unaligned offset (repacked)
    let f4_unaligned = f4_tensor.narrow(0, 1, 4)?;
    let view_unaligned = f4_unaligned.to_safetensors_view()?;
    assert!(
        !view_unaligned.is_borrowed(),
        "Unaligned narrowed F4 tensor must be repacked via fallback"
    );
    assert_eq!(view_unaligned.data_len(), 2);

    let tmp_unaligned = TmpFile::create("f4_unaligned");
    f4_unaligned.save_safetensors("f4_unaligned", &tmp_unaligned)?;
    let reloaded_unaligned = safetensors::load(&tmp_unaligned, &dev)?;
    let reloaded_un_t = reloaded_unaligned
        .get("f4_unaligned")
        .expect("f4_unaligned loaded");
    assert_eq!(reloaded_un_t.to_safetensors_view()?.data_len(), 2);
    assert_eq!(
        reloaded_un_t.to_safetensors_view()?.data(),
        view_unaligned.data()
    );

    // 2. Test F6_E2M3 (MX6) format
    let f6_bytes = [0xABu8, 0xCD, 0xEF];
    let f6_shape = [4];
    let f6_view = RawView {
        dtype: st::Dtype::F6_E2M3,
        shape: &f6_shape,
        data: &f6_bytes,
    };
    let f6_buf = st::serialize([("f6_test", f6_view)], None).map_err(hanzo_ml::Error::from)?;
    let loaded_f6 = safetensors::load_buffer(&f6_buf, &dev)?;
    let f6_tensor = loaded_f6.get("f6_test").expect("f6_test loaded");
    assert_eq!(f6_tensor.dtype(), DType::F6E2M3);

    let view_f6 = f6_tensor.to_safetensors_view()?;
    assert!(view_f6.is_borrowed());
    assert_eq!(view_f6.data_len(), 3);
    assert_eq!(view_f6.data(), &f6_bytes);

    let serialized = safetensors::serialize(&loaded_f6)?;
    let reloaded_f6_map = safetensors::load_buffer(&serialized, &dev)?;
    let reloaded_f6 = reloaded_f6_map.get("f6_test").expect("reloaded f6");
    assert_eq!(reloaded_f6.to_safetensors_view()?.data(), &f6_bytes);

    Ok(())
}

#[test]
fn test_safetensors_duplicate_keys_rejected() -> Result<()> {
    #[derive(PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
    struct CustomKey(String, usize);

    impl AsRef<str> for CustomKey {
        fn as_ref(&self) -> &str {
            &self.0
        }
    }

    impl std::fmt::Display for CustomKey {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}_{}", self.0, self.1)
        }
    }

    let dev = Device::Cpu;
    let t1 = Tensor::arange(0.0f32, 10.0f32, &dev)?;
    let t2 = Tensor::arange(10.0f32, 20.0f32, &dev)?;

    let mut map = HashMap::new();
    map.insert(CustomKey("weight".into(), 1), t1);
    map.insert(CustomKey("weight".into(), 2), t2);

    let res = safetensors::serialize(&map);
    assert!(
        res.is_err(),
        "Serializing map with duplicate string keys must return an error"
    );

    let tmp = TmpFile::create("dup");
    let res_save = safetensors::save(&map, &tmp);
    assert!(
        res_save.is_err(),
        "Saving map with duplicate string keys must return an error"
    );

    Ok(())
}
