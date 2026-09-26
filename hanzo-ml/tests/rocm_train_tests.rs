//! What backprop needs on ROCm beyond the forward pass: `cmp` (the masks behind the relu, abs,
//! maximum and minimum gradients) and `index_add` (the gradient of `index_select`), checked
//! against the CPU as the oracle.
//!
//! The CPU half runs everywhere and pins the fixture to hand-computed values, so a wrong oracle
//! cannot quietly agree with a wrong kernel. The ROCm half (feature `rocm`) runs the same cases on
//! the GPU for every dtype the kernels cover and demands bit-identical results: every value is a
//! small integer that each dtype here, bf16 included, represents exactly.

use hanzo_ml::{DType, Device, Result, Tensor, Var};

// Every dtype the ROCm cmp and index_add kernels cover.
const DTYPES: [DType; 7] = [
    DType::F32,
    DType::F64,
    DType::U8,
    DType::U32,
    DType::I64,
    DType::BF16,
    DType::F16,
];

fn flat(t: &Tensor) -> Result<Vec<f32>> {
    t.to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
}

/// Every case, as (name, flattened result).
fn cases(device: &Device, dtype: DType) -> Result<Vec<(&'static str, Vec<f32>)>> {
    let cpu = &Device::Cpu;
    let put = |t: Tensor| t.to_dtype(dtype)?.to_device(device);
    let a = put(Tensor::new(&[[0f32, 1., 2.], [3., 4., 5.]], cpu)?)?;
    let b = put(Tensor::new(&[[2f32, 1., 0.], [3., 5., 4.]], cpu)?)?;
    let row = put(Tensor::new(&[[2f32, 4., 1.]], cpu)?)?;
    let src = put(Tensor::arange(0f32, 9., cpu)?.reshape((3, 3))?)?;
    let base = put(Tensor::arange(0f32, 12., cpu)?.reshape((3, 4))?)?;
    let ones0 = Tensor::ones((4, 3), dtype, device)?;
    let ones1 = Tensor::ones((3, 4), dtype, device)?;
    let ids = Tensor::new(&[3u32, 0, 3], cpu)?;
    let ids0 = ids.to_device(device)?;
    let ids8 = ids.to_dtype(DType::U8)?.to_device(device)?;
    let ids64 = ids.to_dtype(DType::I64)?.to_device(device)?;
    let ids1 = Tensor::new(&[1u32, 1, 3], cpu)?.to_device(device)?;
    Ok(vec![
        ("eq", flat(&a.eq(&b)?)?),
        ("ne", flat(&a.ne(&b)?)?),
        ("lt", flat(&a.lt(&b)?)?),
        ("le", flat(&a.le(&b)?)?),
        ("gt", flat(&a.gt(&b)?)?),
        ("ge", flat(&a.ge(&b)?)?),
        ("lt_strided", flat(&a.t()?.lt(&b.t()?)?)?),
        ("ge_broadcast", flat(&a.broadcast_ge(&row)?)?),
        ("index_add_dim0", flat(&ones0.index_add(&ids0, &src, 0)?)?),
        ("index_add_u8", flat(&ones0.index_add(&ids8, &src, 0)?)?),
        ("index_add_i64", flat(&ones0.index_add(&ids64, &src, 0)?)?),
        ("index_add_dim1", flat(&ones1.index_add(&ids1, &src, 1)?)?),
        ("index_add_t", flat(&base.t()?.index_add(&ids0, &src, 0)?)?),
    ])
}

fn case<'a>(cases: &'a [(&'static str, Vec<f32>)], name: &str) -> &'a [f32] {
    &cases.iter().find(|(n, _)| *n == name).unwrap().1
}

// Row 0 takes source row 1; row 3 takes source rows 0 and 2.
#[rustfmt::skip]
const INDEX_ADD_DIM0: [f32; 12] = [
    4.0, 5.0, 6.0,
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    7.0, 9.0, 11.0,
];

#[rustfmt::skip]
const INDEX_ADD_DIM1: [f32; 12] = [
    1.0, 2.0, 1.0, 3.0,
    1.0, 8.0, 1.0, 6.0,
    1.0, 14.0, 1.0, 9.0,
];

#[rustfmt::skip]
const INDEX_ADD_T: [f32; 12] = [
    3.0, 8.0, 13.0,
    1.0, 5.0, 9.0,
    2.0, 6.0, 10.0,
    9.0, 15.0, 21.0,
];

#[test]
fn cpu_oracle_is_pinned() -> Result<()> {
    for dtype in DTYPES {
        let c = cases(&Device::Cpu, dtype)?;
        let pin = |name: &str, want: &[f32]| assert_eq!(case(&c, name), want, "{dtype:?} {name}");
        pin("eq", &[0., 1., 0., 1., 0., 0.]);
        pin("ne", &[1., 0., 1., 0., 1., 1.]);
        pin("lt", &[1., 0., 0., 0., 1., 0.]);
        pin("le", &[1., 1., 0., 1., 1., 0.]);
        pin("gt", &[0., 0., 1., 0., 0., 1.]);
        pin("ge", &[0., 1., 1., 1., 0., 1.]);
        pin("lt_strided", &[1., 0., 0., 1., 0., 0.]);
        pin("ge_broadcast", &[0., 0., 1., 1., 1., 1.]);
        pin("index_add_dim0", &INDEX_ADD_DIM0);
        pin("index_add_u8", &INDEX_ADD_DIM0);
        pin("index_add_i64", &INDEX_ADD_DIM0);
        pin("index_add_dim1", &INDEX_ADD_DIM1);
        pin("index_add_t", &INDEX_ADD_T);
    }
    Ok(())
}

/// The gradient of `(relu(w[ids]) * c).sum()` with respect to `w`: relu's backward is a `ge` mask
/// and `index_select`'s is an `index_add`.
fn grad(device: &Device, dtype: DType) -> Result<Vec<f32>> {
    let cpu = &Device::Cpu;
    let put = |t: Tensor| t.to_dtype(dtype)?.to_device(device);
    let w = Var::from_tensor(&put(Tensor::arange(-6f32, 6., cpu)?.reshape((4, 3))?)?)?;
    let c = put(Tensor::arange(1f32, 10., cpu)?.reshape((3, 3))?)?;
    let ids = Tensor::new(&[3u32, 0, 3], cpu)?.to_device(device)?;
    let loss = (w.index_select(&ids, 0)?.relu()? * c)?.sum_all()?;
    flat(loss.backward()?.get(&w).unwrap())
}

#[test]
fn cpu_grad_is_pinned() -> Result<()> {
    for dtype in [DType::F32, DType::BF16] {
        let want = [0., 0., 0., 0., 0., 0., 0., 0., 0., 8., 10., 12.];
        assert_eq!(grad(&Device::Cpu, dtype)?, want, "{dtype:?}");
    }
    Ok(())
}

#[cfg(feature = "rocm")]
#[test]
fn rocm_matches_cpu() -> Result<()> {
    let rocm = Device::new_rocm(0)?;
    for dtype in DTYPES {
        let want = cases(&Device::Cpu, dtype)?;
        let got = cases(&rocm, dtype)?;
        for ((name, want), (_, got)) in want.iter().zip(got.iter()) {
            assert_eq!(got, want, "{dtype:?} {name}");
        }
    }
    for dtype in [DType::F32, DType::BF16] {
        let want = grad(&Device::Cpu, dtype)?;
        assert_eq!(grad(&rocm, dtype)?, want, "{dtype:?} grad");
    }
    Ok(())
}
