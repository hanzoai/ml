//! `scatter` / `scatter_set` / `scatter_add` / `scatter_add_set` on the ROCm backend, checked
//! against the CPU as the oracle.
//!
//! The CPU half runs everywhere and pins the fixture to hand-computed values, so a wrong oracle
//! cannot quietly agree with a wrong kernel. The ROCm half (feature `rocm`) runs the same cases
//! on the GPU for every dtype the ROCm scatter kernels cover and demands bit-identical results:
//! the integer and set paths are exact by construction, and the float cases use small integers
//! that every dtype here, bf16 included, represents exactly.

use hanzo_ml::{DType, Device, Result, Tensor};

// Every dtype the ROCm scatter kernels cover.
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

/// Every case, as (name, flattened result). `src` and `ids` are [4,3]; the destination is [6,3]
/// for dim 0 and [4,5] for dim 1, so the scattered dim differs in length from the source.
fn cases(device: &Device, dtype: DType) -> Result<Vec<(&'static str, Vec<f32>)>> {
    let src = Tensor::arange(0f32, 12f32, &Device::Cpu)?
        .reshape((4, 3))?
        .to_dtype(dtype)?
        .to_device(device)?;
    let ids = Tensor::new(
        &[[0u32, 1, 2], [3, 4, 0], [3, 3, 1], [2, 0, 4]],
        &Device::Cpu,
    )?
    .to_device(device)?;
    let ids_i64 = ids
        .to_device(&Device::Cpu)?
        .to_dtype(DType::I64)?
        .to_device(device)?;
    let ids_skip = Tensor::new(
        &[
            [0u32, u32::MAX, 2],
            [3, 4, u32::MAX],
            [3, 3, 1],
            [u32::MAX, u32::MAX, 4],
        ],
        &Device::Cpu,
    )?
    .to_device(device)?;
    let dst0 = Tensor::ones((6, 3), dtype, &Device::Cpu)?.to_device(device)?;
    let dst1 = Tensor::ones((4, 5), dtype, &Device::Cpu)?.to_device(device)?;

    let set = Tensor::ones((6, 3), dtype, &Device::Cpu)?.to_device(device)?;
    set.scatter_set(&ids, &src, 0)?;
    let add_set = Tensor::ones((6, 3), dtype, &Device::Cpu)?.to_device(device)?;
    add_set.scatter_add_set(&ids, &src, 0)?;

    Ok(vec![
        ("scatter_dim0", flat(&dst0.scatter(&ids, &src, 0)?)?),
        ("scatter_add_dim0", flat(&dst0.scatter_add(&ids, &src, 0)?)?),
        ("scatter_dim1", flat(&dst1.scatter(&ids, &src, 1)?)?),
        ("scatter_add_dim1", flat(&dst1.scatter_add(&ids, &src, 1)?)?),
        ("scatter_ids_i64", flat(&dst0.scatter(&ids_i64, &src, 0)?)?),
        (
            "scatter_add_ids_i64",
            flat(&dst0.scatter_add(&ids_i64, &src, 0)?)?,
        ),
        ("scatter_skip", flat(&dst0.scatter(&ids_skip, &src, 0)?)?),
        (
            "scatter_add_skip",
            flat(&dst0.scatter_add(&ids_skip, &src, 0)?)?,
        ),
        ("scatter_set", flat(&set)?),
        ("scatter_add_set", flat(&add_set)?),
    ])
}

fn case<'a>(cases: &'a [(&'static str, Vec<f32>)], name: &str) -> &'a [f32] {
    &cases.iter().find(|(n, _)| *n == name).unwrap().1
}

#[rustfmt::skip]
const SCATTER_DIM0: [f32; 18] = [
    0.0, 10.0, 5.0,
    1.0, 1.0, 8.0,
    9.0, 1.0, 2.0,
    6.0, 7.0, 1.0,
    1.0, 4.0, 11.0,
    1.0, 1.0, 1.0,
];

#[rustfmt::skip]
const SCATTER_ADD_DIM0: [f32; 18] = [
    1.0, 11.0, 6.0,
    1.0, 2.0, 9.0,
    10.0, 1.0, 3.0,
    10.0, 8.0, 1.0,
    1.0, 5.0, 12.0,
    1.0, 1.0, 1.0,
];

// u32::MAX in the ids means "leave that slot alone".
#[rustfmt::skip]
const SCATTER_SKIP: [f32; 18] = [
    0.0, 1.0, 1.0,
    1.0, 1.0, 8.0,
    1.0, 1.0, 2.0,
    6.0, 7.0, 1.0,
    1.0, 4.0, 11.0,
    1.0, 1.0, 1.0,
];

#[rustfmt::skip]
const SCATTER_DIM1: [f32; 20] = [
    0.0, 1.0, 2.0, 1.0, 1.0,
    5.0, 1.0, 1.0, 3.0, 4.0,
    1.0, 8.0, 1.0, 7.0, 1.0,
    10.0, 1.0, 9.0, 1.0, 11.0,
];

#[rustfmt::skip]
const SCATTER_ADD_DIM1: [f32; 20] = [
    1.0, 2.0, 3.0, 1.0, 1.0,
    6.0, 1.0, 1.0, 4.0, 5.0,
    1.0, 9.0, 1.0, 14.0, 1.0,
    11.0, 1.0, 10.0, 1.0, 12.0,
];

#[rustfmt::skip]
const SCATTER_ADD_SKIP: [f32; 18] = [
    1.0, 1.0, 1.0,
    1.0, 1.0, 9.0,
    1.0, 1.0, 3.0,
    10.0, 8.0, 1.0,
    1.0, 5.0, 12.0,
    1.0, 1.0, 1.0,
];

#[test]
fn cpu_oracle_is_pinned() -> Result<()> {
    for dtype in DTYPES {
        let c = cases(&Device::Cpu, dtype)?;
        assert_eq!(case(&c, "scatter_dim0"), SCATTER_DIM0, "{dtype:?}");
        assert_eq!(case(&c, "scatter_add_dim0"), SCATTER_ADD_DIM0, "{dtype:?}");
        assert_eq!(case(&c, "scatter_dim1"), SCATTER_DIM1, "{dtype:?}");
        assert_eq!(case(&c, "scatter_add_dim1"), SCATTER_ADD_DIM1, "{dtype:?}");
        assert_eq!(case(&c, "scatter_ids_i64"), SCATTER_DIM0, "{dtype:?}");
        assert_eq!(
            case(&c, "scatter_add_ids_i64"),
            SCATTER_ADD_DIM0,
            "{dtype:?}"
        );
        assert_eq!(case(&c, "scatter_skip"), SCATTER_SKIP, "{dtype:?}");
        assert_eq!(case(&c, "scatter_add_skip"), SCATTER_ADD_SKIP, "{dtype:?}");
        assert_eq!(case(&c, "scatter_set"), SCATTER_DIM0, "{dtype:?}");
        assert_eq!(case(&c, "scatter_add_set"), SCATTER_ADD_DIM0, "{dtype:?}");
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
    Ok(())
}
