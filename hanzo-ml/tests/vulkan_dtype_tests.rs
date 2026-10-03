//! Reductions and binary ops on Vulkan answer as the CPU backend does for the same op and input
//! dtype: in the same dtype (argmax and argmin in U32), with equal values. The backend holds f32
//! and u32 only: an uploaded bf16 is held as f32 and an i64 as u32, though the tensor keeps the
//! dtype it was uploaded as. The input an op sees, and the CPU reference computes on, is the dtype
//! the storage holds. An op the backend does not run is refused with an error, never answered in
//! another dtype. Skips with no GPU.
#![cfg(feature = "vulkan")]

use hanzo_ml::{DType, Device, Result, Tensor};

const DTYPES: [DType; 4] = [DType::U32, DType::I64, DType::F32, DType::BF16];

fn gpu() -> Option<Device> {
    match Device::new_vulkan(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("[dtype] no Vulkan GPU ({e}); skipping");
            None
        }
    }
}

// `base` as `dt` on Vulkan, and on the CPU in the dtype Vulkan holds it in.
fn pair(base: &Tensor, dt: DType, vk: &Device) -> Result<(Tensor, Tensor)> {
    let on_vk = base.to_dtype(dt)?.to_device(vk)?;
    let held = on_vk.storage_and_layout().0.dtype();
    Ok((on_vk, base.to_dtype(held)?))
}

// Every value of a tensor, on the CPU and widened to f64 (exact for every dtype tested here).
fn values(t: &Tensor) -> Result<Vec<f64>> {
    t.to_device(&Device::Cpu)?
        .to_dtype(DType::F64)?
        .flatten_all()?
        .to_vec1::<f64>()
}

// `got` (Vulkan) against `want` (CPU): same dtype, same shape, same values.
fn check(got: &Tensor, want: &Tensor, what: &str) -> Result<()> {
    assert_eq!(got.dtype(), want.dtype(), "{what}: dtype");
    assert_eq!(got.dims(), want.dims(), "{what}: shape");
    assert_eq!(values(got)?, values(want)?, "{what}: values");
    Ok(())
}

type Reduce = fn(&Tensor, usize) -> Result<Tensor>;

const REDUCE: [(&str, Reduce); 5] = [
    ("sum", |t, d| t.sum(d)),
    ("max", |t, d| t.max(d)),
    ("min", |t, d| t.min(d)),
    ("argmax", |t, d| t.argmax(d)),
    ("argmin", |t, d| t.argmin(d)),
];

#[test]
fn vulkan_reductions_answer_as_the_cpu_does() -> Result<()> {
    let Some(vk) = gpu() else { return Ok(()) };
    // Small integers, exact in every dtype; ties in rows and columns pin the first-index rule.
    let base = Tensor::new(
        &[[3u32, 9, 1, 9, 4], [7, 0, 7, 2, 0], [5, 5, 5, 5, 5]],
        &Device::Cpu,
    )?;
    for dt in DTYPES {
        let (on_vk, on_cpu) = pair(&base, dt, &vk)?;
        let held = on_cpu.dtype();
        for (name, op) in REDUCE {
            for dim in 0..2 {
                let what = format!("{name} over dim {dim} of {dt:?} (held as {held:?})");
                check(&op(&on_vk, dim)?, &op(&on_cpu, dim)?, &what)?;
            }
        }
        // A strided view: the column-major transpose, reduced along its last dim.
        let t = on_vk.t()?;
        check(&t.sum(1)?, &on_cpu.t()?.sum(1)?, &format!("sum of {dt:?}ᵀ"))?;
        // Over both axes at once: equal to the CPU's answer, or refused.
        let what = format!("sum over both dims of {dt:?}");
        match on_vk.sum((0, 1)) {
            Ok(got) => check(&got, &on_cpu.sum((0, 1))?, &what)?,
            Err(e) => eprintln!("[dtype] {what}: refused ({e})"),
        }
    }
    Ok(())
}

type Binary = fn(&Tensor, &Tensor) -> Result<Tensor>;

const BINARY: [(&str, Binary); 6] = [
    ("add", |a, b| a.broadcast_add(b)),
    ("sub", |a, b| a.broadcast_sub(b)),
    ("mul", |a, b| a.broadcast_mul(b)),
    ("div", |a, b| a.broadcast_div(b)),
    ("maximum", |a, b| a.broadcast_maximum(b)),
    ("minimum", |a, b| a.broadcast_minimum(b)),
];

#[test]
fn vulkan_binary_ops_answer_as_the_cpu_does() -> Result<()> {
    let Some(vk) = gpu() else { return Ok(()) };
    // Each row at least its divisor, a power of two: no u32 underflow, every quotient exact.
    let a = Tensor::new(
        &[[3u32, 9, 1, 9, 4], [7, 2, 7, 2, 4], [5, 5, 5, 5, 5]],
        &Device::Cpu,
    )?;
    let b = Tensor::new(&[[1u32], [2], [4]], &Device::Cpu)?;
    let full = b.broadcast_as(a.shape())?.contiguous()?;
    for dt in DTYPES {
        let (a_vk, a_cpu) = pair(&a, dt, &vk)?;
        let (b_vk, b_cpu) = pair(&b, dt, &vk)?;
        let (f_vk, f_cpu) = pair(&full, dt, &vk)?;
        let held = a_cpu.dtype();
        for (name, op) in BINARY {
            // The column broadcast across each row (stride 0), then laid out in full.
            let what = format!("{name} of {dt:?} by a column (held as {held:?})");
            check(&op(&a_vk, &b_vk)?, &op(&a_cpu, &b_cpu)?, &what)?;
            let what = format!("{name} of {dt:?} by a contiguous tensor (held as {held:?})");
            check(&op(&a_vk, &f_vk)?, &op(&a_cpu, &f_cpu)?, &what)?;
        }
    }
    Ok(())
}
