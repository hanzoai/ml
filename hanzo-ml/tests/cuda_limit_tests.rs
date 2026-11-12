use hanzo_ml::cuda::{check_elem_count, check_shape_elem_count, MAX_CUDA_ELEMENTS};
use hanzo_ml::{Result, Shape};

#[test]
fn test_cuda_max_element_limit() -> Result<()> {
    assert_eq!(MAX_CUDA_ELEMENTS, 2_147_483_647);
    assert_eq!(MAX_CUDA_ELEMENTS, i32::MAX as usize);

    // Boundary conditions:
    // Exactly at the boundary: must succeed.
    assert!(check_elem_count(MAX_CUDA_ELEMENTS).is_ok());

    // Below the boundary: must succeed.
    assert!(check_elem_count(0).is_ok());
    assert!(check_elem_count(1024).is_ok());
    assert!(check_elem_count(MAX_CUDA_ELEMENTS - 1).is_ok());

    // One past the boundary: must be refused.
    assert!(check_elem_count(MAX_CUDA_ELEMENTS + 1).is_err());
    assert!(check_elem_count(usize::MAX).is_err());

    // Specific case from LLM.md:
    // Padded F32 pass at 16,384 tokens with 12 heads:
    // 16,384 * 16,384 * 12 = 3,221,225,472 scores (> 2^31 - 1)
    let shape = Shape::from((16384, 16384, 12));
    assert_eq!(shape.elem_count(), 3_221_225_472);
    assert!(check_elem_count(shape.elem_count()).is_err());
    assert!(check_shape_elem_count(&shape).is_err());

    Ok(())
}

#[test]
fn test_cuda_shape_checked_arithmetic_overflow() -> Result<()> {
    // Normal shape:
    let shape = Shape::from((2, 3, 4));
    assert_eq!(check_shape_elem_count(&shape)?, 24);

    // Shape that wraps to 0 in 64-bit unchecked multiplication (Codex review case):
    // (1usize << 32) * (1usize << 32) wraps to 0 modulo 2^64.
    // check_shape_elem_count uses checked multiplication so it must return Err, not Ok(0).
    let wrapping_shape = Shape::from((1usize << 32, 1usize << 32));
    assert!(check_shape_elem_count(&wrapping_shape).is_err());

    // Shape with multi-dim product exceeding usize:
    let huge_shape = Shape::from((1usize << 40, 1usize << 30));
    assert!(check_shape_elem_count(&huge_shape).is_err());

    // Shape with single dim exceeding MAX_CUDA_ELEMENTS:
    let single_huge_dim = Shape::from((MAX_CUDA_ELEMENTS + 1, 1));
    assert!(check_shape_elem_count(&single_huge_dim).is_err());

    // Shape exactly at the boundary:
    let exact_boundary = Shape::from((MAX_CUDA_ELEMENTS, 1));
    assert_eq!(check_shape_elem_count(&exact_boundary)?, MAX_CUDA_ELEMENTS);

    Ok(())
}

#[test]
fn test_cuda_device_refuses_large_tensor() -> Result<()> {
    if !hanzo_ml::utils::cuda_is_available() {
        return Ok(());
    }
    let dev = hanzo_ml::Device::new_cuda(0)?;
    let shape = Shape::from((16384, 16384, 12));
    let res = hanzo_ml::Tensor::zeros(shape, hanzo_ml::DType::F32, &dev);
    assert!(res.is_err());
    Ok(())
}
