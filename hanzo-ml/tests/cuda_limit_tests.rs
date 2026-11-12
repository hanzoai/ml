use hanzo_ml::cuda::{check_elem_count, MAX_CUDA_ELEMENTS};
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
