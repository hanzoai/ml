import hanzo
from hanzo import Tensor
from hanzo.testing import assert_equal, assert_almost_equal
import pytest


@pytest.mark.parametrize("dtype", [hanzo.f32, hanzo.f64, hanzo.f16, hanzo.u32, hanzo.u8, hanzo.i64])
def test_assert_equal_asserts_correctly(dtype: hanzo.DType):
    a = Tensor([1, 2, 3]).to(dtype)
    b = Tensor([1, 2, 3]).to(dtype)
    assert_equal(a, b)

    with pytest.raises(AssertionError):
        assert_equal(a, b + 1)


@pytest.mark.parametrize("dtype", [hanzo.f32, hanzo.f64, hanzo.f16, hanzo.u32, hanzo.u8, hanzo.i64])
def test_assert_almost_equal_asserts_correctly(dtype: hanzo.DType):
    a = Tensor([1, 2, 3]).to(dtype)
    b = Tensor([1, 2, 3]).to(dtype)
    assert_almost_equal(a, b)

    with pytest.raises(AssertionError):
        assert_almost_equal(a, b + 1)

    assert_almost_equal(a, b + 1, atol=20)
    assert_almost_equal(a, b + 1, rtol=20)

    with pytest.raises(AssertionError):
        assert_almost_equal(a, b + 1, atol=0.9)

    with pytest.raises(AssertionError):
        assert_almost_equal(a, b + 1, rtol=0.1)
