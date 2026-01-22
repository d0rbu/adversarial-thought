"""Tests for core utilities."""

import pytest
import torch

from core.dtype import DTYPE_MAP, get_dtype
from core.type import assert_type


class TestAssertType:
    """Tests for assert_type utility."""

    def test_assert_type_correct_type(self) -> None:
        """Test that assert_type returns the object when type matches."""
        value = "hello"
        result = assert_type(value, str)
        assert result == value
        assert result is value

    def test_assert_type_int(self) -> None:
        """Test assert_type with integers."""
        value = 42
        result = assert_type(value, int)
        assert result == 42

    def test_assert_type_list(self) -> None:
        """Test assert_type with lists."""
        value = [1, 2, 3]
        result = assert_type(value, list)
        assert result == [1, 2, 3]

    def test_assert_type_wrong_type_raises(self) -> None:
        """Test that assert_type raises TypeError for wrong types."""
        with pytest.raises(TypeError, match="Expected str, got int"):
            assert_type(42, str)

    def test_assert_type_none_raises(self) -> None:
        """Test that assert_type raises TypeError for None."""
        with pytest.raises(TypeError, match="Expected str, got NoneType"):
            assert_type(None, str)


class TestGetDtype:
    """Tests for get_dtype utility."""

    def test_get_dtype_float32_aliases(self) -> None:
        """Test all float32 aliases."""
        for alias in ["float32", "float", "fp32", "f32"]:
            result = get_dtype(alias)
            assert result == torch.float32

    def test_get_dtype_float16_aliases(self) -> None:
        """Test all float16 aliases."""
        for alias in ["float16", "fp16", "f16"]:
            result = get_dtype(alias)
            assert result == torch.float16

    def test_get_dtype_bfloat16_aliases(self) -> None:
        """Test all bfloat16 aliases."""
        for alias in ["bfloat16", "bf16"]:
            result = get_dtype(alias)
            assert result == torch.bfloat16

    def test_get_dtype_float64_aliases(self) -> None:
        """Test all float64 aliases."""
        for alias in ["float64", "fp64", "f64"]:
            result = get_dtype(alias)
            assert result == torch.float64

    def test_get_dtype_invalid_raises(self) -> None:
        """Test that invalid dtype raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dtype: invalid"):
            get_dtype("invalid")

    def test_dtype_map_consistency(self) -> None:
        """Test that DTYPE_MAP is consistent."""
        # Ensure all mapped dtypes are valid torch dtypes
        for alias, dtype in DTYPE_MAP.items():
            assert isinstance(alias, str)
            assert isinstance(dtype, torch.dtype)
