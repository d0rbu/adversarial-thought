"""Tests for core utilities (type.py and dtype.py).

This module contains both unit tests and property-based tests for the core utilities.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from core.dtype import DTYPE_ALIASES, DTYPE_MAP, get_dtype
from core.type import assert_type

# =============================================================================
# Strategies for property-based testing
# =============================================================================

# Strategy for valid dtype alias strings
valid_dtype_aliases = st.sampled_from(list(DTYPE_MAP.keys()))

# Strategy for invalid dtype strings (not in DTYPE_MAP)
invalid_dtype_strings = st.text(min_size=1, max_size=20).filter(
    lambda s: s not in DTYPE_MAP
)


# =============================================================================
# Tests for core/type.py - assert_type
# =============================================================================


class TestAssertType:
    """Unit tests for assert_type utility."""

    def test_returns_object_when_type_matches(self) -> None:
        """Test that assert_type returns the object when type matches."""
        value = "hello"
        result = assert_type(value, str)
        assert result == value
        assert result is value

    def test_with_integers(self) -> None:
        """Test assert_type with integers."""
        value = 42
        result = assert_type(value, int)
        assert result == 42

    def test_with_lists(self) -> None:
        """Test assert_type with lists."""
        value = [1, 2, 3]
        result = assert_type(value, list)
        assert result == [1, 2, 3]

    def test_raises_for_wrong_type(self) -> None:
        """Test that assert_type raises TypeError for wrong types."""
        with pytest.raises(TypeError, match="Expected str, got int"):
            assert_type(42, str)

    def test_raises_for_none(self) -> None:
        """Test that assert_type raises TypeError for None."""
        with pytest.raises(TypeError, match="Expected str, got NoneType"):
            assert_type(None, str)


class TestAssertTypeProperties:
    """Property-based tests for assert_type utility."""

    @given(st.integers())
    def test_returns_identity_for_any_int(self, value: int) -> None:
        """For any integer, assert_type returns the exact same object."""
        result = assert_type(value, int)
        assert result is value

    @given(st.text())
    def test_returns_identity_for_any_str(self, value: str) -> None:
        """For any string, assert_type returns the exact same object."""
        result = assert_type(value, str)
        assert result is value

    @given(st.floats(allow_nan=False))
    def test_returns_identity_for_any_float(self, value: float) -> None:
        """For any float, assert_type returns the exact same object."""
        result = assert_type(value, float)
        assert result is value

    @given(st.lists(st.integers()))
    def test_returns_identity_for_any_list(self, value: list[int]) -> None:
        """For any list, assert_type returns the exact same object."""
        result = assert_type(value, list)
        assert result is value

    @given(st.dictionaries(st.text(), st.integers()))
    def test_returns_identity_for_any_dict(self, value: dict[str, int]) -> None:
        """For any dict, assert_type returns the exact same object."""
        result = assert_type(value, dict)
        assert result is value

    @given(st.integers())
    def test_raises_for_int_asserted_as_str(self, value: int) -> None:
        """Asserting int as str should raise TypeError with correct message."""
        try:
            assert_type(value, str)
            raise AssertionError("Expected TypeError")
        except TypeError as e:
            assert "Expected str" in str(e)
            assert "got int" in str(e)

    @given(st.text())
    def test_raises_for_str_asserted_as_int(self, value: str) -> None:
        """Asserting str as int should raise TypeError with correct message."""
        try:
            assert_type(value, int)
            raise AssertionError("Expected TypeError")
        except TypeError as e:
            assert "Expected int" in str(e)
            assert "got str" in str(e)

    @given(st.booleans())
    def test_bool_is_int_subtype(self, value: bool) -> None:
        """Bool is a subtype of int in Python, so assert_type(bool, int) should pass."""
        result = assert_type(value, int)
        assert result is value


# =============================================================================
# Tests for core/dtype.py - get_dtype and DTYPE_MAP
# =============================================================================


class TestGetDtype:
    """Unit tests for get_dtype utility."""

    def test_float32_aliases(self) -> None:
        """Test all float32 aliases."""
        for alias in ["float32", "float", "fp32", "f32"]:
            result = get_dtype(alias)
            assert result == torch.float32

    def test_float16_aliases(self) -> None:
        """Test all float16 aliases."""
        for alias in ["float16", "fp16", "f16"]:
            result = get_dtype(alias)
            assert result == torch.float16

    def test_bfloat16_aliases(self) -> None:
        """Test all bfloat16 aliases."""
        for alias in ["bfloat16", "bf16"]:
            result = get_dtype(alias)
            assert result == torch.bfloat16

    def test_float64_aliases(self) -> None:
        """Test all float64 aliases."""
        for alias in ["float64", "fp64", "f64"]:
            result = get_dtype(alias)
            assert result == torch.float64

    def test_invalid_raises_value_error(self) -> None:
        """Test that invalid dtype raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dtype: invalid"):
            get_dtype("invalid")

    def test_dtype_map_consistency(self) -> None:
        """Test that DTYPE_MAP is consistent."""
        for alias, dtype in DTYPE_MAP.items():
            assert isinstance(alias, str)
            assert isinstance(dtype, torch.dtype)


class TestGetDtypeProperties:
    """Property-based tests for get_dtype utility."""

    @given(valid_dtype_aliases)
    def test_returns_valid_torch_dtype(self, alias: str) -> None:
        """For any valid alias, get_dtype returns a valid torch.dtype."""
        result = get_dtype(alias)
        assert isinstance(result, torch.dtype)

    @given(valid_dtype_aliases)
    def test_consistent_with_dtype_map(self, alias: str) -> None:
        """get_dtype should return the same value as direct DTYPE_MAP lookup."""
        result = get_dtype(alias)
        expected = DTYPE_MAP[alias]
        assert result == expected

    @given(invalid_dtype_strings)
    def test_raises_for_invalid_alias(self, invalid_alias: str) -> None:
        """For any invalid alias, get_dtype raises ValueError."""
        try:
            get_dtype(invalid_alias)
            raise AssertionError("Expected ValueError")
        except ValueError as e:
            assert "Invalid dtype" in str(e)
            assert invalid_alias in str(e)

    @given(st.sampled_from(list(DTYPE_ALIASES.keys())))
    def test_all_aliases_return_same_dtype(self, target_dtype: torch.dtype) -> None:
        """All aliases for a given dtype should return the same torch.dtype."""
        aliases = DTYPE_ALIASES[target_dtype]
        results = [get_dtype(alias) for alias in aliases]
        assert all(r == target_dtype for r in results)

    def test_dtype_map_completeness(self) -> None:
        """Every alias in DTYPE_ALIASES should be in DTYPE_MAP."""
        for dtype, aliases in DTYPE_ALIASES.items():
            for alias in aliases:
                assert alias in DTYPE_MAP
                assert DTYPE_MAP[alias] == dtype
