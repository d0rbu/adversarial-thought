"""Steering hooks for activation injection.

This module provides a dtype-aware reimplementation of nl_probes steering hooks
that properly handles dtype matching to avoid RuntimeError when injecting activations
into models with different dtypes (e.g., float16, bfloat16).
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import torch as t

if TYPE_CHECKING:
    from collections.abc import Callable


@contextlib.contextmanager
def add_hook(
    module: t.nn.Module,
    hook: Callable,
):
    """Temporarily adds a forward hook to a model module.

    Args:
        module: The PyTorch module to hook
        hook: The hook function to apply

    Yields:
        None: Used as a context manager

    Example:
        with add_hook(model.layer, hook_fn):
            output = model(input)
    """
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def get_hf_activation_steering_hook(
    vectors: list[t.Tensor],  # len B, each tensor is (K_b, d_model)
    positions: list[list[int]],  # len B, each list has length K_b
    steering_coefficient: float,
    device: t.device,
    dtype: t.dtype,
) -> Callable:
    """HF hook with proper dtype handling.

    Supports a variable number of target positions per batch element.

    Semantics:
      For each batch item b and slot k, replace the residual at token index positions[b][k]
      with normalize(vectors[b][k]) * ||resid[b, positions[b][k], :]|| * steering_coefficient.

    This is a dtype-aware reimplementation that ensures all operations maintain
    the correct dtype to avoid RuntimeError when injecting into float16/bfloat16 models.

    Args:
        vectors: List of tensors, one per batch element. Each tensor is (K_b, d_model)
        positions: List of position lists, one per batch element. Each list has K_b positions
        steering_coefficient: Coefficient for steering strength
        device: Device to place tensors on
        dtype: Target dtype for all operations (must match model's activation dtype)

    Returns:
        Hook function that can be registered with add_hook()
    """
    assert len(vectors) == len(
        positions
    ), "vectors and positions must have same batch length"
    B = len(vectors)
    if B == 0:
        raise ValueError("Empty batch")

    # Pre-normalize once and convert to target dtype immediately
    # This ensures all subsequent operations maintain the correct dtype
    normed_list = [
        t.nn.functional.normalize(v_b, dim=-1).to(device=device, dtype=dtype).detach()
        for v_b in vectors
    ]

    def hook_fn(_module, _input, output):
        # Normalize output API across model families
        if isinstance(output, tuple):
            resid_BLD, *rest = output
            output_is_tuple = True
        else:
            resid_BLD = output
            output_is_tuple = False

        B_actual, L, _d_model_actual = resid_BLD.shape
        if B_actual != B:
            raise ValueError(
                f"Batch mismatch: module B={B_actual}, provided vectors B={B}"
            )

        # Only touch the prompt forward pass
        if L <= 1:
            return (resid_BLD, *rest) if output_is_tuple else resid_BLD

        # Per-batch element work. Vectorized over K_b where safe.
        for b in range(B):
            pos_b = positions[b]
            pos_b = t.tensor(pos_b, dtype=t.long, device=device)
            assert pos_b.min() >= 0
            assert pos_b.max() < L
            # Gather original activations at requested slots and compute norms
            orig_KD = resid_BLD[b, pos_b, :]  # (K_b, d)
            # Ensure orig_KD is in the correct dtype (should already be, but be safe)
            orig_KD = orig_KD.to(dtype=dtype)
            norms_K1 = orig_KD.norm(dim=-1, keepdim=True)  # (K_b, 1)

            if b == 0 and norms_K1.max() > 300:
                print(
                    f"\n\n\n\n\nWARNING: Large norm detected in batch! {norms_K1}\n\n\n\n\n"
                )

            # Build steered vectors for this b
            # normed_list[b] is already in dtype, norms_K1 is computed from orig_KD which is in dtype
            # so the result will be in dtype
            steered_KD = normed_list[b] * norms_K1 * steering_coefficient  # (K_b, d)

            # Ensure the result is in the correct dtype before assignment
            # The addition should preserve dtype since both operands are in dtype
            steered_result = (steered_KD.detach() + orig_KD).to(dtype=dtype)
            resid_BLD[b, pos_b, :] = steered_result

        return (resid_BLD, *rest) if output_is_tuple else resid_BLD

    return hook_fn
