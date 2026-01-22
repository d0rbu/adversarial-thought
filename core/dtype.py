import torch as t

DTYPE_ALIASES: dict[t.dtype, set[str]] = {
    t.float32: {"float32", "float", "fp32", "f32"},
    t.float16: {"float16", "fp16", "f16"},
    t.bfloat16: {"bfloat16", "bf16"},
    t.float64: {"float64", "fp64", "f64"},
}
DTYPE_MAP: dict[str, t.dtype] = {
    alias: dtype for dtype, aliases in DTYPE_ALIASES.items() for alias in aliases
}


def get_dtype(dtype_str: str) -> t.dtype:
    dtype = DTYPE_MAP.get(dtype_str)
    if dtype is None:
        raise ValueError(f"Invalid dtype: {dtype_str}")

    return dtype
