# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Isolated real-shape driver for DSpark RoPE row preparation."""

import pypto.language as pl

from dspark_context_kv import DSPARK_QUERY_TOKENS
from qkv_proj_rope import ROPE_DIM, rope_prepare


@pl.jit
def rope_prepare_test(
    rope_cos: pl.Tensor[[DSPARK_QUERY_TOKENS, ROPE_DIM], pl.BF16],
    rope_sin: pl.Tensor[[DSPARK_QUERY_TOKENS, ROPE_DIM], pl.BF16],
    rope_cos_il: pl.Out[pl.Tensor[[DSPARK_QUERY_TOKENS, ROPE_DIM], pl.FP32]],
    rope_sin_signed: pl.Out[pl.Tensor[[DSPARK_QUERY_TOKENS, ROPE_DIM], pl.FP32]],
    rope_swap_idx: pl.Out[pl.Tensor[[DSPARK_QUERY_TOKENS, ROPE_DIM], pl.INT32]],
):
    rope_prepare(rope_cos, rope_sin, rope_cos_il, rope_sin_signed, rope_swap_idx)
    return rope_cos_il, rope_sin_signed, rope_swap_idx


def golden_rope_prepare(tensors):
    import torch

    columns = torch.arange(ROPE_DIM, dtype=torch.int64)
    duplicate_indices = torch.div(columns, 2, rounding_mode="floor")
    signs = torch.where(columns.remainder(2) == 0, -1.0, 1.0)

    tensors["rope_cos_il"][:] = tensors["rope_cos"].float()[:, duplicate_indices]
    tensors["rope_sin_signed"][:] = tensors["rope_sin"].float()[:, duplicate_indices] * signs
    tensors["rope_swap_idx"][:] = columns.bitwise_xor(1).to(torch.int32).expand(DSPARK_QUERY_TOKENS, ROPE_DIM)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_rope_rows(phase):
        rows = torch.arange(DSPARK_QUERY_TOKENS, dtype=torch.float32).reshape(-1, 1)
        columns = torch.arange(ROPE_DIM, dtype=torch.float32).reshape(1, -1)
        return torch.sin(rows * 0.03125 + columns * 0.0078125 + phase).to(torch.bfloat16)

    return [
        TensorSpec(
            "rope_cos",
            [DSPARK_QUERY_TOKENS, ROPE_DIM],
            torch.bfloat16,
            init_value=lambda: init_rope_rows(0.0),
        ),
        TensorSpec(
            "rope_sin",
            [DSPARK_QUERY_TOKENS, ROPE_DIM],
            torch.bfloat16,
            init_value=lambda: init_rope_rows(0.5),
        ),
        TensorSpec("rope_cos_il", [DSPARK_QUERY_TOKENS, ROPE_DIM], torch.float32, is_output=True),
        TensorSpec("rope_sin_signed", [DSPARK_QUERY_TOKENS, ROPE_DIM], torch.float32, is_output=True),
        TensorSpec("rope_swap_idx", [DSPARK_QUERY_TOKENS, ROPE_DIM], torch.int32, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--dump-passes", action="store_true")
    args = parser.parse_args()

    result = run_jit(
        fn=rope_prepare_test,
        specs=build_tensor_specs(),
        golden_fn=golden_rope_prepare,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=0,
        atol=0,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
