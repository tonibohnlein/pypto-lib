# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Isolated real-shape driver for dynamic DSpark RoPE-row materialization."""

import pypto.language as pl

from config import DECODE_BATCH, DECODE_SEQ, TP
from qkv_proj_rope import MAX_SEQ_LEN, ROPE_DIM, materialize_rope_rows_dynamic


T = DECODE_BATCH // TP * DECODE_SEQ


@pl.jit
def qkv_rope_rows_dynamic_test(
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    position_ids: pl.Tensor[[T], pl.INT32],
    rope_cos_t: pl.Out[pl.Tensor[[T, ROPE_DIM], pl.BF16]],
    rope_sin_t: pl.Out[pl.Tensor[[T, ROPE_DIM], pl.BF16]],
):
    materialize_rope_rows_dynamic(
        freqs_cos,
        freqs_sin,
        position_ids,
        rope_cos_t,
        rope_sin_t,
    )
    return rope_cos_t, rope_sin_t


def golden_qkv_rope_rows_dynamic(tensors):
    positions = tensors["position_ids"].long()
    tensors["rope_cos_t"][:] = tensors["freqs_cos"].index_select(0, positions)
    tensors["rope_sin_t"][:] = tensors["freqs_sin"].index_select(0, positions)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    positions = torch.arange(T, dtype=torch.int32) * 17 + 3

    def init_table(phase: float):
        rows = torch.arange(MAX_SEQ_LEN, dtype=torch.float32).reshape(-1, 1)
        columns = torch.arange(ROPE_DIM, dtype=torch.float32).reshape(1, -1)
        # Constructing the full table analytically is deterministic and avoids
        # relying on the process-global Torch RNG for a 128 MiB input.
        return torch.sin(rows * 0.00003125 + columns * 0.0078125 + phase).to(torch.bfloat16)

    return [
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_DIM], torch.bfloat16, init_value=lambda: init_table(0.0)),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_DIM], torch.bfloat16, init_value=lambda: init_table(1.0)),
        TensorSpec("position_ids", [T], torch.int32, init_value=lambda: positions.clone()),
        TensorSpec("rope_cos_t", [T, ROPE_DIM], torch.bfloat16, is_output=True),
        TensorSpec("rope_sin_t", [T, ROPE_DIM], torch.bfloat16, is_output=True),
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
        fn=qkv_rope_rows_dynamic_test,
        specs=build_tensor_specs(),
        golden_fn=golden_qkv_rope_rows_dynamic,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=0,
        atol=0,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
