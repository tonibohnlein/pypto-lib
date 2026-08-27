# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Isolated real-shape driver for DSpark KV commit and sparse validity bias."""

import pypto.language as pl

from dspark_attention import (
    B,
    BLOCK_SIZE,
    HEAD_DIM,
    INDEX_WIDTH,
    KV_ORI_BLOCK_NUM,
    NEG_INF,
    T,
    dspark_kv_commit_valid_bias,
)


@pl.jit
def kv_commit_valid_bias_test(
    kv: pl.Tensor[[T, HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[pl.Tensor[[KV_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    kv_slot_mapping: pl.Tensor[[T], pl.INT64],
    swa_lens: pl.Tensor[[B], pl.INT32],
    sparse_bias: pl.Out[pl.Tensor[[B, INDEX_WIDTH], pl.FP32]],
):
    dspark_kv_commit_valid_bias(kv, kv_cache, kv_slot_mapping, swa_lens, sparse_bias)
    return kv_cache, sparse_bias


def golden_kv_commit_valid_bias(tensors):
    import torch

    slots = tensors["kv_slot_mapping"].long()
    valid = slots >= 0
    tensors["kv_cache"].reshape(-1, HEAD_DIM)[slots[valid]] = tensors["kv"][valid]
    columns = torch.arange(INDEX_WIDTH, dtype=torch.int32).reshape(1, -1)
    valid_columns = columns < tensors["swa_lens"].reshape(-1, 1)
    tensors["sparse_bias"][:] = torch.where(valid_columns, 0.0, NEG_INF)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_kv():
        rows = torch.arange(T, dtype=torch.float32).reshape(-1, 1)
        columns = torch.arange(HEAD_DIM, dtype=torch.float32).reshape(1, -1)
        return torch.sin(rows * 0.03125 + columns * 0.0078125).to(torch.bfloat16)

    return [
        TensorSpec("kv", [T, HEAD_DIM], torch.bfloat16, init_value=init_kv),
        TensorSpec(
            "kv_cache",
            [KV_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16,
            init_value=lambda: torch.zeros(KV_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16),
            is_output=True,
        ),
        TensorSpec(
            "kv_slot_mapping", [T], torch.int64, init_value=lambda: torch.arange(T, dtype=torch.int64)
        ),
        TensorSpec(
            "swa_lens",
            [B],
            torch.int32,
            init_value=lambda: torch.linspace(1, INDEX_WIDTH, B, dtype=torch.float32).to(torch.int32),
        ),
        TensorSpec("sparse_bias", [B, INDEX_WIDTH], torch.float32, is_output=True),
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
        fn=kv_commit_valid_bias_test,
        specs=build_tensor_specs(),
        golden_fn=golden_kv_commit_valid_bias,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=0,
        atol=0,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
