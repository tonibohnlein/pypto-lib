# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Isolated real-shape driver for the production sparse-attention bias builder."""

import pypto.language as pl

from config import FP32_NEG_INF
from prefill_sparse_attn import (
    IDX_TOPK,
    PREFILL_SPARSE_PAD,
    SPARSE_BIAS_COLS,
    SPARSE_CMP_BIAS_COLS,
    T,
    WIN,
    build_sparse_bias,
)


@pl.jit
def sparse_bias_test(
    swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    cmp_indices: pl.Tensor[[T, IDX_TOPK], pl.INT32],
    sparse_bias: pl.Out[pl.Tensor[[T, PREFILL_SPARSE_PAD], pl.FP32]],
):
    build_sparse_bias(swa_indices, cmp_indices, sparse_bias)
    return sparse_bias


def golden_sparse_bias(tensors):
    import torch

    output = torch.full(
        (T, PREFILL_SPARSE_PAD),
        FP32_NEG_INF,
        dtype=torch.float32,
    )
    output[:, :WIN] = torch.where(
        tensors["swa_indices"] >= 0,
        0.0,
        FP32_NEG_INF,
    )
    if SPARSE_CMP_BIAS_COLS > 0:
        output[:, WIN:SPARSE_BIAS_COLS] = torch.where(
            tensors["cmp_indices"][:, :SPARSE_CMP_BIAS_COLS] >= 0,
            0.0,
            FP32_NEG_INF,
        )
    tensors["sparse_bias"][:] = output


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    rows = torch.arange(T, dtype=torch.int32).reshape(-1, 1)
    swa_columns = torch.arange(WIN, dtype=torch.int32).reshape(1, -1)
    cmp_columns = torch.arange(IDX_TOPK, dtype=torch.int32).reshape(1, -1)
    swa_indices = rows * WIN + swa_columns
    cmp_indices = rows * IDX_TOPK + cmp_columns
    swa_indices = torch.where((rows + swa_columns) % 5 == 0, -1, swa_indices)
    cmp_indices = torch.where((rows + cmp_columns) % 7 == 0, -1, cmp_indices)
    return [
        TensorSpec("swa_indices", [T, WIN], torch.int32, init_value=lambda: swa_indices.clone()),
        TensorSpec("cmp_indices", [T, IDX_TOPK], torch.int32, init_value=lambda: cmp_indices.clone()),
        TensorSpec("sparse_bias", [T, PREFILL_SPARSE_PAD], torch.float32, is_output=True),
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
        fn=sparse_bias_test,
        specs=build_tensor_specs(),
        golden_fn=golden_sparse_bias,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=0,
        atol=0,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
