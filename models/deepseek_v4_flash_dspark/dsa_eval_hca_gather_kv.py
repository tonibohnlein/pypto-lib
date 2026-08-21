# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Isolated real-shape driver for the HCA sliding-window KV gather."""

import pypto.language as pl

from decode_sparse_attn_hca import (
    B,
    BLOCK_SIZE,
    HEAD_DIM,
    ORI_BLOCK_NUM,
    REQUEST_KV_ROWS,
    S,
    T,
    WIN,
    hca_gather_kv,
)


ORI_ROWS = ORI_BLOCK_NUM * BLOCK_SIZE


@pl.jit
def hca_gather_kv_test(
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    window_swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T], pl.INT32],
    raw_kv: pl.Out[pl.Tensor[[B * REQUEST_KV_ROWS, HEAD_DIM], pl.BF16]],
):
    ori_kv_flat = pl.reshape(ori_kv, [ORI_ROWS, HEAD_DIM])
    cache_ready_dep = pl.system.task_dummy(deps=[])
    hca_gather_kv(ori_kv_flat, window_swa_indices, window_swa_lens, raw_kv, cache_ready_dep)
    return raw_kv


def golden_hca_gather_kv(tensors):
    import torch

    ori_kv_flat = tensors["ori_kv"].reshape(ORI_ROWS, HEAD_DIM)
    indices = tensors["window_swa_indices"].to(torch.int64)
    lens = tensors["window_swa_lens"].to(torch.int64)
    output = tensors["raw_kv"]
    output.zero_()
    for request in range(B):
        first_token = request * S
        first_len = int(lens[first_token])
        output_base = request * REQUEST_KV_ROWS
        first_indices = indices[first_token, :first_len]
        valid = first_indices >= 0
        output[output_base : output_base + first_len][valid] = ori_kv_flat[first_indices[valid]]
        for token_offset in range(1, S):
            token = first_token + token_offset
            slot = int(indices[token, int(lens[token]) - 1])
            if slot >= 0:
                output[output_base + first_len + token_offset - 1] = ori_kv_flat[slot]


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_ori_kv():
        values = torch.arange(ORI_ROWS * HEAD_DIM, dtype=torch.int64)
        return ((values % 257) - 128).reshape(ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM).to(torch.bfloat16)

    def init_window_swa_indices():
        lanes = torch.arange(WIN, dtype=torch.int32)
        indices = torch.full([T, WIN], -1, dtype=torch.int32)
        for request in range(B):
            first_token = request * S
            base = request * (WIN + S + 17)
            indices[first_token] = (base + lanes).remainder(ORI_ROWS)
            for token_offset in range(1, S):
                indices[first_token + token_offset, 0] = (base + WIN + token_offset) % ORI_ROWS
        return indices

    def init_window_swa_lens():
        lens = torch.ones([T], dtype=torch.int32)
        lens[::S] = WIN
        return lens

    return [
        TensorSpec(
            "ori_kv", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_ori_kv
        ),
        TensorSpec("window_swa_indices", [T, WIN], torch.int32, init_value=init_window_swa_indices),
        TensorSpec("window_swa_lens", [T], torch.int32, init_value=init_window_swa_lens),
        TensorSpec("raw_kv", [B * REQUEST_KV_ROWS, HEAD_DIM], torch.bfloat16, is_output=True),
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
        fn=hca_gather_kv_test,
        specs=build_tensor_specs(),
        golden_fn=golden_hca_gather_kv,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=0,
        atol=0,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
