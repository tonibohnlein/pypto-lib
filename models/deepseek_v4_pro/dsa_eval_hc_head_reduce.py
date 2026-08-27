# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Isolated real-shape driver for the Hyper-Connections head reduction."""

import pypto.language as pl

from hc_head import (
    D,
    HC_DIM,
    HC_DIM_INV,
    HC_EPS,
    HC_MULT,
    MIX_SPLIT_TILE,
    STAT_COLS,
    T,
    T_MAX,
    T_TILE,
    EPS,
    hc_head_reduce,
    hc_head_stats_golden,
)


STATS_ROWS = (T_MAX // T_TILE) * (HC_DIM // MIX_SPLIT_TILE)


@pl.jit
def hc_head_reduce_test(
    stats: pl.Tensor[[STATS_ROWS, STAT_COLS], pl.FP32],
    x_flat: pl.Tensor[[T, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    hc_head_reduce(stats, x_flat, hc_head_scale, hc_head_base, y)
    return y


def golden_hc_head_reduce(tensors):
    import torch

    active_rows = (T // T_TILE) * (HC_DIM // MIX_SPLIT_TILE)
    packed = tensors["stats"][:active_rows].reshape(T // T_TILE, HC_DIM // MIX_SPLIT_TILE, STAT_COLS)
    total = (
        packed.sum(dim=1).reshape(T // T_TILE, HC_MULT + 1, T_TILE).permute(0, 2, 1).reshape(T, HC_MULT + 1)
    )
    inv = torch.rsqrt(total[:, 0:1] * HC_DIM_INV + EPS)
    logits = total[:, 1:] * inv * tensors["hc_head_scale"].float() + tensors["hc_head_base"].float()
    pre = torch.sigmoid(logits) + HC_EPS
    x = tensors["x_flat"].reshape(T, HC_MULT, D)
    tensors["y"][:] = (x * pre.unsqueeze(-1)).sum(dim=1).to(torch.bfloat16)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(20260827)
    x_flat = torch.randn(T, HC_DIM, generator=generator) * 0.05
    hc_head_fn = torch.randn(HC_MULT, HC_DIM, generator=generator) * 0.0519
    sq_sum, mixes = hc_head_stats_golden(x_flat, hc_head_fn)
    packed = torch.zeros(STATS_ROWS, STAT_COLS, dtype=torch.float32)
    for token_tile in range(T // T_TILE):
        t0 = token_tile * T_TILE
        for split in range(HC_DIM // MIX_SPLIT_TILE):
            k0 = split * MIX_SPLIT_TILE
            k1 = k0 + MIX_SPLIT_TILE
            x_chunk = x_flat[t0 : t0 + T_TILE, k0:k1]
            stats_row = token_tile * (HC_DIM // MIX_SPLIT_TILE) + split
            packed[stats_row, 0:T_TILE] = x_chunk.square().sum(dim=1)
            for head in range(HC_MULT):
                packed[stats_row, (head + 1) * T_TILE : (head + 2) * T_TILE] = (
                    x_chunk * hc_head_fn[head, k0:k1]
                ).sum(dim=1)
    assert torch.allclose(
        packed[: T // T_TILE * (HC_DIM // MIX_SPLIT_TILE)]
        .reshape(T // T_TILE, -1, STAT_COLS)
        .sum(1)[:, 0:T_TILE]
        .reshape(-1, 1),
        sq_sum,
    )
    assert torch.allclose(
        packed[: T // T_TILE * (HC_DIM // MIX_SPLIT_TILE)]
        .reshape(T // T_TILE, -1, STAT_COLS)
        .sum(1)
        .reshape(T // T_TILE, HC_MULT + 1, T_TILE)[:, 1:]
        .permute(0, 2, 1)
        .reshape(T, HC_MULT),
        mixes,
    )

    return [
        TensorSpec("stats", [STATS_ROWS, STAT_COLS], torch.float32, init_value=lambda: packed),
        TensorSpec("x_flat", [T, HC_DIM], torch.float32, init_value=lambda: x_flat),
        TensorSpec("hc_head_scale", [1], torch.float32, init_value=lambda: torch.tensor([0.076099])),
        TensorSpec(
            "hc_head_base",
            [HC_MULT],
            torch.float32,
            init_value=lambda: torch.tensor([5.9166, -3.6223, -2.9324, -3.3124]),
        ),
        TensorSpec("y", [T, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--dump-passes", action="store_true")
    args = parser.parse_args()

    result = run_jit(
        fn=hc_head_reduce_test,
        specs=build_tensor_specs(),
        golden_fn=golden_hc_head_reduce,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={"y": ratio_allclose(atol=1e-4, rtol=1.0 / 128)},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
