# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Isolated real-shape driver for Hyper-Connections pre/post gate splitting."""

import pypto.language as pl

from hc_pre import (
    DECODE_BATCH,
    DECODE_SEQ,
    HC_DIM,
    HC_EPS,
    HC_MULT,
    HC_PAD,
    LINEAR_K_SPLIT_TILE,
    LINEAR_T_TILE,
    MIX_HC,
    MIX_PAD,
    split_pre_post,
)


T = DECODE_BATCH * DECODE_SEQ
T_LINEAR = ((T + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
LINEAR_SPLITS = HC_DIM // LINEAR_K_SPLIT_TILE
PARTIAL_ROWS = LINEAR_SPLITS * T_LINEAR


@pl.jit
def split_pre_post_test(
    inv_rms: pl.Tensor[[T, 1], pl.FP32],
    mixes_partials: pl.Tensor[[PARTIAL_ROWS, MIX_PAD], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    pre_val: pl.Out[pl.Tensor[[T, HC_PAD], pl.FP32]],
    post: pl.Out[pl.Tensor[[T, HC_MULT], pl.FP32]],
):
    scale0 = pl.read(hc_scale, [0])
    scale1 = pl.read(hc_scale, [1])
    split_pre_post(inv_rms, mixes_partials, hc_base, scale0, scale1, pre_val, post)
    return pre_val, post


def golden_split_pre_post(tensors):
    import torch

    partials = tensors["mixes_partials"].reshape(LINEAR_SPLITS, T_LINEAR, MIX_PAD)
    combined = partials[:, :T].sum(dim=0)
    inv = tensors["inv_rms"].float()
    pre_logits = combined[:, :HC_PAD] * inv * tensors["hc_scale"][0] + tensors["hc_base"][:HC_PAD]
    post_logits = (
        combined[:, HC_MULT : HC_MULT + HC_PAD] * inv * tensors["hc_scale"][1]
        + tensors["hc_base"][HC_MULT : HC_MULT + HC_PAD]
    )
    tensors["pre_val"][:] = torch.sigmoid(pre_logits) + HC_EPS
    tensors["post"][:] = (torch.sigmoid(post_logits) * 2.0)[:, :HC_MULT]


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(20260827)
    inv_rms = torch.rand(T, 1, generator=generator) * 0.25 + 0.75
    partials = torch.randn(PARTIAL_ROWS, MIX_PAD, generator=generator) * 0.1
    hc_scale = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32)
    hc_base = torch.linspace(-0.5, 0.5, MIX_HC, dtype=torch.float32)
    return [
        TensorSpec("inv_rms", [T, 1], torch.float32, init_value=lambda: inv_rms),
        TensorSpec("mixes_partials", [PARTIAL_ROWS, MIX_PAD], torch.float32, init_value=lambda: partials),
        TensorSpec("hc_scale", [3], torch.float32, init_value=lambda: hc_scale),
        TensorSpec("hc_base", [MIX_HC], torch.float32, init_value=lambda: hc_base),
        TensorSpec("pre_val", [T, HC_PAD], torch.float32, is_output=True),
        TensorSpec("post", [T, HC_MULT], torch.float32, is_output=True),
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
        fn=split_pre_post_test,
        specs=build_tensor_specs(),
        golden_fn=golden_split_pre_post,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=1e-4,
        atol=1e-5,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
