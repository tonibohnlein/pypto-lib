# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Isolated real-shape driver for the MTP projection dequantization epilogue."""

import pypto.language as pl

from config import DECODE_BATCH, DECODE_SEQ
from mtp_projection import D, HC_MULT, LINEAR_T_TILE, mtp_dequant


T = DECODE_BATCH * DECODE_SEQ
T_LINEAR = ((T + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
T_LINEAR_HC = T_LINEAR * HC_MULT


@pl.jit
def mtp_dequant_test(
    hidden_acc_pad: pl.Tensor[[T_LINEAR, D], pl.INT32],
    prev_acc_pad: pl.Tensor[[T_LINEAR_HC, D], pl.INT32],
    hidden_scale_dq: pl.Tensor[[T_LINEAR, 1], pl.FP32],
    prev_scale_dq: pl.Tensor[[HC_MULT, T_LINEAR], pl.FP32],
    e_proj_w_scale: pl.Tensor[[D], pl.FP32],
    h_proj_w_scale: pl.Tensor[[D], pl.FP32],
    hidden_states_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
):
    return mtp_dequant(
        hidden_acc_pad,
        prev_acc_pad,
        hidden_scale_dq,
        prev_scale_dq,
        e_proj_w_scale,
        h_proj_w_scale,
        hidden_states_out,
    )


def golden_mtp_dequant(tensors):
    hidden = tensors["hidden_acc_pad"][:T].float()
    hidden = hidden * tensors["hidden_scale_dq"][:T].float()
    hidden = hidden * tensors["e_proj_w_scale"].float().view(1, D)
    output = tensors["hidden_states_out"]
    for hc in range(HC_MULT):
        row0 = hc * LINEAR_T_TILE
        previous = tensors["prev_acc_pad"][row0 : row0 + T].float()
        previous = previous * tensors["prev_scale_dq"][hc, :T].float().view(T, 1)
        previous = previous * tensors["h_proj_w_scale"].float().view(1, D)
        output[:, hc] = hidden + previous


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    return [
        TensorSpec(
            "hidden_acc_pad",
            [T_LINEAR, D],
            torch.int32,
            init_value=lambda: torch.randint(-4096, 4097, (T_LINEAR, D), dtype=torch.int32),
        ),
        TensorSpec(
            "prev_acc_pad",
            [T_LINEAR_HC, D],
            torch.int32,
            init_value=lambda: torch.randint(-4096, 4097, (T_LINEAR_HC, D), dtype=torch.int32),
        ),
        TensorSpec(
            "hidden_scale_dq", [T_LINEAR, 1], torch.float32, init_value=lambda: torch.rand(T_LINEAR, 1) * 0.01
        ),
        TensorSpec(
            "prev_scale_dq",
            [HC_MULT, T_LINEAR],
            torch.float32,
            init_value=lambda: torch.rand(HC_MULT, T_LINEAR) * 0.01,
        ),
        TensorSpec("e_proj_w_scale", [D], torch.float32, init_value=lambda: torch.rand(D) * 0.02),
        TensorSpec("h_proj_w_scale", [D], torch.float32, init_value=lambda: torch.rand(D) * 0.02),
        TensorSpec("hidden_states_out", [T, HC_MULT, D], torch.float32, is_output=True),
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
        fn=mtp_dequant_test,
        specs=build_tensor_specs(),
        golden_fn=golden_mtp_dequant,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=1e-4,
        atol=1e-5,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
