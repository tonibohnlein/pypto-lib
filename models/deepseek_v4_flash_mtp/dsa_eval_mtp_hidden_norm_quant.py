# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Isolated real-shape driver for MTP hidden-state RMSNorm and quantization."""

import pypto.language as pl

from config import DECODE_BATCH, DECODE_SEQ
from mtp_projection import (
    D,
    _quantize_rows,
    _rms_norm,
    mtp_hidden_norm_quant,
)


T = DECODE_BATCH * DECODE_SEQ


@pl.jit
def mtp_hidden_norm_quant_test(
    hidden_states: pl.Tensor[[T, D], pl.BF16],
    enorm_w: pl.Tensor[[D], pl.FP32],
    e_proj_smooth: pl.Tensor[[D], pl.FP32],
    hidden_i8: pl.Out[pl.Tensor[[T, D], pl.INT8]],
    hidden_scale_dq: pl.Out[pl.Tensor[[T, 1], pl.FP32]],
):
    mtp_hidden_norm_quant(hidden_states, enorm_w, e_proj_smooth, hidden_i8, hidden_scale_dq)
    return hidden_i8, hidden_scale_dq


def golden_mtp_hidden_norm_quant(tensors):
    hidden_norm = _rms_norm(tensors["hidden_states"], tensors["enorm_w"])
    hidden_weighted = hidden_norm * tensors["e_proj_smooth"].float()
    hidden_i8, hidden_scale = _quantize_rows(hidden_weighted)
    tensors["hidden_i8"][:] = hidden_i8
    tensors["hidden_scale_dq"][:] = hidden_scale


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    return [
        TensorSpec("hidden_states", [T, D], torch.bfloat16, init_value=lambda: torch.randn(T, D)),
        TensorSpec("enorm_w", [D], torch.float32, init_value=lambda: torch.randn(D) * 0.1 + 1.0),
        TensorSpec("e_proj_smooth", [D], torch.float32, init_value=lambda: torch.randn(D) * 0.05 + 1.0),
        TensorSpec("hidden_i8", [T, D], torch.int8, is_output=True),
        TensorSpec("hidden_scale_dq", [T, 1], torch.float32, is_output=True),
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
        fn=mtp_hidden_norm_quant_test,
        specs=build_tensor_specs(),
        golden_fn=golden_mtp_hidden_norm_quant,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=1e-4,
        atol=1e-5,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
