# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Isolated real-shape driver for the decode-indexer query dequantization."""

import pypto.language as pl

from decode_indexer import (
    IDX_HEAD_DIM,
    IDX_N_HEADS,
    T,
    T_PAD,
    idx_qr_proj_dequant,
)


Q_OUT = IDX_N_HEADS * IDX_HEAD_DIM


@pl.jit
def idx_qr_proj_dequant_test(
    qr_acc_pad: pl.Tensor[[T_PAD, Q_OUT], pl.INT32],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    wq_b_scale: pl.Tensor[[Q_OUT], pl.FP32],
    qr_proj: pl.Out[pl.Tensor[[T, Q_OUT], pl.FP32]],
):
    idx_qr_proj_dequant(qr_acc_pad, qr_scale, wq_b_scale, qr_proj)
    return qr_proj


def golden_idx_qr_proj_dequant(tensors):
    tensors["qr_proj"][:] = (
        tensors["qr_acc_pad"][:T].float()
        * tensors["qr_scale"].float()
        * tensors["wq_b_scale"].float().view(1, Q_OUT)
    )


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    return [
        TensorSpec(
            "qr_acc_pad",
            [T_PAD, Q_OUT],
            torch.int32,
            init_value=lambda: torch.randint(-4096, 4097, (T_PAD, Q_OUT), dtype=torch.int32),
        ),
        TensorSpec("qr_scale", [T, 1], torch.float32, init_value=lambda: torch.rand(T, 1) * 0.01),
        TensorSpec("wq_b_scale", [Q_OUT], torch.float32, init_value=lambda: torch.rand(Q_OUT) * 0.02),
        TensorSpec("qr_proj", [T, Q_OUT], torch.float32, is_output=True),
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
        fn=idx_qr_proj_dequant_test,
        specs=build_tensor_specs(),
        golden_fn=golden_idx_qr_proj_dequant,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=1e-4,
        atol=1e-5,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
