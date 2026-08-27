# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Isolated real-shape driver for DSpark output-LoRA quantization."""

import pypto.language as pl

from dspark_attention import O_LORA_DIM, T, dspark_o_lora_quant


@pl.jit
def o_lora_quant_test(
    o_lora: pl.Tensor[[T, O_LORA_DIM], pl.BF16],
    o_lora_i8: pl.Out[pl.Tensor[[T, O_LORA_DIM], pl.INT8]],
    o_lora_scale: pl.Out[pl.Tensor[[T, 1], pl.FP32]],
):
    dspark_o_lora_quant(o_lora, o_lora_i8, o_lora_scale)
    return o_lora_i8, o_lora_scale


def golden_o_lora_quant(tensors):
    from utils import int8_quant_per_row

    o_lora_i8, o_lora_scale = int8_quant_per_row(tensors["o_lora"].float())
    tensors["o_lora_i8"][:] = o_lora_i8
    tensors["o_lora_scale"][:] = o_lora_scale


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_o_lora():
        rows = torch.arange(T, dtype=torch.float32).reshape(-1, 1)
        columns = torch.arange(O_LORA_DIM, dtype=torch.float32).reshape(1, -1)
        return (torch.sin(rows * 0.03125 + columns * 0.001953125) * 2.0).to(torch.bfloat16)

    return [
        TensorSpec("o_lora", [T, O_LORA_DIM], torch.bfloat16, init_value=init_o_lora),
        TensorSpec("o_lora_i8", [T, O_LORA_DIM], torch.int8, is_output=True),
        TensorSpec("o_lora_scale", [T, 1], torch.float32, is_output=True),
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
        fn=o_lora_quant_test,
        specs=build_tensor_specs(),
        golden_fn=golden_o_lora_quant,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=1e-4,
        atol=1e-5,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
