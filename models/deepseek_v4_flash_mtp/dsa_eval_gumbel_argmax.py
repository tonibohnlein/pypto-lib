# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Isolated real-shape driver for the production Gumbel/greedy argmax stage."""

import pypto.language as pl

from sample import (
    FP32_NEG_INF,
    SAMPLE_ROWS,
    SAMPLED_IDS_PAD,
    SAMPLING_EPS,
    VOCAB,
    _gumbel_noise,
    gumbel_sample,
)


@pl.jit
def gumbel_argmax_test(
    filtered_logits: pl.Tensor[[SAMPLE_ROWS, VOCAB], pl.FP32],
    temperatures: pl.Tensor[[SAMPLE_ROWS], pl.FP32],
    seeds: pl.Tensor[[SAMPLE_ROWS], pl.INT32],
    positions: pl.Tensor[[SAMPLE_ROWS], pl.INT32],
    sampled_ids: pl.Out[pl.Tensor[[SAMPLE_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
):
    return gumbel_sample(filtered_logits, temperatures, seeds, positions, sampled_ids)


def golden_gumbel_argmax(tensors):
    import torch

    tensors["sampled_ids"].zero_()
    for row in range(SAMPLE_ROWS):
        scores = tensors["filtered_logits"][row].float()
        if float(tensors["temperatures"][row]) >= SAMPLING_EPS:
            seed = int(tensors["seeds"][row])
            position = int(tensors["positions"][row])
            scores = scores + torch.from_numpy(_gumbel_noise(seed, position))
        tensors["sampled_ids"][row, 0] = torch.argmax(scores).to(torch.int32)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def repeat_rows(values, dtype):
        repeats = (SAMPLE_ROWS + len(values) - 1) // len(values)
        return torch.tensor(values, dtype=dtype).repeat(repeats)[:SAMPLE_ROWS]

    def init_filtered_logits():
        generator = torch.Generator().manual_seed(20260828)
        logits = torch.randn(SAMPLE_ROWS, VOCAB, generator=generator, dtype=torch.float32)
        logits[:, VOCAB // 2 :] = FP32_NEG_INF
        for row in range(SAMPLE_ROWS):
            logits[row, 17 + row] = 12.0
        return logits

    return [
        TensorSpec(
            "filtered_logits",
            [SAMPLE_ROWS, VOCAB],
            torch.float32,
            init_value=init_filtered_logits,
        ),
        TensorSpec(
            "temperatures",
            [SAMPLE_ROWS],
            torch.float32,
            init_value=lambda: repeat_rows([0.0, 0.3, 0.7, 1.0], torch.float32),
        ),
        TensorSpec(
            "seeds",
            [SAMPLE_ROWS],
            torch.int32,
            init_value=lambda: repeat_rows([1, 7, 19, 1234], torch.int32),
        ),
        TensorSpec(
            "positions",
            [SAMPLE_ROWS],
            torch.int32,
            init_value=lambda: repeat_rows([0, 1, 17, 1024], torch.int32),
        ),
        TensorSpec("sampled_ids", [SAMPLE_ROWS, SAMPLED_IDS_PAD], torch.int32, is_output=True),
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
        fn=gumbel_argmax_test,
        specs=build_tensor_specs(),
        golden_fn=golden_gumbel_argmax,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=0,
        atol=0,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
