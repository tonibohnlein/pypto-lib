# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Isolated real-capacity prefill driver for the DSpark HC post-mix."""

import pypto.language as pl

from config import PREFILL_BATCH, PREFILL_SEQ
from hc_post import D, HC_MULT, build_tensor_specs as _build_specs, golden_hc_post_prefill, hc_post_prefill


T = PREFILL_BATCH * PREFILL_SEQ
ACTIVE_TOKENS = T - 3


@pl.jit
def hc_post_prefill_test(
    x: pl.Tensor[[T, D], pl.BF16],
    residual: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    post: pl.Tensor[[T, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T, HC_MULT * HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    return hc_post_prefill(x, residual, post, comb, y, num_tokens)


def build_tensor_specs():
    import torch
    from golden import ScalarSpec

    specs = _build_specs(PREFILL_BATCH, PREFILL_SEQ)
    for index, spec in enumerate(specs):
        initializer = spec.init_value
        if callable(initializer):
            seed = 20260828 + index

            def deterministic_init(initializer=initializer, seed=seed):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    return initializer()

            spec.init_value = deterministic_init
    return [*specs, ScalarSpec("num_tokens", torch.int32, ACTIVE_TOKENS)]


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--dump-passes", action="store_true")
    args = parser.parse_args()
    result = run_jit(
        fn=hc_post_prefill_test,
        specs=build_tensor_specs(),
        golden_fn=golden_hc_post_prefill,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=1e-3,
        atol=1e-3,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
