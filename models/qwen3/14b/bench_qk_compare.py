# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""Reconstruct the rank-2 Qwen3-14B decode QK comparison from pypto#2079.

The issue's original ``bench_qk_compare.py`` was not committed at its stated
pypto-lib revision.  This reconstruction follows the published description:

* ``stack`` dispatches 128 work items, each computing one ``[16,512]`` QK tile;
* ``pages`` dispatches 512 work items, each computing one independent
  ``[16,128]`` page tile;
* both paths use direct rank-2 ``pl.load(..., target_memory=Mat)`` operations,
  so tensor slice/reshape lowering is deliberately absent.

The committed CCE bridge exposes only the complete fused-attention operation,
not its private ``BlockMmadQK`` template.  The ``cce`` implementation wraps
that vendored template in a QK-only AIC entry point so it can be compared on
equal arithmetic.  This wrapper is a reconstruction, not recovered source
from the issue author.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch

import pypto.language as pl
from config import QWEN3_14B, QWEN3_14B_TILING
from golden import TensorSpec, run_jit
from paged_attention_cce import _CANN_INCLUDE_DIRS as CANN_INCLUDE_DIRS


BATCH = QWEN3_14B.batch
NUM_KV_HEADS = QWEN3_14B.num_kv_heads
Q_ROWS = QWEN3_14B.q_head_pad
HEAD_DIM = QWEN3_14B.head_dim
PAGE_SIZE = QWEN3_14B_TILING.block_size
PAGES = 4
CONTEXT_SIZE = PAGES * PAGE_SIZE
WORK_ITEMS = BATCH * NUM_KV_HEADS
PAGE_TASKS = WORK_ITEMS * PAGES
_QK_CCE_ENTRY = Path(__file__).parent / "kernels" / "qk_compare_cce" / "entry.cpp"


@pl.jit
def qk_stack(
    query: pl.Tensor[[WORK_ITEMS * Q_ROWS, HEAD_DIM], pl.BF16],
    key: pl.Tensor[[WORK_ITEMS * CONTEXT_SIZE, HEAD_DIM], pl.BF16],
    out: pl.Out[pl.Tensor[[WORK_ITEMS * Q_ROWS, CONTEXT_SIZE], pl.FP32]],
) -> pl.Tensor[[WORK_ITEMS * Q_ROWS, CONTEXT_SIZE], pl.FP32]:
    """Dispatch one direct rank-2 ``N=512`` QK task per work item."""
    with pl.spmd(WORK_ITEMS, name_hint="qk_stack"):
        work_item = pl.tile.get_block_idx()
        query_offset = work_item * Q_ROWS
        key_offset = work_item * CONTEXT_SIZE
        query_tile = pl.load(
            query,
            [query_offset, 0],
            [Q_ROWS, HEAD_DIM],
            target_memory=pl.MemorySpace.Mat,
        )
        key_tile = pl.load(
            key,
            [key_offset, 0],
            [CONTEXT_SIZE, HEAD_DIM],
            target_memory=pl.MemorySpace.Mat,
        )
        scores = pl.matmul(
            query_tile,
            pl.tile.transpose_view(key_tile),
            out_dtype=pl.FP32,
        )
        out = pl.store(scores, [query_offset, 0], out)
    return out


@pl.jit
def qk_pages(
    query: pl.Tensor[[WORK_ITEMS * Q_ROWS, HEAD_DIM], pl.BF16],
    key: pl.Tensor[[WORK_ITEMS * CONTEXT_SIZE, HEAD_DIM], pl.BF16],
    out: pl.Out[pl.Tensor[[PAGE_TASKS * Q_ROWS, PAGE_SIZE], pl.FP32]],
) -> pl.Tensor[[PAGE_TASKS * Q_ROWS, PAGE_SIZE], pl.FP32]:
    """Dispatch four independent direct rank-2 ``N=128`` tasks per work item."""
    with pl.spmd(PAGE_TASKS, name_hint="qk_page"):
        task = pl.tile.get_block_idx()
        work_item = task // PAGES
        query_offset = work_item * Q_ROWS
        key_offset = task * PAGE_SIZE
        out_offset = task * Q_ROWS
        query_tile = pl.load(
            query,
            [query_offset, 0],
            [Q_ROWS, HEAD_DIM],
            target_memory=pl.MemorySpace.Mat,
        )
        key_tile = pl.load(
            key,
            [key_offset, 0],
            [PAGE_SIZE, HEAD_DIM],
            target_memory=pl.MemorySpace.Mat,
        )
        scores = pl.matmul(
            query_tile,
            pl.tile.transpose_view(key_tile),
            out_dtype=pl.FP32,
        )
        out = pl.store(scores, [out_offset, 0], out)
    return out


@pl.jit.extern(
    core_type="aic",
    source=_QK_CCE_ENTRY,
    include_dirs=CANN_INCLUDE_DIRS,
)
def qk_stack_cce(
    query: pl.Tensor[[WORK_ITEMS * Q_ROWS, HEAD_DIM], pl.BF16],
    key: pl.Tensor[[WORK_ITEMS * CONTEXT_SIZE, HEAD_DIM], pl.BF16],
    out: pl.Out[pl.Tensor[[WORK_ITEMS * Q_ROWS, CONTEXT_SIZE], pl.FP32]],
) -> pl.Tensor[[WORK_ITEMS * Q_ROWS, CONTEXT_SIZE], pl.FP32]: ...


@pl.jit
def qk_cce(
    query: pl.Tensor[[WORK_ITEMS * Q_ROWS, HEAD_DIM], pl.BF16],
    key: pl.Tensor[[WORK_ITEMS * CONTEXT_SIZE, HEAD_DIM], pl.BF16],
    out: pl.Out[pl.Tensor[[WORK_ITEMS * Q_ROWS, CONTEXT_SIZE], pl.FP32]],
) -> pl.Tensor[[WORK_ITEMS * Q_ROWS, CONTEXT_SIZE], pl.FP32]:
    """Dispatch the reconstructed QK-only CCE stack kernel."""
    with pl.spmd(WORK_ITEMS, name_hint="qk_cce"):
        out = qk_stack_cce(query, key, out)
    return out


PYPTO_PROGRAMS = {
    "stack": qk_stack,
    "pages": qk_pages,
}


def build_specs(decomposition: str) -> list[TensorSpec]:
    """Build equal-arithmetic flat rank-2 inputs and decomposition-specific output."""
    out_shape = (
        [WORK_ITEMS * Q_ROWS, CONTEXT_SIZE] if decomposition == "stack" else [PAGE_TASKS * Q_ROWS, PAGE_SIZE]
    )
    return [
        TensorSpec(
            "query",
            [WORK_ITEMS * Q_ROWS, HEAD_DIM],
            torch.bfloat16,
            init_value=lambda: torch.randn(WORK_ITEMS * Q_ROWS, HEAD_DIM),
        ),
        TensorSpec(
            "key",
            [WORK_ITEMS * CONTEXT_SIZE, HEAD_DIM],
            torch.bfloat16,
            init_value=lambda: torch.randn(WORK_ITEMS * CONTEXT_SIZE, HEAD_DIM),
        ),
        TensorSpec("out", out_shape, torch.float32, is_output=True),
    ]


def golden_qk(values: dict[str, torch.Tensor], decomposition: str) -> None:
    """Compute the stack or page output from the same logical query/key tensors."""
    query = values["query"].reshape(WORK_ITEMS, Q_ROWS, HEAD_DIM).float()
    key = values["key"].reshape(WORK_ITEMS, PAGES, PAGE_SIZE, HEAD_DIM).float()
    pages = torch.matmul(query[:, None], key.transpose(2, 3))
    if decomposition == "stack":
        values["out"][:] = pages.permute(0, 2, 1, 3).reshape(
            WORK_ITEMS * Q_ROWS,
            CONTEXT_SIZE,
        )
    else:
        values["out"][:] = pages.reshape(PAGE_TASKS * Q_ROWS, PAGE_SIZE)


def _value_stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _report_benchmark(result, impl: str, decomposition: str) -> None:
    if result.bench is None:
        return
    effective = [value for value in result.bench.per_round("effective") if value > 0.0]
    device = [value for value in result.bench.per_round("device") if value > 0.0]
    report = {
        "impl": impl,
        "decomposition": decomposition,
        "effective_us": _value_stats(effective),
        "device_wall_us": _value_stats(device),
        "effective_us_samples": effective,
        "device_wall_us_samples": device,
    }
    if result.work_dir is not None:
        summary_path = Path(result.work_dir) / "qk_benchmark_summary.json"
        summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        report["summary_path"] = str(summary_path)
    report.pop("effective_us_samples")
    report.pop("device_wall_us_samples")
    print(f"[QK_BENCH] {json.dumps(report, sort_keys=True)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["pypto", "cce"], default="pypto")
    parser.add_argument("--decomposition", choices=PYPTO_PROGRAMS, default="stack")
    parser.add_argument("-p", "--platform", choices=["a2a3", "a2a3sim"], default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--dump-passes", action="store_true")
    parser.add_argument("--enable-l2-swimlane", action="store_true")
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.impl == "cce" and args.decomposition != "stack":
        parser.error("the reconstructed CCE comparator supports only the stack decomposition")

    expected_tasks = WORK_ITEMS if args.decomposition == "stack" else PAGE_TASKS
    print(
        "[QK_CONFIG] "
        + json.dumps(
            {
                "impl": args.impl,
                "decomposition": args.decomposition,
                "expected_tasks": expected_tasks,
                "work_items": WORK_ITEMS,
                "pages": PAGES,
                "m": Q_ROWS,
                "n": CONTEXT_SIZE if args.decomposition == "stack" else PAGE_SIZE,
                "k": HEAD_DIM,
            },
            sort_keys=True,
        )
    )

    torch.manual_seed(2079)
    program = qk_cce if args.impl == "cce" else PYPTO_PROGRAMS[args.decomposition]
    result = run_jit(
        fn=program,
        specs=build_specs(args.decomposition),
        golden_fn=((lambda values: golden_qk(values, args.decomposition)) if args.check else None),
        compile_cfg={"dump_passes": args.dump_passes},
        runtime_cfg={
            "platform": args.platform,
            "device_id": args.device,
            "enable_l2_swimlane": args.enable_l2_swimlane,
        },
        compile_only=args.compile_only or args.platform.endswith("sim"),
        rtol=2e-2,
        atol=2e-2,
        save_data=False,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
    if result.work_dir is not None:
        print(f"[QK_WORK_DIR] {result.work_dir}")
    _report_benchmark(result, args.impl, args.decomposition)


if __name__ == "__main__":
    main()
