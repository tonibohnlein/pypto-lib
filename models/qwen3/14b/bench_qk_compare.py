# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""Compare equivalent Qwen3-14B decode QK task decompositions for pypto#2079."""

import argparse
import json
import os
import statistics
from pathlib import Path

import pypto.language as pl
import torch

from config import QWEN3_14B, QWEN3_14B_TILING
from golden import TensorSpec, run_jit


BATCH = QWEN3_14B.batch
NUM_KV_HEADS = QWEN3_14B.num_kv_heads
Q_ROWS = QWEN3_14B.q_head_pad
HEAD_DIM = QWEN3_14B.head_dim
PAGE_SIZE = QWEN3_14B_TILING.block_size
CONTEXT_SIZE = 4 * PAGE_SIZE
WORK_ITEMS = BATCH * NUM_KV_HEADS


@pl.jit
def qk_stack(
    query: pl.Tensor[[WORK_ITEMS, Q_ROWS, HEAD_DIM], pl.BF16],
    key: pl.Tensor[[WORK_ITEMS, CONTEXT_SIZE, HEAD_DIM], pl.BF16],
    out: pl.Out[pl.Tensor[[WORK_ITEMS, Q_ROWS, CONTEXT_SIZE], pl.FP32]],
) -> pl.Tensor[[WORK_ITEMS, Q_ROWS, CONTEXT_SIZE], pl.FP32]:
    """Dispatch one N=512 QK task per batch/KV-head work item."""
    for work_item in pl.parallel(WORK_ITEMS):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_stack"):
            query_3d = pl.slice(query, [1, Q_ROWS, HEAD_DIM], [work_item, 0, 0])
            query_tile = pl.reshape(query_3d, [Q_ROWS, HEAD_DIM])
            key_3d = pl.slice(key, [1, CONTEXT_SIZE, HEAD_DIM], [work_item, 0, 0])
            key_tile = pl.reshape(key_3d, [CONTEXT_SIZE, HEAD_DIM])
            scores = pl.matmul(query_tile, key_tile, b_trans=True, out_dtype=pl.FP32)
            scores_3d = pl.reshape(scores, [1, Q_ROWS, CONTEXT_SIZE])
            out = pl.assemble(out, scores_3d, [work_item, 0, 0])
    return out


@pl.jit
def qk_pages_parallel(
    query: pl.Tensor[[WORK_ITEMS, Q_ROWS, HEAD_DIM], pl.BF16],
    key: pl.Tensor[[WORK_ITEMS, CONTEXT_SIZE, HEAD_DIM], pl.BF16],
    out: pl.Out[pl.Tensor[[WORK_ITEMS, Q_ROWS, CONTEXT_SIZE], pl.FP32]],
) -> pl.Tensor[[WORK_ITEMS, Q_ROWS, CONTEXT_SIZE], pl.FP32]:
    """Dispatch four independent N=128 page tasks per work item."""
    for task in pl.parallel(WORK_ITEMS * (CONTEXT_SIZE // PAGE_SIZE)):
        work_item = task // (CONTEXT_SIZE // PAGE_SIZE)
        page = task % (CONTEXT_SIZE // PAGE_SIZE)
        page_offset = page * PAGE_SIZE
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_page_parallel"):
            query_3d = pl.slice(query, [1, Q_ROWS, HEAD_DIM], [work_item, 0, 0])
            query_tile = pl.reshape(query_3d, [Q_ROWS, HEAD_DIM])
            key_3d = pl.slice(key, [1, PAGE_SIZE, HEAD_DIM], [work_item, page_offset, 0])
            key_tile = pl.reshape(key_3d, [PAGE_SIZE, HEAD_DIM])
            scores = pl.matmul(query_tile, key_tile, b_trans=True, out_dtype=pl.FP32)
            scores_3d = pl.reshape(scores, [1, Q_ROWS, PAGE_SIZE])
            out = pl.assemble(out, scores_3d, [work_item, 0, page_offset])
    return out


@pl.jit
def qk_pages_serial(
    query: pl.Tensor[[WORK_ITEMS, Q_ROWS, HEAD_DIM], pl.BF16],
    key: pl.Tensor[[WORK_ITEMS, CONTEXT_SIZE, HEAD_DIM], pl.BF16],
    out: pl.Out[pl.Tensor[[WORK_ITEMS, Q_ROWS, CONTEXT_SIZE], pl.FP32]],
) -> pl.Tensor[[WORK_ITEMS, Q_ROWS, CONTEXT_SIZE], pl.FP32]:
    """Dispatch one task per work item with four serial N=128 matmuls."""
    for work_item in pl.parallel(WORK_ITEMS):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_pages_serial"):
            query_3d = pl.slice(query, [1, Q_ROWS, HEAD_DIM], [work_item, 0, 0])
            query_tile = pl.reshape(query_3d, [Q_ROWS, HEAD_DIM])
            for page in pl.range(CONTEXT_SIZE // PAGE_SIZE):
                page_offset = page * PAGE_SIZE
                key_3d = pl.slice(key, [1, PAGE_SIZE, HEAD_DIM], [work_item, page_offset, 0])
                key_tile = pl.reshape(key_3d, [PAGE_SIZE, HEAD_DIM])
                scores = pl.matmul(query_tile, key_tile, b_trans=True, out_dtype=pl.FP32)
                scores_3d = pl.reshape(scores, [1, Q_ROWS, PAGE_SIZE])
                out = pl.assemble(out, scores_3d, [work_item, 0, page_offset])
    return out


VARIANTS = {
    "stack": qk_stack,
    "pages_parallel": qk_pages_parallel,
    "pages_serial": qk_pages_serial,
}

EXPECTED_TASKS = {
    "stack": WORK_ITEMS,
    "pages_parallel": WORK_ITEMS * (CONTEXT_SIZE // PAGE_SIZE),
    "pages_serial": WORK_ITEMS,
}


def build_specs() -> list[TensorSpec]:
    return [
        TensorSpec(
            "query",
            [WORK_ITEMS, Q_ROWS, HEAD_DIM],
            torch.bfloat16,
            init_value=lambda: torch.randn(WORK_ITEMS, Q_ROWS, HEAD_DIM),
        ),
        TensorSpec(
            "key",
            [WORK_ITEMS, CONTEXT_SIZE, HEAD_DIM],
            torch.bfloat16,
            init_value=lambda: torch.randn(WORK_ITEMS, CONTEXT_SIZE, HEAD_DIM),
        ),
        TensorSpec(
            "out",
            [WORK_ITEMS, Q_ROWS, CONTEXT_SIZE],
            torch.float32,
            is_output=True,
        ),
    ]


def golden_qk(values: dict[str, torch.Tensor]) -> None:
    values["out"][:] = torch.matmul(values["query"].float(), values["key"].float().transpose(1, 2))


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _max_concurrency(tasks: list[dict]) -> int:
    events = []
    for task in tasks:
        events.append((float(task["start_time_us"]), 1))
        events.append((float(task["end_time_us"]), -1))
    active = 0
    maximum = 0
    for _, delta in sorted(events, key=lambda event: (event[0], event[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def _value_stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "median": statistics.median(values),
        "p10": _percentile(values, 0.10),
        "p90": _percentile(values, 0.90),
        "minimum": min(values),
        "maximum": max(values),
    }


def _duration_stats(tasks: list[dict]) -> dict[str, float] | None:
    return _value_stats([float(task["duration_us"]) for task in tasks])


def _summarize_l2(work_dir: Path, variant: str, force_l0_tile: str | None) -> None:
    from simpler_setup.tools.swimlane_converter import read_perf_data

    dfx_dir = work_dir / "dfx_outputs"
    records = dfx_dir / "l2_swimlane_records.json"
    if not records.is_file():
        records = dfx_dir / "l2_perf_records.json"
    if not records.is_file():
        raise RuntimeError(f"L2 timing records not found under {dfx_dir}")

    tasks = read_perf_data(str(records)).get("tasks", [])
    aic_tasks = [task for task in tasks if task.get("core_type") == "aic"]
    aiv_tasks = [task for task in tasks if task.get("core_type") == "aiv"]
    if not aic_tasks:
        raise RuntimeError(f"no AIC task records found in {records}")

    start = min(float(task["start_time_us"]) for task in tasks)
    end = max(float(task["end_time_us"]) for task in tasks)
    span = end - start
    summary = {
        "variant": variant,
        "force_l0_tile": force_l0_tile,
        "expected_tasks": EXPECTED_TASKS[variant],
        "recorded_aic_tasks": len(aic_tasks),
        "recorded_aiv_tasks": len(aiv_tasks),
        "aic_duration_us": _duration_stats(aic_tasks),
        "aiv_duration_us": _duration_stats(aiv_tasks),
        "task_span_us": span,
        "aic_busy_us": sum(float(task["duration_us"]) for task in aic_tasks),
        "aiv_busy_us": sum(float(task["duration_us"]) for task in aiv_tasks),
        "average_aic_concurrency": (
            sum(float(task["duration_us"]) for task in aic_tasks) / span if span > 0.0 else 0.0
        ),
        "average_aiv_concurrency": (
            sum(float(task["duration_us"]) for task in aiv_tasks) / span if span > 0.0 else 0.0
        ),
        "maximum_aic_concurrency": _max_concurrency(aic_tasks),
        "maximum_aiv_concurrency": _max_concurrency(aiv_tasks),
        "aic_core_count": len({int(task["core_id"]) for task in aic_tasks}),
        "aiv_core_count": len({int(task["core_id"]) for task in aiv_tasks}),
        "records": str(records),
    }
    summary_path = dfx_dir / "qk_task_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[QK_TASKS] {json.dumps(summary, sort_keys=True)}")
    if len(aic_tasks) != EXPECTED_TASKS[variant]:
        print(f"[QK_TASKS] warning: expected {EXPECTED_TASKS[variant]} AIC tasks, recorded {len(aic_tasks)}")


def _report_benchmark(result, variant: str, force_l0_tile: str | None) -> None:
    if result.bench is None:
        return
    effective = [value for value in result.bench.per_round("effective") if value > 0.0]
    device = [value for value in result.bench.per_round("device") if value > 0.0]
    report = {
        "variant": variant,
        "force_l0_tile": force_l0_tile,
        "effective_us": _value_stats(effective),
        "device_wall_us": _value_stats(device),
        "rounds": len(effective),
    }
    if result.work_dir is not None:
        report["effective_us_samples"] = effective
        report["device_wall_us_samples"] = device
        summary_path = result.work_dir / "qk_benchmark_summary.json"
        summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        report["summary_path"] = str(summary_path)
    report.pop("effective_us_samples", None)
    report.pop("device_wall_us_samples", None)
    print(f"[QK_BENCH] {json.dumps(report, sort_keys=True)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, default="stack")
    parser.add_argument("-p", "--platform", choices=["a2a3", "a2a3sim"], default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--enable-l2-swimlane", action="store_true")
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-l0-tile")
    args = parser.parse_args()

    if args.force_l0_tile:
        os.environ["PYPTO_FORCE_L0_TILE"] = args.force_l0_tile
    else:
        os.environ.pop("PYPTO_FORCE_L0_TILE", None)

    torch.manual_seed(2079)
    config_summary = {
        "variant": args.variant,
        "force_l0_tile": args.force_l0_tile,
        "work_items": WORK_ITEMS,
        "q_rows": Q_ROWS,
        "context_size": CONTEXT_SIZE,
        "head_dim": HEAD_DIM,
        "page_size": PAGE_SIZE,
        "expected_tasks": EXPECTED_TASKS[args.variant],
    }
    print(f"[QK_CONFIG] {json.dumps(config_summary, sort_keys=True)}")
    result = run_jit(
        fn=VARIANTS[args.variant],
        specs=build_specs(),
        golden_fn=golden_qk if args.check else None,
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
        if args.enable_l2_swimlane and not args.compile_only and not args.platform.endswith("sim"):
            _summarize_l2(result.work_dir, args.variant, args.force_l0_tile)
    _report_benchmark(result, args.variant, args.force_l0_tile)


if __name__ == "__main__":
    main()
