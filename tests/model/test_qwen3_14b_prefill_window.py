# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Replay the prefill dispatch and causal width expressions without a device."""

import ast
import ctypes
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.qwen3_14b.constants import QWEN3_14B, QWEN3_14B_TILING


_SOURCE = Path(__file__).resolve().parents[2] / "models" / "qwen3_14b" / "prefill_fwd.py"
_TREE = ast.parse(_SOURCE.read_text(encoding="utf-8"), filename=str(_SOURCE))
_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1


def _function(name):
    return next(node for node in _TREE.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _assignments(body, names):
    selected = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in names
    ]
    assert len(selected) == len(names)
    return sorted(selected, key=lambda node: node.lineno)


def _compile(statements):
    # Execute only trusted scalar statements from the checked-in model.
    return compile(ast.Module(body=statements, type_ignores=[]), str(_SOURCE), "exec")


def _run(code, **values):
    exec(code, values)
    return values


def _value(name, expression):
    statement = ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=expression)
    return ast.fix_missing_locations(statement)


@pytest.mark.parametrize(
    "seq_len,chunk_len,group_p0,seq_tile,expected",
    [
        (127, 127, 0, 128, "general"),
        (128, 128, 0, 128, "full"),
        (129, 129, 0, 128, "full"),
        (129, 129, 128, 128, "general"),
        (129, 128, 0, 128, "general"),
        (256, 256, 128, 128, "general"),
        (128, 128, 0, 64, "general"),
        (128, 128, 0, 256, "full"),
        (256, 128, 0, 256, "general"),
        (0, 0, 0, 128, "inactive"),
        (4096, 0, 0, 128, "inactive"),
        (127, 128, 0, 128, "inactive"),
        (128, 256, 128, 128, "inactive"),
        (_INT32_MIN, _INT32_MAX, 0, 128, "inactive"),
        (_INT32_MAX, _INT32_MIN, 128, 128, "inactive"),
    ],
)
def test_prefill_dispatch_uses_checked_initial_window(seq_len, chunk_len, group_p0, seq_tile, expected):
    layer = _function("prefill_layer")
    fast = "_attention_phase_window_full_single_block"
    general = "_attention_phase_window"
    dispatch = next(
        node
        for node in ast.walk(layer)
        if isinstance(node, ast.If)
        and isinstance(node.body[0], ast.Assign)
        and isinstance(call := node.body[0].value, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == fast
    )
    assert any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == general
               for call in ast.walk(dispatch.orelse[0]))
    active = next(
        node for node in ast.walk(layer) if isinstance(node, ast.If) and node is not dispatch
        and dispatch in ast.walk(node)
    )
    scalar_names = {
        "chunk_start_wide", "chunk_bounds_valid", "remaining_tok", "chunk_len_b", "tok_blocks",
        "valid_tok", "finalize_tok", "window_offset", "first_pos",
    }
    values = _run(
        _compile([
            *_assignments(_TREE, {"TOK_TILE", "ATTN_TOK_GROUP", "FINALIZE_TOK_GROUP", "FULL_GROUP_FITS_BLOCK"}),
            *_assignments(layer, scalar_names),
            _value("active", active.test),
            _value("full", dispatch.test),
        ]),
        pl=SimpleNamespace(
            INT32=ctypes.c_int32, INT64=ctypes.c_int64, cast=lambda value, dtype: dtype(value).value,
            min=min, max=max,
        ),
        MAX_SEQ=QWEN3_14B.max_seq,
        MLP_M_TILE=128,
        SEQ_TILE=seq_tile,
        seq_len_b=seq_len,
        full_chunk_len_b=chunk_len,
        group_p0=group_p0,
        p0_idx=0,
        p0=0,
        final_ti0=0,
    )
    route = "inactive" if not values["active"] else "full" if values["full"] else "general"
    assert route == expected
    assert values["chunk_start_wide"] == seq_len - chunk_len


def test_prefill_widens_both_runtime_lengths_before_subtraction():
    subtraction = _assignments(_function("prefill_layer"), {"chunk_start_wide"})[0].value
    assert isinstance(subtraction.op, ast.Sub)
    assert [ast.unparse(operand.args[1]) for operand in (subtraction.left, subtraction.right)] == [
        "pl.INT64", "pl.INT64",
    ]


def test_full_initial_window_emits_all_causal_widths():
    kernel = _function("_attention_phase_window_full_single_block")
    constants = _run(
        _compile(_assignments(_TREE, {
            "TOK_TILE", "ATTN_TOK_GROUP", "ATTN_GI_GROUP", "FINALIZE_TOK_GROUP",
            "ATTN_PHASE_MICRO_GROUPS", "ATTN_GI_BLOCKS", "ATTN_PHASE_WORK_ITEMS",
        })),
        TOTAL_Q_GROUPS=QWEN3_14B.total_q_groups,
    )
    extent = next(
        call.args[2] for call in ast.walk(kernel)
        if isinstance(call, ast.Call) and ast.unparse(call.func) == "pl.set_validshape"
    )
    width = _compile(_assignments(kernel, {"micro_id", "attn_dt0", "rel_ti0", extent.id}))
    widths = {
        _run(width, work_id=work_id, dd0=dd0, **constants)[extent.id]
        for work_id in range(constants["ATTN_PHASE_WORK_ITEMS"])
        for dd0 in range(constants["ATTN_TOK_GROUP"])
    }
    assert widths == set(range(1, QWEN3_14B_TILING.seq_tile + 1))
