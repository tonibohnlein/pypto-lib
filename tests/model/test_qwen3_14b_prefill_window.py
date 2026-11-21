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


@pytest.mark.parametrize("seq_tile", [64, QWEN3_14B_TILING.seq_tile, 256])
def test_partial_window_issues_transport_only_for_nonempty_causal_pages(seq_tile):
    partial = _function("_attention_phase_window")
    width = _assignments(partial, {"s0", "valid_len"})
    guard = next(
        node
        for node in ast.walk(partial)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "valid_len"
    )
    code = _compile([*width, _value("active", guard.test)])
    for context in range(1, QWEN3_14B.max_seq + 1):
        pages = (context + seq_tile - 1) // seq_tile
        # Exercise the first, last and first empty page for every legal prefix.
        for page in {0, pages - 1, pages}:
            values = _run(code, pl=SimpleNamespace(min=min), SEQ_TILE=seq_tile, sb=page, ctx_len=context)
            assert values["active"] == (page < pages)
            if values["active"]:
                assert values["valid_len"] == min(seq_tile, context - page * seq_tile)
                assert 1 <= values["valid_len"] <= seq_tile
            else:
                assert values["valid_len"] <= 0


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


def _torch_padding_ops(torch, tile_rows):
    def slice_tile(source, shape, offsets, *, valid_shape=None):
        rows, cols = shape if valid_shape is None else valid_shape
        row, col = offsets
        result = source[row : row + rows, col : col + cols]
        assert result.shape == (rows, cols)
        return result

    def fillpad(source, *, pad_value):
        assert pad_value == 0
        return torch.nn.functional.pad(source, (0, 0, 0, tile_rows - source.shape[0]))

    def assemble(output, source, offset):
        row, col = offset
        rows, cols = source.shape
        output[row : row + rows, col : col + cols] = source
        return output

    return SimpleNamespace(
        FP32=torch.float32,
        BF16=torch.bfloat16,
        PadValue=SimpleNamespace(zero=0),
        slice=slice_tile,
        fillpad=fillpad,
        assemble=assemble,
        cast=lambda source, target_type: source.to(target_type),
        min=min,
        range=range,
        full=lambda shape, value, dtype: torch.full(shape, value, dtype=dtype),
        reshape=torch.reshape,
        mul=torch.mul,
        row_sum=lambda source: source.sum(dim=1, keepdim=True),
        add=torch.add,
        sub=torch.sub,
        col_expand_mul=torch.mul,
        sqrt=torch.sqrt,
        recip=torch.reciprocal,
    )


@pytest.mark.parametrize("valid_rows", [0, 3, 8])
def test_residual_body_preserves_projection_for_padded_rows(valid_rows):
    torch = pytest.importorskip("torch")
    rows, cols = 8, 16
    hidden = torch.arange(valid_rows * cols, dtype=torch.float32).reshape(valid_rows, cols).bfloat16()
    projection = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols) / 3
    body = next(
        node.body
        for node in ast.walk(_function("prefill_layer"))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "ob"
        and any(
            isinstance(stmt, ast.Assign) and ast.unparse(stmt.targets[0]) == "out_proj_chunk"
            for stmt in node.body
        )
    )
    output = torch.full((rows + 1, cols), -1.0)
    _run(
        _compile(body),
        pl=_torch_padding_ops(torch, rows),
        ob=0,
        OUT_RESID_CHUNK=cols,
        TOK_TILE=rows,
        hidden_states=hidden,
        token_p0=0,
        valid_tok=valid_rows,
        out_proj_tile=projection,
        resid1_tile=output,
    )
    expected = projection.clone()
    expected[:valid_rows] += hidden.float()
    torch.testing.assert_close(output[:rows], expected)
    assert torch.all(output[rows:] == -1)


@pytest.mark.parametrize("valid_rows", [0, 3, 8])
def test_reciprocal_rms_body_defines_written_tail_and_leaves_inactive_groups_untouched(valid_rows):
    torch = pytest.importorskip("torch")
    rows, chunk, blocks, eps = 8, 16, 2, 1e-6
    columns = blocks * chunk
    hidden = (
        torch.arange(valid_rows * columns, dtype=torch.float32).reshape(valid_rows, columns) / 17
    ).bfloat16()
    guard = next(
        node
        for node in ast.walk(_function("prefill_layer"))
        if isinstance(node, ast.If)
        and any(isinstance(stmt, ast.For) and ast.unparse(stmt.target) == "rb" for stmt in node.body)
        and any(
            isinstance(stmt, ast.Assign) and ast.unparse(stmt.targets[0]) == "inv_rms" for stmt in node.body
        )
    )
    output = torch.full((2 * rows, 1), -1.0)
    _run(
        _compile([guard]),
        pl=_torch_padding_ops(torch, rows),
        ti0=0,
        RMSNORM_TOK_GROUP=rows,
        K_CHUNK=chunk,
        HIDDEN_BLOCKS=blocks,
        hidden_states=hidden,
        token_p0=0,
        valid_tok=valid_rows,
        HIDDEN_INV=1 / columns,
        EPS=eps,
        input_inv_rms_tile=output,
    )
    expected = torch.full_like(output, -1)
    if valid_rows:
        expected[:valid_rows] = torch.rsqrt(hidden.float().square().mean(dim=1, keepdim=True) + eps)
        expected[valid_rows:rows] = eps**-0.5
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("valid_rows", [0, 3, 8])
def test_prefill_rope_rotation_reads_only_available_positions_and_updates_guarded_cache_rows(valid_rows):
    torch = pytest.importorskip("torch")
    rows, width, heads, padded_heads = 8, 16, 3, 4
    pos0 = 5
    projection = torch.arange(rows * width, dtype=torch.float32).reshape(rows, width) / 7
    query = torch.arange(rows * heads * width, dtype=torch.float32).reshape(rows, heads, width) / 11
    cos = torch.linspace(0.0, 1.0, (pos0 + valid_rows) * width).reshape(pos0 + valid_rows, width)
    sin = torch.sqrt(1 - cos.square())
    rope = _function("_rope_kv_cache_phase")
    key_rotation = _compile(
        _assignments(
            rope,
            {
                "cos_lo_active",
                "cos_hi_active",
                "sin_lo_active",
                "sin_hi_active",
                "cos_lo_g",
                "cos_hi_g",
                "sin_lo_g",
                "sin_hi_g",
                "k_lo_g",
                "k_hi_g",
                "k_rot_lo_g",
                "k_rot_hi_g",
            },
        )
    )
    observation = next(
        node
        for node in ast.walk(rope)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "dt < group_tok"
    )
    outputs = _compile(
        [
            ast.If(
                test=observation.test,
                body=[
                    stmt
                    for stmt in observation.body
                    if isinstance(stmt, ast.Assign)
                    and ast.unparse(stmt.targets[0])
                    not in {
                        "cache_slot",
                        "cache_slot_block",
                        "cache_slot_offset",
                        "cache_row",
                        "v_cache_bsnd",
                    }
                ],
                orelse=[],
                lineno=observation.lineno,
                col_offset=observation.col_offset,
            )
        ]
    )
    output = torch.full((rows, width), -1.0, dtype=torch.bfloat16)
    query_output = torch.full((rows * padded_heads, width), -1.0, dtype=torch.bfloat16)
    values = _run(
        key_rotation,
        pl=_torch_padding_ops(torch, rows),
        ROPE_T_GROUP=rows,
        HALF_DIM=width // 2,
        group_tok=valid_rows,
        rope_cos=cos,
        rope_sin=sin,
        pos0=pos0,
        k_normed=projection,
    )
    for dt in range(rows):
        _run(
            outputs,
            **values,
            dt=dt,
            ti0=0,
            HEAD_DIM=width,
            Q_HEAD_BATCH=heads,
            Q_HEAD_PAD=padded_heads,
            TOTAL_Q_GROUPS=1,
            ki=0,
            cache_row=rows - dt - 1,
            cache_col=0,
            k_cache_bsnd=output,
            all_q_padded_tile=query_output,
            q_normed=query.flatten(0, 1),
        )

    lo, hi = projection[:valid_rows].chunk(2, dim=1)
    cos_lo, cos_hi = cos[pos0:].chunk(2, dim=1)
    sin_lo, sin_hi = sin[pos0:].chunk(2, dim=1)
    rotated = torch.cat((lo * cos_lo - hi * sin_lo, hi * cos_hi + lo * sin_hi), dim=1)
    if valid_rows:
        torch.testing.assert_close(output[-valid_rows:], rotated.flip(0).bfloat16())
    assert torch.all(output[: rows - valid_rows] == -1)
    query_lo, query_hi = query[:valid_rows].chunk(2, dim=-1)
    rotated_query = torch.cat(
        (
            query_lo * cos_lo[:, None] - query_hi * sin_lo[:, None],
            query_hi * cos_hi[:, None] + query_lo * sin_hi[:, None],
        ),
        dim=-1,
    )
    query_output = query_output.reshape(rows, padded_heads, width)
    torch.testing.assert_close(query_output[:valid_rows, :heads], rotated_query.bfloat16())
    assert torch.all(query_output[:valid_rows, heads:] == 0)
    assert torch.all(query_output[valid_rows:] == -1)
