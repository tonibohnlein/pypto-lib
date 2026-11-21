# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Replay the production decode sampler source against real-vocabulary argmax."""

import ast
import struct
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.qwen3_14b.constants import QWEN3_14B, QWEN3_14B_TILING


_SOURCE = Path(__file__).resolve().parents[2] / "models" / "qwen3_14b" / "decode_fwd.py"
_MODEL = QWEN3_14B
_TILING = QWEN3_14B_TILING
_FP32_MIN = struct.unpack(">f", bytes.fromhex("ff7fffff"))[0]
_FP32_NEXT = struct.unpack(">f", bytes.fromhex("ff7ffffe"))[0]


def _fp32(value):
    return struct.unpack(">f", struct.pack(">f", value))[0]


@dataclass
class _Tile:
    data: list
    active: tuple


def _full(shape, dtype, value=0):
    rows, cols = shape
    return _Tile([[dtype(value)] * cols for _ in range(rows)], (rows, cols))


def _load(source, offsets, shape, *, target_memory):
    assert target_memory == "vec"
    row, col = offsets
    rows, cols = shape
    data = [values[col : col + cols] for values in source[row : row + rows]]
    assert len(data) == rows and all(len(values) == cols for values in data)
    return _Tile(data, (rows, cols))


def _assemble(destination, source, offsets):
    row, col = offsets
    rows, cols = source.active
    for i, values in enumerate(source.data[:rows]):
        destination.data[row + i][col : col + cols] = values[:cols]
    return destination


def _row_maximum(tile, scratch):
    rows, cols = tile.active
    assert scratch.active == (rows, cols)
    return _Tile([[max(values[:cols])] for values in tile.data[:rows]], (rows, 1))


def _point(tile, indices):
    row, col = indices
    rows, cols = tile.active
    assert 0 <= row < rows and 0 <= col < cols
    return tile.data[row], col


def _read(tile, indices):
    values, index = _point(tile, indices)
    return values[index]


def _write(tile, indices, value):
    values, index = _point(tile, indices)
    values[index] = value


def _store(tile, indices, tensor):
    _assemble(_Tile(tensor, (len(tensor), len(tensor[0]))), tile, indices)


def _sampler():
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"), filename=str(_SOURCE))
    sampler = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_greedy_sample_inline"
    )
    sampler.decorator_list = []
    sampler.returns = None
    for argument in sampler.args.args:
        argument.annotation = None
    constants = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("SAMPLE_")
    ]
    env = dict(
        VOCAB=_MODEL.vocab, REAL_VOCAB=_MODEL.real_vocab, SAMPLED_IDS_PAD=_MODEL.sampled_ids_pad, T=_TILING
    )
    env["pl"] = SimpleNamespace(
        FP32=_fp32,
        INT32=int,
        INDEX=int,
        MemorySpace=SimpleNamespace(Vec="vec"),
        Mem=SimpleNamespace(Vec="vec"),
        Level=SimpleNamespace(CORE_GROUP="core_group"),
        tensor=SimpleNamespace(dim=lambda tensor, axis: (len(tensor), len(tensor[0]))[axis]),
        parallel=range,
        range=range,
        at=lambda **_: nullcontext(),
        load=_load,
        store=_store,
        cast=lambda value, target_type: target_type(value),
        create_tile=lambda shape, dtype, target_memory: _full(shape, dtype),
        tile=SimpleNamespace(
            full=_full,
            assemble=_assemble,
            row_max=_row_maximum,
            read=_read,
            write=_write,
        ),
    )
    exec(compile(ast.Module(body=[*constants, sampler], type_ignores=[]), str(_SOURCE), "exec"), env)
    return env["_greedy_sample_inline"], env


@pytest.mark.parametrize(
    "base,peaks",
    [
        (-50.0, [(33, -2.0)]),
        (-50.0, [(_TILING.vocab_chunk - 1, -2.0), (_TILING.vocab_chunk, -2.0)]),
        (-50.0, [(_MODEL.real_vocab - 1, -2.0)]),
        (_FP32_MIN, [(43, _FP32_NEXT), (_MODEL.real_vocab - 1, _FP32_NEXT)]),
        (_FP32_MIN, []),
        (float("-inf"), []),
        (float("-inf"), [(_MODEL.real_vocab - 1, _FP32_MIN)]),
    ],
)
def test_decode_sampler_matches_first_argmax_of_real_vocabulary(base, peaks):
    sampler, env = _sampler()
    logits = [base] * _MODEL.real_vocab + [42.0] * (_MODEL.vocab - _MODEL.real_vocab)
    for token, score in peaks:
        logits[token] = score
    expected = max(range(_MODEL.real_vocab), key=logits.__getitem__)
    ids = [[-1] * _MODEL.sampled_ids_pad]
    assert sampler([logits], ids) == [[expected, *([0] * (_MODEL.sampled_ids_pad - 1))]]
    if peaks and base == _FP32_MIN:
        # The shorter literal rounds two ulps above the FP32 floor and
        # outranks every real score here.
        env["SAMPLE_MIN_SCORE"] = _fp32(-3.402823e38)
        assert sampler([logits], ids)[0][0] != expected
