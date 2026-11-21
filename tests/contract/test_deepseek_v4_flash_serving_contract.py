# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import subprocess
import sys
from pathlib import Path

import pypto
import pytest

from models.deepseek_v4_flash_mtp import config
from models.deepseek_v4_flash_mtp.serving_contract import (
    DEEPSEEK_V4_FLASH_SERVING_CONTRACT as CONTRACT,
)


def test_deepseek_v4_flash_serving_capabilities() -> None:
    assert CONTRACT.schema_version == "1"
    assert CONTRACT.prefill_tile_tokens == 128
    assert CONTRACT.max_prefill_tokens_per_request == 8192
    assert CONTRACT.max_prefill_requests_per_partition == 1
    assert CONTRACT.requires_homogeneous_prefill_decode is True


@pytest.mark.parametrize(
    ("active_tokens", "expected"),
    [(1, 128), (128, 128), (129, 256), (8191, 8192), (8192, 8192)],
)
def test_deepseek_v4_flash_prefill_padding(
    active_tokens: int,
    expected: int,
) -> None:
    assert CONTRACT.padded_prefill_tokens(active_tokens) == expected


@pytest.mark.parametrize("active_tokens", [0, -1, 8193])
def test_deepseek_v4_flash_prefill_padding_rejects_invalid_extents(
    active_tokens: int,
) -> None:
    with pytest.raises(ValueError):
        CONTRACT.padded_prefill_tokens(active_tokens)


@pytest.mark.parametrize("active_tokens", [True, 128.5, "128"])
def test_deepseek_v4_flash_prefill_padding_rejects_non_integer_extents(
    active_tokens: object,
) -> None:
    with pytest.raises(TypeError, match="must be an int"):
        CONTRACT.padded_prefill_tokens(active_tokens)


def test_deepseek_v4_flash_contract_matches_kernel_prefill_shape() -> None:
    assert config.PREFILL_BATCH == CONTRACT.max_prefill_requests_per_partition
    assert config.PREFILL_SEQ == CONTRACT.prefill_tile_tokens


@pytest.mark.skipif(getattr(pypto, "__pypto_stub__", False), reason="decode_fwd.py imports PyPTO kernels")
@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--auto-qkv-input-layout", "wq_a", "--runtime-dir", "."], "requires a fresh compile"),
        (["--steady-state"], "requires --auto-qkv-input-layout"),
        (["--auto-qkv-input-layout", "wq_a", "--expected-weight-uses", "1000", "--steady-state"],
         "not allowed with argument"),
    ],
)
def test_decode_forward_rejects_invalid_layout_options(arguments: list[str], message: str) -> None:
    repo = Path(__file__).resolve().parents[2]
    driver = repo / "models" / "deepseek_v4_flash_mtp" / "decode_fwd.py"
    search_paths = [str(repo)]
    if os.environ.get("PYTHONPATH"):
        search_paths.append(os.environ["PYTHONPATH"])
    result = subprocess.run(
        [
            sys.executable, str(driver), "--ep", "2", "--tp", "2",
            *arguments,
        ],
        cwd=repo,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(search_paths)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stderr
    assert message in result.stderr
