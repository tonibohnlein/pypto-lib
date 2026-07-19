/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <cstdint>

#include "tensor.h"

#ifdef __CPU_SIM
#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) { (void)args; }

#else

#include "../paged_attention_cce/kernel/fai_body.hpp"

namespace {

constexpr uint32_t kRows = 16;
constexpr uint32_t kColumns = 512;
constexpr uint32_t kDepth = 128;

using namespace NpuArch;

using ElementQ = bfloat16_t;
using ElementK = bfloat16_t;
using ElementS = float;
using LayoutQ = layout::RowMajor;
using LayoutK = layout::ColumnMajor;
using LayoutS = layout::RowMajor;
using L1TileShapeQK = GemmShape<KernelCommon::Q_TILE_CEIL, 128, 128>;
using L0TileShapeQK = GemmShape<128, 128, 128>;
using DispatchPolicyQK = Gemm::MmadAtlasA2FAIQK<false, false>;
using QType = Gemm::GemmType<ElementQ, LayoutQ>;
using KType = Gemm::GemmType<ElementK, LayoutK>;
using SType = Gemm::GemmType<ElementS, LayoutS>;
using BlockMmadQK =
    Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, QType, KType, SType>;

template <typename T>
__aicore__ __attribute__((always_inline)) __gm__ T* tensor_ptr(__gm__ int64_t* args, int32_t index) {
  return reinterpret_cast<__gm__ T*>(tensor_data<T>(args, index));
}

}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#ifdef __DAV_C220_CUBE__
  uint32_t task = static_cast<uint32_t>(get_block_idx(args));

  AscendC::GlobalTensor<ElementQ> query;
  query.SetGlobalBuffer(tensor_ptr<ElementQ>(args, 0) + task * kRows * kDepth);
  AscendC::GlobalTensor<ElementK> key;
  key.SetGlobalBuffer(tensor_ptr<ElementK>(args, 1) + task * kColumns * kDepth);
  AscendC::GlobalTensor<ElementS> scores;
  scores.SetGlobalBuffer(tensor_ptr<ElementS>(args, 2) + task * kRows * kColumns);
  AscendC::GlobalTensor<int32_t> unused_block_table;

  Arch::Resource<Arch::AtlasA2> resource;
  AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
  AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
  AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
  AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
  AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
  AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
  AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
  AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
  AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
  AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
  AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
  AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
  AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
  AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
  AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
  AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
  AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
  AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);

  BlockMmadQK matmul;
  matmul.init(resource, 128, 128, kColumns);
  LayoutQ layout_q(kRows, kDepth);
  LayoutK layout_k(kDepth, kColumns, kDepth);
  LayoutS layout_s(kRows, kColumns);
  GemmCoord actual_shape{kRows, kColumns, kDepth};
  uint32_t group_heads = kRows;
  uint32_t query_heads = kRows;
  matmul.loadQGM(query, layout_q, kRows, group_heads, query_heads);
  matmul(query, key, scores, unused_block_table, layout_q, layout_k, layout_s, actual_shape, 0, 1, 128,
         kDepth);
#else
  (void)args;
#endif
}

#endif
