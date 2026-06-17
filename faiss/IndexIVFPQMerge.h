/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <vector>

#include <faiss/IndexIVFFlatMerge.h>
#include <faiss/IndexIVFPQ.h>

namespace faiss {

/// Concatenate clusters from src into dst (nlist grows).
/// Preserves existing assignments by shifting src list ids by dst.nlist.
/// Vector ids from src are optionally shifted by add_id.
/// NOTE: Concat algorithm is a stub - implementation left empty.
void ivfpq_concat_merge(IndexIVFPQ& dst, IndexIVFPQ& src, idx_t add_id);

/// Merge multiple IVFPQ indices. Input indices are not modified.
/// - Concat: IndexShards (query each shard, merge top-k; each shard keeps own PQ codebook; successive_ids).
/// - Merge: IVFFlat merge (dedup + adjust nlist + sample K-means remap) + PQ re-encode.
/// Returns Index* (IndexShards for Concat, IndexIVFPQ for Merge).
std::unique_ptr<Index> merge_ivfpq(
        const std::vector<IndexIVFPQ*>& indices,
        const MergeArguments& options);

} // namespace faiss
