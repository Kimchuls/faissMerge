#pragma once

#include <faiss/IndexIVFFlatMerge.h>

namespace faiss {

using IVFMergeRemapBatchCallback = void (*)(
        void* user_data,
        const idx_t* ids,
        const float* vectors,
        const idx_t* target_lists,
        const float* target_centroids,
        size_t n);

void merge_ivf_data_with_remap_callback(
        IVFDataForMerge& data,
        const IVFMergeOptions& options,
        IVFMergeRemapBatchCallback callback,
        void* callback_user_data,
        MergeRunStats* stats = nullptr);

} // namespace faiss
