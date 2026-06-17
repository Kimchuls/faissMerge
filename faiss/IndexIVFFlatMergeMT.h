#pragma once

#include <faiss/IndexIVFFlatMerge.h>

#include <memory>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace faiss {
namespace ivfflat_merge_mt {

/// True when OpenMP is enabled and more than one thread is active.
inline bool use_mt_merge() {
#ifdef _OPENMP
    return omp_get_max_threads() > 1;
#else
    return false;
#endif
}

/// Cluster-parallel reassign uses static scheduling up to this many threads;
/// above it, dynamic scheduling reduces load imbalance across many cores.
inline int reassign_low_thread_max() {
    return 8;
}

/// Above this thread count, stage2 uses custom parallel Lloyd / snap paths.
inline int high_parallel_stage2_min_threads() {
    return 32;
}

std::unique_ptr<IndexIVFFlat> merge_ivfflat(
        const std::vector<IndexIVFFlat*>& indices,
        const MergeArguments& options);

void merge_ivf_data(
        IVFDataForMerge& data,
        const MergeOptions& options,
        MergeRunStats* stats = nullptr);

} // namespace ivfflat_merge_mt
} // namespace faiss
