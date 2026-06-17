#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <faiss/IndexIVFFlat.h>
#include <faiss/Index.h>

namespace faiss {

enum class MergeMethod {
    Concat,
    Merge,
};

/// IVF merge + optional IVFPQ fields (PQ keys ignored by IVFFlat merge).
struct MergeOptions {
    size_t target_nlist = 0;
    float merge_threshold = 0.03f;
    int neighbor_k = 512;
    int batch_size = 100000;
    int split_kmeans_niter = 5;
    int split_kmeans_nredo = 1;
    int random_state = 42;
    float sample_fraction = 0.05f;
    int sample_kmeans_niter = 5;
    int remap_neighbor_k = 256;

    // IVFPQ-only (ignored by IVFFlat merge)
    size_t target_M = 0;
    size_t target_nbits = 0;
    size_t pq_train_max_pts = 0;
    const float* training_vectors = nullptr;
    const idx_t* training_ids = nullptr;
    size_t n_training_vectors = 0;
    bool ivf_merge_use_raw = false;
    int pq_fast_add_k = 0;
    int pq_fast_add_l = 1;
    size_t pq_fast_add_min_count = 0;
    bool pq_fast_add_time_split = false;
    bool pq_fast_add_measure_subcode_recall = false;
    bool pq_fast_add_measure_subcode_recall_with_neighbors = false;
    int pq_fast_add_neighbor_kk = 32;
    bool ivfpq_smart_filtering = true;
    float ivfpq_smart_filter_threshold = 0.15f;
    bool ivfpq_adaptive_candidates = true;
    bool ivfpq_train_pq_on_raw_residuals = false;
    bool ivfpq_use_subcode_remap = false;
    float ivfpq_full_scan_margin_ratio = 0.0f;
    float ivfpq_full_scan_margin_abs = 0.0f;
    bool ivfpq_adaptive_k1_by_margin = false;
    float ivfpq_adaptive_k1_high_margin_ratio = 0.0f;
    float ivfpq_adaptive_k1_mid_margin_ratio = 0.0f;
    int ivfpq_adaptive_k1_min_neighbor_kk = 32;
    bool ivfpq_subspace_neighbor_schedule = false;
    float ivfpq_subspace_neighbor_high_fraction = 1.0f;
    float ivfpq_subspace_neighbor_mid_fraction = 0.0f;
    int ivfpq_subspace_neighbor_mid_kk = 48;
    int ivfpq_subspace_neighbor_low_kk = 40;
    bool ivfpq_cache_rough_k1 = false;
};

struct MergeRunStats {
    /// Time to assemble shard inverted lists into one logical index (no vector copy
    /// when using HStack; metadata scan only).
    double concat_s = 0.0;
    /// Time to materialize final ArrayInvertedLists after merge (0 for Concat-only).
    double build_index_s = 0.0;
    /// Sum of merge compute stages (excludes concat and build_index).
    double merge_total_s = 0.0;

    double optimize_s = 0.0;
    double merge_close_s = 0.0;
    double split_s = 0.0;
    /// Wall-clock time of merge_ivfflat() (excludes load/save).
    double total_s = 0.0;
    double remap_centroid_train_s = 0.0;
    double remap_snap_to_data_s = 0.0;
    double remap_neighbor_map_s = 0.0;
    double remap_full_reassign_s = 0.0;

    // IVFPQ timing (used by IndexIVFPQMerge)
    double extract_data_s = 0.0;
    double pq_codebook_selection_s = 0.0;
    double subcode_mapping_s = 0.0;
    double add_to_index_s = 0.0;
    double ivf_merge_s = 0.0;
    double ivfpq_pq_codebook_s = 0.0;
    double ivfpq_fcode_precompute_s = 0.0;
    double ivfpq_reencode_s = 0.0;
    double ivfpq_fast_add_rough_topk_s = 0.0;
    double ivfpq_fast_add_exact_l2_s = 0.0;
    double ivfpq_fast_add_neighbor_precompute_s = 0.0;
    double fast_add_num_tables = 0.0;
    double fast_add_table_hits = 0.0;
    double fast_add_full_encodes = 0.0;
    double fast_add_lazy_full_scans = 0.0;
    static constexpr int FASTADD_SUBCODE_RECALL_NUM_K = 7;
    uint64_t ivfpq_fast_add_subcode_recall_hits[FASTADD_SUBCODE_RECALL_NUM_K] = {};
    uint64_t ivfpq_fast_add_subcode_recall_pairs = 0;
    static constexpr int FASTADD_NEIGHBOR_RECALL_NUM_K = 4;
    uint64_t ivfpq_fast_add_subcode_neighbor_recall_hits[FASTADD_NEIGHBOR_RECALL_NUM_K] = {};
    uint64_t ivfpq_fast_add_subcode_neighbor_recall_union_size_sum[FASTADD_NEIGHBOR_RECALL_NUM_K] = {};
    uint64_t ivfpq_fast_add_subcode_neighbor_recall_pairs = 0;
};

struct MergeArguments {
    MergeMethod method = MergeMethod::Concat;
    MergeOptions merge;
    MergeRunStats* run_stats = nullptr;
};

/// Concatenate clusters from src into dst (nlist grows).
void ivfflat_concat_merge(IndexIVFFlat& dst, IndexIVFFlat& src, idx_t add_id);

std::unique_ptr<IndexIVFFlat> merge_ivfflat(
        const std::vector<IndexIVFFlat*>& indices,
        const MergeArguments& options);

std::unique_ptr<Index> shard_ivfflat(const std::vector<IndexIVFFlat*>& indices);

struct IVFDataForMerge {
    size_t d = 0;
    size_t nlist = 0;
    size_t ntotal = 0;
    MetricType metric = METRIC_L2;
    std::vector<float> centroids;
    std::vector<std::vector<idx_t>> lists;
    std::vector<float> vectors;
    const float* vectors_view = nullptr;
};

/// Merge on in-memory IVF data: dedup + adjust nlist, then snap-to-data K-means remap.
void merge_ivf_data(
        IVFDataForMerge& data,
        const MergeOptions& options,
        MergeRunStats* stats = nullptr);

/// Recompute merge_total_s from stage fields (total_s is set by merge_ivfflat).
void finalize_merge_run_stats(MergeRunStats* stats);

} // namespace faiss
