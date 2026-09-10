#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <faiss/IndexIVFFlat.h>
#include <faiss/Index.h>

namespace faiss {


using MergeRemapBatchCallback = void (*)(
        void* user_data,
        const idx_t* ids,
        const float* vectors,
        const idx_t* target_lists,
        const float* target_centroids,
        size_t n);

enum class MergeMethod {
    Concat,
    Merge,
};

/// IVF merge + optional IVFPQ fields (PQ keys ignored by IVFFlat merge).
struct MergeOptions {
    size_t target_nlist = 0;
    // Disabled default-path ablation parameters (implementation retained under #if 0):
    // float merge_threshold = 0.08f;
    // int neighbor_k = 512;
    int batch_size = 100000;
    // Split behavior is fixed at niter=5 and nredo=1.
    int split_max_k_per_cluster = 64;
    // Disabled default-path split ablations (implementation retained behind fixed false branches):
    // bool split_debug = false;
    // double split_quota_sse_alpha = 0.0;
    int random_state = 42;
    float sample_fraction = 0.05f;
    int sample_kmeans_niter = 5;
    // Required for Merge; zero is an invalid unset sentinel.
    int remap_neighbor_k = 0;
    // Disabled default-path centroid initialization ablation:
    // bool snap_centroids_to_data = false;
    // Force remap from the post-stage1 target lists. For target_nlist=3000,
    // this makes every IVFFlat merge use a 3000->3000 remap, including
    // small_nlist=1000 where the source has 10000 lists.
    bool force_current_lists_remap = false;
    bool use_split_centroids_final_exact_assign = false;
    bool stage1_sample_only = false;
    bool stage1_reduce_use_training_vectors = false;
    bool stage1_reduce_per_shard = false;
    size_t stage1_reduce_num_shards = 0;
    std::string stage1_reduce_weight_mode = "none";
    size_t stage1_reduce_weight_train_max = 50000;
    const float* reference_centroids = nullptr;
    size_t n_reference_centroids = 0;
    bool skip_ivf_merge_use_reference_centroids = false;
    bool use_ivf_merge_lists_direct_reencode = false;
    bool use_listwise_rabitq_reencode = false;
    bool use_direct_1bit_rabitq_reencode = false;
    bool return_final_assign_without_lists = false;
    bool use_source_list_order_rabitq_reencode = false;
    bool use_in_remap_rabitq_encode = false;
    bool reserve_in_remap_rabitq_buffers = false;
    bool use_old_state_rabitq_reencode = false;
    bool old_state_mirror_only = false;
    std::string old_state_t_init_mode = "norm_ratio";
    int old_state_local_t_steps = 0;
    float old_state_local_t_step = 1.0f / 128.0f;
    bool old_state_adaptive_t = false;
    float old_state_adaptive_t_factor = 1.0317434f;
    int old_state_adaptive_t_max_steps = 32;
    int old_state_adaptive_t_patience = 4;
    float old_state_adaptive_t_min_gain = 1e-5f;
    const float* old_state_t0_by_id = nullptr;
    size_t n_old_state_t0 = 0;
    std::string rabitq_t_diagnostic_path;
    size_t rabitq_t_diagnostic_max_vectors = 0;
    bool rabitq_t_diagnostic_only = false;
    const float* rabitq_distance_queries = nullptr;
    size_t n_rabitq_distance_queries = 0;
    MergeRemapBatchCallback remap_batch_callback = nullptr;
    void* remap_batch_callback_user_data = nullptr;
    std::string remap_candidate_diagnostic_path;
    size_t remap_candidate_diagnostic_max_vectors = 0;
    std::vector<int> remap_candidate_diagnostic_ks;
    // IVFPQ-only (ignored by IVFFlat merge)
    size_t target_M = 0;
    size_t target_nbits = 0;
    size_t pq_train_max_pts = 0;
    const float* training_vectors = nullptr;
    const idx_t* training_ids = nullptr;
    size_t n_training_vectors = 0;
    // The full dense database may be supplied for downstream re-encoding.
    // Keep stage-2 k-means on sample_fraction instead of treating all of it as
    // an explicit training override.
    bool sample_stage2_from_training_vectors = false;
    bool ivf_merge_use_raw = false;
    int pq_fast_add_neighbor_kk = 64;
    bool ivfpq_merge_aware_pq_hotstart = false;
    int ivfpq_merge_aware_pq_niter = 15;
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
    double remap_pack_s = 0.0;
    double remap_vector_copy_s = 0.0;
    double remap_knn_s = 0.0;
    double remap_assign_write_s = 0.0;

    // IVFPQ timing (used by IndexIVFPQMerge)
    double extract_data_s = 0.0;
    double pq_codebook_selection_s = 0.0;
    double subcode_mapping_s = 0.0;
    double add_to_index_s = 0.0;
    double ivf_merge_s = 0.0;
    double ivfpq_pq_codebook_s = 0.0;
    double ivfpq_fcode_precompute_s = 0.0;
    double ivfpq_reencode_s = 0.0;
    double ivfpq_reencode_index_init_s = 0.0;
    double ivfpq_reencode_id_map_s = 0.0;
    double ivfpq_reencode_output_alloc_s = 0.0;
    double ivfpq_reencode_encode_codes_s = 0.0;
    double ivfpq_reencode_add_entries_s = 0.0;
    double ivfpq_fast_add_neighbor_precompute_s = 0.0;
    double fast_add_num_tables = 0.0;
    double fast_add_full_encodes = 0.0;
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
    std::vector<uint8_t> old_codes;
    size_t old_code_size = 0;
    std::vector<float> old_centroids;
    std::vector<idx_t> old_list_for_id;
    std::vector<float> old_t0_by_id;
    std::vector<idx_t> final_assign;
};

/// Merge on in-memory IVF data: dedup + adjust nlist, then snap-to-data K-means remap.
void merge_ivf_data(
        IVFDataForMerge& data,
        const MergeOptions& options,
        MergeRunStats* stats = nullptr);

/// Recompute merge_total_s from stage fields (total_s is set by merge_ivfflat).
void finalize_merge_run_stats(MergeRunStats* stats);

} // namespace faiss
