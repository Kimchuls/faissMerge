#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <faiss/IndexIVFFlat.h>
#include <faiss/Index.h>

namespace faiss {


enum class MergeMethod {
    Concat,
    Merge,
};

/// Parameters shared by IVF merge implementations.
struct IVFMergeOptions {
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
    // force_current_lists_remap was removed; post-stage1 lists are always used for final remap.
    // bool use_split_centroids_final_exact_assign = false;
    // bool stage1_sample_only = false;
    // Stage 1 reduction is fixed to global list-size-weighted centroids.
    // bool stage1_reduce_use_training_vectors = false;
    // bool stage1_reduce_per_shard = false;
    // size_t stage1_reduce_num_shards = 0;
    // std::string stage1_reduce_weight_mode = "list_size";
    size_t stage1_reduce_weight_train_max = 50000;
    // Disabled non-default oracle centroid input:
    // const float* reference_centroids = nullptr;
    // size_t n_reference_centroids = 0;
    // RaBitQ oracle/reference-centroid bypass is disabled.
    // bool skip_ivf_merge_use_reference_centroids = false;
    // RaBitQ always re-encodes directly from IVF remap lists.
    // bool use_ivf_merge_lists_direct_reencode = false;
    // Disabled remap-candidate diagnostics.
    // std::string remap_candidate_diagnostic_path;
    // size_t remap_candidate_diagnostic_max_vectors = 0;
    // std::vector<int> remap_candidate_diagnostic_ks;
    // Optional explicit Stage 2 training set. When absent, Stage 2 samples
    // sample_fraction from the vectors held by IVFDataForMerge.
    const float* stage2_training_vectors = nullptr;
    size_t n_stage2_training_vectors = 0;
    // training_ids was used only by the disabled stage1_sample_only path.
    // const idx_t* training_ids = nullptr;
};

/// Parameters specific to the selected IVFRaBitQ re-encoding paths.
struct IVFRaBitQMergeOptions {
    size_t target_nbits = 0;
    const float* raw_vectors = nullptr;
    size_t n_raw_vectors = 0;
    float old_state_local_t_step = 1.0f / 128.0f;
    int old_state_local_t_probe_steps = 2;
    int old_state_local_t_directional_steps = 2;
    const float* old_state_t0_by_id = nullptr;
    size_t n_old_state_t0 = 0;
};

/// IVFPQ merge options are paused until the IVFPQ merge is finalized.
struct IVFPQMergeOptions {
    const float* raw_vectors = nullptr;
    size_t n_raw_vectors = 0;
    // size_t target_M = 0;
    // size_t target_nbits = 0;
    // size_t pq_train_max_pts = 256000;
    // int pq_fast_add_neighbor_kk = 64;
    // bool ivf_merge_use_raw = true;
    // bool ivfpq_merge_aware_pq_hotstart = false;
    // int ivfpq_merge_aware_pq_niter = 15;
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
    IVFMergeOptions merge;
    IVFRaBitQMergeOptions rabitq;
    IVFPQMergeOptions ivfpq;
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
    std::vector<float> old_centroids;
    std::vector<idx_t> old_list_for_id;
    std::vector<float> old_t0_by_id;
    std::vector<idx_t> final_assign;
};

/// Merge on in-memory IVF data: dedup + adjust nlist, then snap-to-data K-means remap.
void merge_ivf_data(
        IVFDataForMerge& data,
        const IVFMergeOptions& options,
        MergeRunStats* stats = nullptr);

/// Recompute merge_total_s from stage fields (total_s is set by merge_ivfflat).
void finalize_merge_run_stats(MergeRunStats* stats);

} // namespace faiss
