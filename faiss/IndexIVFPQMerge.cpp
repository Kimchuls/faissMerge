/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexIVFPQMerge.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

// OpenMP for coarse-grained parallelism across shards/lists.
// Tests limit thread count via OMP_NUM_THREADS=1; pragmas are always emitted.
#ifdef _OPENMP
#include <omp.h>
#endif

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexShards.h>
#include <faiss/clone_index.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/utils/distances.h>

namespace faiss {

namespace {

// Fraction of total vectors used as PQ codebook training sample.
// Hardcoded to 5 %; can be promoted to Merge4Options::pq_sample_percent if
// callers need finer control (e.g. larger sample for small datasets, capped
// sample for very large ones via pq_train_max_pts).
static constexpr size_t PQ_SAMPLE_PERCENT = 5;

static bool have_full_raw_vectors(const MergeOptions& merge2, size_t ntotal) {
    return merge2.training_vectors != nullptr && merge2.n_training_vectors == ntotal;
}

static void ensure_ivfpq_compatible(const IndexIVFPQ& idx) {
    FAISS_THROW_IF_NOT(idx.is_trained);
    FAISS_THROW_IF_NOT(idx.quantizer);
    FAISS_THROW_IF_NOT(idx.quantizer->ntotal == idx.nlist);
    FAISS_THROW_IF_NOT_MSG(idx.by_residual, "IVFPQ merge requires by_residual==true");
    FAISS_THROW_IF_NOT_MSG(idx.direct_map.no(), "IVFPQ merge requires direct_map disabled");
    FAISS_THROW_IF_NOT_MSG(idx.metric_type == METRIC_L2, "IVFPQ merge supports METRIC_L2 only");
}

// ── SIMD helpers ─────────────────────────────────────────────────────────────
// All helpers work on K floats (K = 2^nbits, typically 256).
// When src and dst alias (in-place add/undo of the prefix table), AVX2
// load-then-store is safe because the load completes before the store.

#ifdef __AVX2__
/// out[b] = a[b] + bv[b]  (aliased src/dst allowed)
static inline void add_floats_K(const float* __restrict__ a, const float* __restrict__ bv, float* __restrict__ out, size_t K) {
    size_t i = 0;
    for (; i + 8 <= K; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(bv + i)));
    }
    for (; i < K; i++)
        out[i] = a[i] + bv[i];
}

static inline void add_inplace_floats_K(float* __restrict__ dst, const float* __restrict__ src, size_t K) {
    size_t i = 0;
    for (; i + 8 <= K; i += 8) {
        _mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(src + i)));
    }
    for (; i < K; i++)
        dst[i] += src[i];
}

static inline void sub_inplace_floats_K(float* __restrict__ dst, const float* __restrict__ src, size_t K) {
    size_t i = 0;
    for (; i + 8 <= K; i += 8) {
        _mm256_storeu_ps(dst + i, _mm256_sub_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(src + i)));
    }
    for (; i < K; i++)
        dst[i] -= src[i];
}

/// out[b] = a[b] + bv[b] + c[b]
static inline void add3_floats_K(const float* __restrict__ a, const float* __restrict__ bv, const float* __restrict__ c, float* __restrict__ out, size_t K) {
    size_t i = 0;
    for (; i + 8 <= K; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(bv + i)), _mm256_loadu_ps(c + i)));
    }
    for (; i < K; i++)
        out[i] = a[i] + bv[i] + c[i];
}
#else
static inline void add_floats_K(const float* __restrict__ a, const float* __restrict__ bv, float* __restrict__ out, size_t K) {
    size_t i = 0;
    for (; i + 8 <= K; i += 8) {
        out[i + 0] = a[i + 0] + bv[i + 0];
        out[i + 1] = a[i + 1] + bv[i + 1];
        out[i + 2] = a[i + 2] + bv[i + 2];
        out[i + 3] = a[i + 3] + bv[i + 3];
        out[i + 4] = a[i + 4] + bv[i + 4];
        out[i + 5] = a[i + 5] + bv[i + 5];
        out[i + 6] = a[i + 6] + bv[i + 6];
        out[i + 7] = a[i + 7] + bv[i + 7];
    }
    for (; i < K; i++)
        out[i] = a[i] + bv[i];
}
static inline void add_inplace_floats_K(float* __restrict__ dst, const float* __restrict__ src, size_t K) {
    size_t i = 0;
    for (; i + 8 <= K; i += 8) {
        dst[i + 0] += src[i + 0];
        dst[i + 1] += src[i + 1];
        dst[i + 2] += src[i + 2];
        dst[i + 3] += src[i + 3];
        dst[i + 4] += src[i + 4];
        dst[i + 5] += src[i + 5];
        dst[i + 6] += src[i + 6];
        dst[i + 7] += src[i + 7];
    }
    for (; i < K; i++)
        dst[i] += src[i];
}
static inline void sub_inplace_floats_K(float* __restrict__ dst, const float* __restrict__ src, size_t K) {
    size_t i = 0;
    for (; i + 8 <= K; i += 8) {
        dst[i + 0] -= src[i + 0];
        dst[i + 1] -= src[i + 1];
        dst[i + 2] -= src[i + 2];
        dst[i + 3] -= src[i + 3];
        dst[i + 4] -= src[i + 4];
        dst[i + 5] -= src[i + 5];
        dst[i + 6] -= src[i + 6];
        dst[i + 7] -= src[i + 7];
    }
    for (; i < K; i++)
        dst[i] -= src[i];
}
static inline void add3_floats_K(const float* __restrict__ a, const float* __restrict__ bv, const float* __restrict__ c, float* __restrict__ out, size_t K) {
    size_t i = 0;
    for (; i + 8 <= K; i += 8) {
        out[i + 0] = a[i + 0] + bv[i + 0] + c[i + 0];
        out[i + 1] = a[i + 1] + bv[i + 1] + c[i + 1];
        out[i + 2] = a[i + 2] + bv[i + 2] + c[i + 2];
        out[i + 3] = a[i + 3] + bv[i + 3] + c[i + 3];
        out[i + 4] = a[i + 4] + bv[i + 4] + c[i + 4];
        out[i + 5] = a[i + 5] + bv[i + 5] + c[i + 5];
        out[i + 6] = a[i + 6] + bv[i + 6] + c[i + 6];
        out[i + 7] = a[i + 7] + bv[i + 7] + c[i + 7];
    }
    for (; i < K; i++)
        out[i] = a[i] + bv[i] + c[i];
}
#endif

// ── Data structures ───────────────────────────────────────────────────────────

struct ConcatMeta {
    std::vector<size_t> list_offsets;
    std::vector<idx_t> id_offsets;

    std::vector<uint32_t> id_to_shard;
    std::vector<uint32_t> id_to_local_list;
    std::vector<uint32_t> id_to_global_old_list;

    std::vector<uint32_t> global_old_list_to_shard;
    std::vector<uint32_t> global_old_list_to_local_list;
};

// ── concat_ivf_meta_only ─────────────────────────────────────────────────────

static std::pair<IVFDataForMerge, ConcatMeta> concat_ivf_meta_only(const std::vector<IndexIVFPQ*>& indices, MergeRunStats* stats) {
    auto t0 = std::chrono::steady_clock::now();

    FAISS_THROW_IF_NOT(!indices.empty());
    size_t total_nlist = 0;
    size_t total_ntotal = 0;
    for (const IndexIVFPQ* idx : indices) {
        ensure_ivfpq_compatible(*idx);
        total_nlist += idx->nlist;
        total_ntotal += idx->ntotal;
    }

    IVFDataForMerge data;
    data.d = indices[0]->d;
    data.nlist = total_nlist;
    data.ntotal = total_ntotal;
    data.metric = indices[0]->metric_type;
    data.centroids.resize(total_nlist * data.d);
    data.lists.resize(total_nlist);

    ConcatMeta meta;
    meta.list_offsets.reserve(indices.size());
    meta.id_offsets.reserve(indices.size());
    meta.id_to_shard.resize(data.ntotal, 0);
    meta.id_to_local_list.resize(data.ntotal, 0);
    meta.id_to_global_old_list.resize(data.ntotal, 0);
    meta.global_old_list_to_shard.resize(total_nlist, 0);
    meta.global_old_list_to_local_list.resize(total_nlist, 0);

    size_t list_off = 0;
    idx_t id_off = 0;
    std::vector<float> centroid(data.d);

    for (size_t si = 0; si < indices.size(); si++) {
        IndexIVFPQ* idx = indices[si];
        meta.list_offsets.push_back(list_off);
        meta.id_offsets.push_back(id_off);

        for (size_t l = 0; l < idx->nlist; l++) {
            const size_t global_old_list = list_off + l;
            meta.global_old_list_to_shard[global_old_list] = static_cast<uint32_t>(si);
            meta.global_old_list_to_local_list[global_old_list] = static_cast<uint32_t>(l);

            idx->quantizer->reconstruct(l, centroid.data());
            std::memcpy(data.centroids.data() + global_old_list * data.d, centroid.data(), data.d * sizeof(float));

            const size_t list_size = idx->invlists->list_size(l);
            if (list_size == 0)
                continue;

            InvertedLists::ScopedIds ids(idx->invlists, l);
            const idx_t* id_ptr = ids.get();

            auto& out = data.lists[global_old_list];
            out.reserve(out.size() + list_size);

            for (size_t i = 0; i < list_size; i++) {
                const idx_t gid = id_ptr[i] + id_off;
                FAISS_THROW_IF_NOT(gid >= 0);
                FAISS_THROW_IF_NOT(static_cast<size_t>(gid) < data.ntotal);
                out.push_back(gid);
                meta.id_to_shard[gid] = static_cast<uint32_t>(si);
                meta.id_to_local_list[gid] = static_cast<uint32_t>(l);
                meta.id_to_global_old_list[gid] = static_cast<uint32_t>(global_old_list);
            }
        }

        list_off += idx->nlist;
        id_off += idx->ntotal;
    }

    if (stats) {
        auto t1 = std::chrono::steady_clock::now();
        stats->concat_s = std::chrono::duration<double>(t1 - t0).count();
    }

    return {std::move(data), std::move(meta)};
}

// ── decode_all_vectors ────────────────────────────────────────────────────────
// Parallelised over shards; each shard writes to a disjoint global-ID range.

static void decode_all_vectors(const std::vector<IndexIVFPQ*>& indices, const ConcatMeta& meta, size_t d, float* out_vectors, MergeRunStats* stats) {
    auto t0 = std::chrono::steady_clock::now();
    const int n_shards = static_cast<int>(indices.size());

#pragma omp parallel for schedule(dynamic, 1)
    for (int si_int = 0; si_int < n_shards; si_int++) {
        const size_t si = static_cast<size_t>(si_int);
        const IndexIVFPQ& idx = *indices[si];
        const idx_t id_off = meta.id_offsets[si];

        std::vector<idx_t> listnos;
        std::vector<float> decoded;

        for (size_t l = 0; l < idx.nlist; l++) {
            const size_t list_size = idx.invlists->list_size(l);
            if (list_size == 0)
                continue;

            InvertedLists::ScopedIds ids(idx.invlists, l);
            InvertedLists::ScopedCodes codes(idx.invlists, l);
            const idx_t* id_ptr = ids.get();
            const uint8_t* code_ptr = codes.get();

            listnos.assign(list_size, static_cast<idx_t>(l));
            decoded.resize(list_size * d);
            idx.decode_multiple(list_size, listnos.data(), code_ptr, decoded.data());

            for (size_t i = 0; i < list_size; i++) {
                const idx_t gid = id_ptr[i] + id_off;
                std::memcpy(out_vectors + static_cast<size_t>(gid) * d, decoded.data() + i * d, d * sizeof(float));
            }
        }
    }

    if (stats) {
        auto t1 = std::chrono::steady_clock::now();
        stats->extract_data_s += std::chrono::duration<double>(t1 - t0).count();
    }
}

// ── PQ codebook rebuild from old-code frequencies ────────────────────────────

static inline uint32_t extract_pq_subcode(const uint8_t* code, size_t m, size_t nbits) {
    const size_t bit_pos = m * nbits;
    const size_t byte_pos = bit_pos >> 3;
    const size_t bit_off = bit_pos & 7;

    uint32_t word = static_cast<uint32_t>(code[byte_pos]);
    if (bit_off + nbits > 8)
        word |= static_cast<uint32_t>(code[byte_pos + 1]) << 8;
    if (bit_off + nbits > 16)
        word |= static_cast<uint32_t>(code[byte_pos + 2]) << 16;
    if (bit_off + nbits > 24)
        word |= static_cast<uint32_t>(code[byte_pos + 3]) << 24;

    const uint32_t mask = (nbits >= 32) ? 0xffffffffu : ((1u << nbits) - 1u);
    return (word >> bit_off) & mask;
}

static void init_weighted_kmeans_centers(const std::vector<float>& points, const std::vector<double>& weights, size_t dim, size_t k, std::vector<float>* centers_out) {
    const size_t npts = weights.size();
    centers_out->assign(k * dim, 0.0f);
    if (npts == 0 || k == 0) {
        return;
    }

    std::vector<float>& centers = *centers_out;
    std::vector<float> min_dist(npts, std::numeric_limits<float>::infinity());

    size_t first = 0;
    for (size_t i = 1; i < npts; i++) {
        if (weights[i] > weights[first]) {
            first = i;
        }
    }
    std::memcpy(centers.data(), points.data() + first * dim, dim * sizeof(float));

    for (size_t i = 0; i < npts; i++) {
        min_dist[i] = fvec_L2sqr(points.data() + i * dim, centers.data(), dim);
    }

    for (size_t c = 1; c < k; c++) {
        size_t best_i = c % npts;
        double best_score = -1.0;
        for (size_t i = 0; i < npts; i++) {
            const double score = weights[i] * static_cast<double>(min_dist[i]);
            if (score > best_score) {
                best_score = score;
                best_i = i;
            }
        }

        std::memcpy(centers.data() + c * dim, points.data() + best_i * dim, dim * sizeof(float));
        const float* new_center = centers.data() + c * dim;
        for (size_t i = 0; i < npts; i++) {
            const float dis = fvec_L2sqr(points.data() + i * dim, new_center, dim);
            if (dis < min_dist[i]) {
                min_dist[i] = dis;
            }
        }
    }
}

static void weighted_kmeans_small(const std::vector<float>& points, const std::vector<double>& weights, size_t dim, size_t k, size_t niter, std::vector<float>* centers_out) {
    const size_t npts = weights.size();
    centers_out->clear();
    if (npts == 0 || k == 0) {
        return;
    }

    init_weighted_kmeans_centers(points, weights, dim, k, centers_out);
    std::vector<float>& centers = *centers_out;

    std::vector<int> assign(npts, -1);
    std::vector<int> next_assign(npts, -1);
    std::vector<double> sums(k * dim, 0.0);
    std::vector<double> sumw(k, 0.0);
    std::vector<float> point_best_dist(npts, 0.0f);

    for (size_t it = 0; it < niter; it++) {
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(sumw.begin(), sumw.end(), 0.0);
        bool changed = false;

        for (size_t i = 0; i < npts; i++) {
            const float* p = points.data() + i * dim;
            size_t best_c = 0;
            float best_dis = fvec_L2sqr(p, centers.data(), dim);
            for (size_t c = 1; c < k; c++) {
                const float dis = fvec_L2sqr(p, centers.data() + c * dim, dim);
                if (dis < best_dis) {
                    best_dis = dis;
                    best_c = c;
                }
            }
            next_assign[i] = static_cast<int>(best_c);
            point_best_dist[i] = best_dis;
            if (next_assign[i] != assign[i]) {
                changed = true;
            }

            const double w = weights[i];
            if (w <= 0.0) {
                continue;
            }
            sumw[best_c] += w;
            double* dst = sums.data() + best_c * dim;
            for (size_t j = 0; j < dim; j++) {
                dst[j] += w * static_cast<double>(p[j]);
            }
        }

        for (size_t c = 0; c < k; c++) {
            if (sumw[c] > 0.0) {
                float* center = centers.data() + c * dim;
                const double inv = 1.0 / sumw[c];
                const double* src = sums.data() + c * dim;
                for (size_t j = 0; j < dim; j++) {
                    center[j] = static_cast<float>(src[j] * inv);
                }
            }
        }

        for (size_t c = 0; c < k; c++) {
            if (sumw[c] > 0.0) {
                continue;
            }
            size_t best_i = 0;
            double best_score = -1.0;
            for (size_t i = 0; i < npts; i++) {
                const double score = weights[i] * static_cast<double>(point_best_dist[i]);
                if (score > best_score) {
                    best_score = score;
                    best_i = i;
                }
            }
            std::memcpy(centers.data() + c * dim, points.data() + best_i * dim, dim * sizeof(float));
            changed = true;
        }

        assign.swap(next_assign);
        if (!changed) {
            break;
        }
    }
}

static void train_pq_from_old_codeword_frequencies(const std::vector<IndexIVFPQ*>& indices, size_t d, size_t target_M, size_t target_nbits, MergeRunStats* stats, ProductQuantizer* pq_out) {
    auto t0 = std::chrono::steady_clock::now();

    FAISS_THROW_IF_NOT(pq_out);
    FAISS_THROW_IF_NOT(!indices.empty());

    pq_out->d = d;
    pq_out->M = target_M;
    pq_out->nbits = target_nbits;
    pq_out->set_derived_values();
    pq_out->verbose = false;

    const size_t M = pq_out->M;
    const size_t K = pq_out->ksub;
    const size_t dsub = pq_out->dsub;
    const size_t nshards = indices.size();

    for (size_t si = 0; si < nshards; si++) {
        const IndexIVFPQ& idx = *indices[si];
        FAISS_THROW_IF_NOT_MSG(idx.pq.M == M, "frequency-based PQ codebook rebuild requires target_M == old pq.M");
        FAISS_THROW_IF_NOT_MSG(idx.pq.nbits == target_nbits, "frequency-based PQ codebook rebuild requires target_nbits == old pq.nbits");
        FAISS_THROW_IF_NOT_MSG(static_cast<size_t>(idx.pq.dsub) == dsub, "frequency-based PQ codebook rebuild requires matching dsub across old and new PQ");
        FAISS_THROW_IF_NOT_MSG(static_cast<size_t>(idx.code_size) == idx.pq.code_size, "unexpected code_size mismatch in source IVFPQ index");
    }

    std::vector<double> code_freq(nshards * M * K, 0.0);

#pragma omp parallel for schedule(dynamic, 1)
    for (int si_int = 0; si_int < static_cast<int>(nshards); si_int++) {
        const size_t si = static_cast<size_t>(si_int);
        const IndexIVFPQ& idx = *indices[si];
        double* freq = code_freq.data() + si * M * K;

        for (size_t l = 0; l < idx.nlist; l++) {
            const size_t list_size = idx.invlists->list_size(l);
            if (list_size == 0) {
                continue;
            }
            InvertedLists::ScopedCodes codes(idx.invlists, l);
            const uint8_t* code_ptr = codes.get();
            for (size_t i = 0; i < list_size; i++) {
                const uint8_t* code = code_ptr + i * idx.code_size;
                for (size_t m = 0; m < M; m++) {
                    const uint32_t sub = extract_pq_subcode(code, m, target_nbits);
                    freq[m * K + static_cast<size_t>(sub)] += 1.0;
                }
            }
        }
    }

    pq_out->centroids.resize(M * K * dsub);

    for (size_t m = 0; m < M; m++) {
        std::vector<float> points;
        std::vector<double> weights;
        points.reserve(nshards * K * dsub);
        weights.reserve(nshards * K);

        for (size_t si = 0; si < nshards; si++) {
            const ProductQuantizer& old_pq = indices[si]->pq;
            const double* freq = code_freq.data() + si * M * K + m * K;
            for (size_t b = 0; b < K; b++) {
                const double w = freq[b];
                if (w <= 0.0) {
                    continue;
                }
                const float* c = old_pq.get_centroids(m, static_cast<uint8_t>(b));
                points.insert(points.end(), c, c + dsub);
                weights.push_back(w);
            }
        }

        if (weights.empty()) {
            const ProductQuantizer& fallback = indices[0]->pq;
            for (size_t b = 0; b < K; b++) {
                const float* src = fallback.get_centroids(m, static_cast<uint8_t>(b));
                float* dst = pq_out->centroids.data() + (m * K + b) * dsub;
                std::memcpy(dst, src, dsub * sizeof(float));
            }
            continue;
        }

        const size_t ktrain = std::min(K, weights.size());
        std::vector<float> centers;
        weighted_kmeans_small(points, weights, dsub, ktrain, 25, &centers);

        for (size_t b = 0; b < K; b++) {
            const size_t src_b = b % ktrain;
            const float* src = centers.data() + src_b * dsub;
            float* dst = pq_out->centroids.data() + (m * K + b) * dsub;
            std::memcpy(dst, src, dsub * sizeof(float));
        }
    }

    if (stats) {
        auto t1 = std::chrono::steady_clock::now();
        const double dt = std::chrono::duration<double>(t1 - t0).count();
        stats->pq_codebook_selection_s += dt;
        stats->ivfpq_pq_codebook_s += dt;
    }
}

static void train_pq_from_raw_residual_sample(
        const IVFDataForMerge& merged_ivf,
        const float* vectors,
        size_t target_M,
        size_t target_nbits,
        size_t max_train_points,
        bool hot_start,
        int hot_start_niter,
        MergeRunStats* stats,
        ProductQuantizer* pq_out) {
    auto t0 = std::chrono::steady_clock::now();

    FAISS_THROW_IF_NOT(pq_out);
    FAISS_THROW_IF_NOT(vectors);
    FAISS_THROW_IF_NOT(merged_ivf.ntotal > 0);
    FAISS_THROW_IF_NOT(merged_ivf.d % target_M == 0);

    pq_out->d = merged_ivf.d;
    pq_out->M = target_M;
    pq_out->nbits = target_nbits;
    pq_out->set_derived_values();
    pq_out->verbose = false;
    if (hot_start) {
        FAISS_THROW_IF_NOT_MSG(
                pq_out->centroids.size() ==
                        pq_out->M * pq_out->ksub * pq_out->dsub,
                "merge-aware PQ hot start requires initialized centroids");
        pq_out->train_type = ProductQuantizer::Train_hot_start;
        if (hot_start_niter > 0) {
            pq_out->cp.niter = hot_start_niter;
        }
    }

    if (stats) {
        std::fprintf(stderr, "[ivfpq_merge] raw residual PQ train: begin max_train_points=%zu\n", max_train_points);
        std::fflush(stderr);
    }

    const size_t d = merged_ivf.d;
    const size_t ntotal = merged_ivf.ntotal;
    size_t ntrain = max_train_points > 0 ? max_train_points : 256000;
    ntrain = std::max<size_t>(1, std::min(ntrain, ntotal));

    std::vector<uint32_t> id_to_list(ntotal, 0);
    for (size_t list_no = 0; list_no < merged_ivf.nlist; list_no++) {
        for (idx_t id : merged_ivf.lists[list_no]) {
            FAISS_THROW_IF_NOT(id >= 0);
            FAISS_THROW_IF_NOT(static_cast<size_t>(id) < ntotal);
            id_to_list[static_cast<size_t>(id)] = static_cast<uint32_t>(list_no);
        }
    }

    std::vector<float> residuals(ntrain * d);

#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(ntrain); i++) {
        const size_t id = (static_cast<size_t>(i) * ntotal) / ntrain;
        const size_t list_no = id_to_list[id];
        const float* x = vectors + id * d;
        const float* c = merged_ivf.centroids.data() + list_no * d;
        float* r = residuals.data() + static_cast<size_t>(i) * d;
        for (size_t j = 0; j < d; j++) {
            r[j] = x[j] - c[j];
        }
    }

    pq_out->train(static_cast<idx_t>(ntrain), residuals.data());

    if (stats) {
        std::fprintf(stderr, "[ivfpq_merge] raw residual PQ train: done ntrain=%zu\n", ntrain);
        std::fflush(stderr);
    }

    if (stats) {
        auto t1 = std::chrono::steady_clock::now();
        const double dt = std::chrono::duration<double>(t1 - t0).count();
        stats->pq_codebook_selection_s += dt;
        stats->ivfpq_pq_codebook_s += dt;
    }
}

// ── Index helpers for fcode tables ───────────────────────────────────────────

static inline size_t fcode_list_mk_index(size_t list_idx, size_t m, size_t b, size_t M, size_t K) {
    return ((list_idx * M + m) * K + b);
}

/// Index for -2<q_{s,m,a}^{old}, q_{m,b}^{new}> table: [shard][m][a][b].
static inline size_t pq_old_new_ip_index(size_t shard, size_t m, size_t a, size_t b, size_t M, size_t K) {
    return (((shard * M + m) * K + a) * K + b);
}

// ── PQMergeFCodeTables ────────────────────────────────────────────────────────

/**
 * Decomposed score tables for fast PQ re-encode.
 *
 * argmin_b ||residual^{(m)} - q_{m,b}^{new}||^2  (drop b-independent terms)
 * = argmin_b [ (1) + (2) + (3) + (4) ]:
 *   (1) ||q_{m,b}^{new}||^2
 *   (2) -2 <r^{(m)}, q_{m,b}^{new}>  with r^{(m)} = q_{m,a}^{old}
 *       (by_residual means decoded residual == old PQ codeword)
 *   (3) -2 <c_j^{(m)}, q_{m,b}^{new}>  (old IVF centroid term)
 *   (4) +2 <c_x^{(m)}, q_{m,b}^{new}>  (new IVF centroid term)
 */
struct PQMergeFCodeTables {
    size_t nshards = 0;
    size_t nlist_old = 0;
    size_t nlist_new = 0;
    size_t M = 0;
    size_t K = 0;
    /// (1) ||q_{m,b}^{new}||^2,                 layout [m][b]
    std::vector<float> norm_sq_new_codebook;
    /// (2) -2 <q_{s,m,a}^{old}, q_{m,b}^{new}>, layout [s][m][a][b]
    std::vector<float> minus2_ip_old_new_q;
    /// (3) -2 <c_j^{(m)}, q_{m,b}^{new}>,       layout [j][m][b]
    std::vector<float> minus2_ip_old_centroid_q;
    /// (4) +2 <c_x^{(m)}, q_{m,b}^{new}>,       layout [x][m][b]
    std::vector<float> plus2_ip_new_centroid_q;
};

static void fill_norm_sq_new_codebook(const ProductQuantizer& merged_pq, std::vector<float>* out) {
    const size_t M = merged_pq.M;
    const size_t K = merged_pq.ksub;
    const size_t dsub = merged_pq.dsub;
    out->resize(M * K);
    if (merged_pq.centroids_sq_lengths.size() == M * K) {
        std::memcpy(out->data(), merged_pq.centroids_sq_lengths.data(), M * K * sizeof(float));
    } else {
        for (size_t m = 0; m < M; m++) {
            for (size_t b = 0; b < K; b++) {
                (*out)[m * K + b] = fvec_norm_L2sqr(merged_pq.get_centroids(m, b), dsub);
            }
        }
    }
}

// ── precompute_fcode_tables_decomposed ────────────────────────────────────────
// Each of the three fills is parallelised over its own outer dimension.
// FAISS_THROW is called serially before entering any parallel region.

static void precompute_fcode_tables_decomposed(
        const std::vector<IndexIVFPQ*>& indices,
        const std::vector<float>& old_centroids,
        size_t nlist_old,
        const IVFDataForMerge& merged_ivf,
        const ProductQuantizer& merged_pq,
        PQMergeFCodeTables* out,
        MergeRunStats* stats) {
    auto t0 = std::chrono::steady_clock::now();
    FAISS_THROW_IF_NOT(out);
    FAISS_THROW_IF_NOT(!indices.empty());
    FAISS_THROW_IF_NOT(old_centroids.size() == nlist_old * merged_ivf.d);

    const size_t d = merged_ivf.d;
    const size_t M = merged_pq.M;
    const size_t K = merged_pq.ksub;
    const size_t dsub = merged_pq.dsub;
    const size_t nlist_new = merged_ivf.nlist;
    const size_t S = indices.size();

    out->nshards = S;
    out->nlist_old = nlist_old;
    out->nlist_new = nlist_new;
    out->M = M;
    out->K = K;

    fill_norm_sq_new_codebook(merged_pq, &out->norm_sq_new_codebook);
    out->minus2_ip_old_new_q.resize(S * M * K * K);
    out->minus2_ip_old_centroid_q.resize(nlist_old * M * K);
    out->plus2_ip_new_centroid_q.resize(nlist_new * M * K);

    // Serial compatibility check (FAISS_THROW must not fire inside omp).
    for (size_t s = 0; s < S; s++) {
        const ProductQuantizer& old_pq = indices[s]->pq;
        FAISS_THROW_IF_NOT(old_pq.M == M);
        FAISS_THROW_IF_NOT(old_pq.ksub == K);
        FAISS_THROW_IF_NOT(old_pq.dsub == dsub);
    }

    // Table (2): parallel over shards — each shard owns a disjoint slice.
#pragma omp parallel for schedule(static)
    for (int si = 0; si < static_cast<int>(S); si++) {
        const size_t s = static_cast<size_t>(si);
        std::vector<float> ip_row(K);
        const ProductQuantizer& old_pq = indices[s]->pq;
        for (size_t m = 0; m < M; m++) {
            const float* qnew_block = merged_pq.get_centroids(m, 0);
            for (size_t a = 0; a < K; a++) {
                const float* qold = old_pq.get_centroids(m, a);
                fvec_inner_products_ny(ip_row.data(), qold, qnew_block, dsub, K);
                float* dst = out->minus2_ip_old_new_q.data() + pq_old_new_ip_index(s, m, a, 0, M, K);
                for (size_t b = 0; b < K; b++)
                    dst[b] = -2.0f * ip_row[b];
            }
        }
    }

    // Table (3): parallel over old lists.
#pragma omp parallel for schedule(static)
    for (int ji = 0; ji < static_cast<int>(nlist_old); ji++) {
        const size_t j = static_cast<size_t>(ji);
        std::vector<float> ip_row(K);
        const float* cj = old_centroids.data() + j * d;
        for (size_t m = 0; m < M; m++) {
            const float* cj_m = cj + m * dsub;
            const float* qnew_block = merged_pq.get_centroids(m, 0);
            fvec_inner_products_ny(ip_row.data(), cj_m, qnew_block, dsub, K);
            float* dst = out->minus2_ip_old_centroid_q.data() + fcode_list_mk_index(j, m, 0, M, K);
            for (size_t b = 0; b < K; b++)
                dst[b] = -2.0f * ip_row[b];
        }
    }

    // Table (4): parallel over new lists.
#pragma omp parallel for schedule(static)
    for (int xi = 0; xi < static_cast<int>(nlist_new); xi++) {
        const size_t x = static_cast<size_t>(xi);
        std::vector<float> ip_row(K);
        const float* cx = merged_ivf.centroids.data() + x * d;
        for (size_t m = 0; m < M; m++) {
            const float* cx_m = cx + m * dsub;
            const float* qnew_block = merged_pq.get_centroids(m, 0);
            fvec_inner_products_ny(ip_row.data(), cx_m, qnew_block, dsub, K);
            float* dst = out->plus2_ip_new_centroid_q.data() + fcode_list_mk_index(x, m, 0, M, K);
            for (size_t b = 0; b < K; b++)
                dst[b] = 2.0f * ip_row[b];
        }
    }

    if (stats) {
        auto t1 = std::chrono::steady_clock::now();
        const double dt = std::chrono::duration<double>(t1 - t0).count();
        stats->subcode_mapping_s += dt;
        stats->ivfpq_fcode_precompute_s += dt;
    }
}

// ── k=1 rough shortlist helper ───────────────────────────────────────────────

static inline uint8_t argmin_sum_code(const float* __restrict__ a, const float* __restrict__ b, size_t K) {
    float best_score = a[0] + b[0];
    uint8_t best_code = 0;
    size_t idx = 1;
    for (; idx + 8 <= K; idx += 8) {
        const float s0 = a[idx + 0] + b[idx + 0];
        const float s1 = a[idx + 1] + b[idx + 1];
        const float s2 = a[idx + 2] + b[idx + 2];
        const float s3 = a[idx + 3] + b[idx + 3];
        const float s4 = a[idx + 4] + b[idx + 4];
        const float s5 = a[idx + 5] + b[idx + 5];
        const float s6 = a[idx + 6] + b[idx + 6];
        const float s7 = a[idx + 7] + b[idx + 7];
        if (s0 < best_score) {
            best_score = s0;
            best_code = static_cast<uint8_t>(idx + 0);
        }
        if (s1 < best_score) {
            best_score = s1;
            best_code = static_cast<uint8_t>(idx + 1);
        }
        if (s2 < best_score) {
            best_score = s2;
            best_code = static_cast<uint8_t>(idx + 2);
        }
        if (s3 < best_score) {
            best_score = s3;
            best_code = static_cast<uint8_t>(idx + 3);
        }
        if (s4 < best_score) {
            best_score = s4;
            best_code = static_cast<uint8_t>(idx + 4);
        }
        if (s5 < best_score) {
            best_score = s5;
            best_code = static_cast<uint8_t>(idx + 5);
        }
        if (s6 < best_score) {
            best_score = s6;
            best_code = static_cast<uint8_t>(idx + 6);
        }
        if (s7 < best_score) {
            best_score = s7;
            best_code = static_cast<uint8_t>(idx + 7);
        }
    }
    for (; idx < K; idx++) {
        const float score = a[idx] + b[idx];
        if (score < best_score) {
            best_score = score;
            best_code = static_cast<uint8_t>(idx);
        }
    }
    return best_code;
}

// ── build_index_from_merged_ivf_and_vectors ───────────────────────────────────

static std::unique_ptr<IndexIVFPQ> build_index_from_merged_ivf_and_vectors(
        const IVFDataForMerge& merged_ivf,
        const ProductQuantizer& pq,
        const float* vectors,
        MergeRunStats* stats,
        const std::vector<IndexIVFPQ*>& indices,
        const ConcatMeta* meta,
        const MergeOptions& merge2,
        const std::vector<float>& old_centroids) {
    auto t_build0 = std::chrono::steady_clock::now();

    const size_t d = merged_ivf.d;
    const size_t nlist = merged_ivf.nlist;
    const size_t ntotal = merged_ivf.ntotal;
    const size_t M = pq.M;
    const size_t K = pq.ksub;
    const size_t dsub = pq.dsub;
    const size_t code_size = pq.code_size;
    const int neighbor_kk = std::min(
            std::max(0, merge2.pq_fast_add_neighbor_kk),
            static_cast<int>(K > 0 ? (K - 1) : 0));

    FAISS_THROW_IF_NOT_MSG(
            vectors != nullptr,
            "final IVFPQ merge path requires raw vectors");
    FAISS_THROW_IF_NOT_MSG(
            meta != nullptr && !indices.empty(),
            "final IVFPQ merge path requires IVFPQ shard metadata");
    FAISS_THROW_IF_NOT_MSG(
            K <= 256,
            "final IVFPQ merge path assumes ksub <= 256");
    FAISS_THROW_IF_NOT_MSG(
            neighbor_kk > 0,
            "final IVFPQ merge path requires pq_fast_add_neighbor_kk > 0");

    std::unique_ptr<Index> quantizer = std::make_unique<IndexFlatL2>(static_cast<int>(d));
    quantizer->add(static_cast<idx_t>(nlist), merged_ivf.centroids.data());

    auto index = std::make_unique<IndexIVFPQ>(quantizer.get(), d, nlist, pq.M, pq.nbits, merged_ivf.metric, true);
    index->pq = pq;
    index->code_size = pq.code_size;
    index->invlists->code_size = pq.code_size;
    index->is_trained = true;
    index->by_residual = true;
    index->own_fields = true;
    index->quantizer = quantizer.release();
    index->precompute_table();

    ArrayInvertedLists* ail = dynamic_cast<ArrayInvertedLists*>(index->invlists);
    FAISS_THROW_IF_NOT(ail);

    if (stats) {
        const auto t_index_init1 = std::chrono::steady_clock::now();
        stats->ivfpq_reencode_index_init_s +=
                std::chrono::duration<double>(t_index_init1 - t_build0).count();
    }

    // ── Final IVFPQ merge algorithm ─────────────────────────────────────────
    // Fixed path: kk64 + rough k=1 cache + k1 exact fast path + generic batch4.
    // The rough shortlist is exactly one codeword per subspace; exact refine
    // scans that rough code plus its kk nearest new-PQ codewords.

    const auto t_fcode0 = std::chrono::steady_clock::now();
    PQMergeFCodeTables fcode;
    const size_t nlist_old = old_centroids.size() / d;
    precompute_fcode_tables_decomposed(
            indices, old_centroids, nlist_old, merged_ivf, pq, &fcode, stats);
    const double fcode_precompute_duration =
            std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t_fcode0)
                    .count();

    std::vector<uint8_t> newcode_nn(M * K * static_cast<size_t>(neighbor_kk));
    {
        const auto t_nn0 = std::chrono::steady_clock::now();
        std::vector<float> best_d(static_cast<size_t>(neighbor_kk));
        std::vector<uint8_t> best_b(static_cast<size_t>(neighbor_kk));
        const float INF = std::numeric_limits<float>::infinity();

        for (size_t m = 0; m < M; m++) {
            for (size_t b = 0; b < K; b++) {
                std::fill(best_d.begin(), best_d.end(), INF);
                std::fill(best_b.begin(), best_b.end(), 0);
                const float* cb = pq.get_centroids(m, static_cast<uint8_t>(b));
                for (size_t b2 = 0; b2 < K; b2++) {
                    if (b2 == b) {
                        continue;
                    }
                    const float dis = fvec_L2sqr(
                            cb, pq.get_centroids(m, static_cast<uint8_t>(b2)), dsub);
                    if (dis >= best_d[static_cast<size_t>(neighbor_kk - 1)]) {
                        continue;
                    }
                    int pos = neighbor_kk - 1;
                    while (pos > 0 && dis < best_d[static_cast<size_t>(pos - 1)]) {
                        best_d[static_cast<size_t>(pos)] =
                                best_d[static_cast<size_t>(pos - 1)];
                        best_b[static_cast<size_t>(pos)] =
                                best_b[static_cast<size_t>(pos - 1)];
                        --pos;
                    }
                    best_d[static_cast<size_t>(pos)] = dis;
                    best_b[static_cast<size_t>(pos)] = static_cast<uint8_t>(b2);
                }
                uint8_t* dst = newcode_nn.data() +
                        (m * K + b) * static_cast<size_t>(neighbor_kk);
                for (int t = 0; t < neighbor_kk; t++) {
                    dst[t] = best_b[static_cast<size_t>(t)];
                }
            }
        }
        if (stats) {
            const auto t_nn1 = std::chrono::steady_clock::now();
            stats->ivfpq_fast_add_neighbor_precompute_s +=
                    std::chrono::duration<double>(t_nn1 - t_nn0).count();
        }
    }

    // Reverse map: global id → new-list assignment.
    std::vector<uint32_t> id_to_new_list;
    {
        const auto t_id_map0 = std::chrono::steady_clock::now();
        id_to_new_list.assign(ntotal, 0);
        for (size_t x = 0; x < nlist; x++) {
            for (idx_t id : merged_ivf.lists[x]) {
                id_to_new_list[static_cast<size_t>(id)] = static_cast<uint32_t>(x);
            }
        }
        if (stats) {
            const auto t_id_map1 = std::chrono::steady_clock::now();
            stats->ivfpq_reencode_id_map_s +=
                    std::chrono::duration<double>(t_id_map1 - t_id_map0).count();
        }
    }

    // Pre-allocated per-new-list output buffers. Threads claim positions with
    // atomics; each final list is materialized with one add_entries call.
    std::vector<std::vector<idx_t>> out_ids(nlist);
    std::vector<std::vector<uint8_t>> out_codes(nlist);
    std::vector<std::atomic<size_t>> out_pos(nlist);
    {
        const auto t_alloc0 = std::chrono::steady_clock::now();
        for (size_t x = 0; x < nlist; x++) {
            const size_t sz = merged_ivf.lists[x].size();
            out_ids[x].resize(sz);
            out_codes[x].resize(sz * code_size);
        }
        for (auto& p : out_pos) {
            p.store(0, std::memory_order_relaxed);
        }
        if (stats) {
            const auto t_alloc1 = std::chrono::steady_clock::now();
            stats->ivfpq_reencode_output_alloc_s +=
                    std::chrono::duration<double>(t_alloc1 - t_alloc0).count();
        }
    }

    const size_t S = indices.size();
    const auto t_encode0 = std::chrono::steady_clock::now();
#pragma omp parallel
    {
        std::vector<float> prefix_tab(M * K * K);
        std::vector<float> residual(d);
        std::vector<uint8_t> code_buf(code_size);
        std::vector<int> rough_cache_x_to_pos(nlist, -1);
        std::vector<uint32_t> rough_cache_unique_x;
        std::vector<uint8_t> rough_k1_cache;
        std::vector<uint8_t> rough_k1_cache_valid;

#pragma omp for schedule(dynamic, 1)
        for (int si_int = 0; si_int < static_cast<int>(S); si_int++) {
            const size_t si = static_cast<size_t>(si_int);
            const IndexIVFPQ& idx = *indices[si];
            const idx_t id_off = meta->id_offsets[si];
            const size_t list_off = meta->list_offsets[si];

            // Level 1: prefix_tab[m][old_code][new_code] =
            // ||q_new||^2 - 2<q_old, q_new>.
            for (size_t m = 0; m < M; m++) {
                const float* q_norm = fcode.norm_sq_new_codebook.data() + m * K;
                for (size_t a = 0; a < K; a++) {
                    const float* p2 = fcode.minus2_ip_old_new_q.data() +
                            pq_old_new_ip_index(si, m, a, 0, M, K);
                    float* dst = prefix_tab.data() + (m * K + a) * K;
                    add_floats_K(q_norm, p2, dst, K);
                }
            }

            // Level 2: walk old lists in this shard, adding old centroid term.
            for (size_t l = 0; l < idx.nlist; l++) {
                const size_t list_size = idx.invlists->list_size(l);
                if (list_size == 0) {
                    continue;
                }
                const size_t global_j = list_off + l;

                for (size_t m = 0; m < M; m++) {
                    const float* p3 = fcode.minus2_ip_old_centroid_q.data() +
                            fcode_list_mk_index(global_j, m, 0, M, K);
                    for (size_t a = 0; a < K; a++) {
                        float* row = prefix_tab.data() + (m * K + a) * K;
                        add_inplace_floats_K(row, p3, K);
                    }
                }

                InvertedLists::ScopedIds ids(idx.invlists, l);
                InvertedLists::ScopedCodes codes(idx.invlists, l);
                const idx_t* id_ptr = ids.get();
                const uint8_t* code_ptr = codes.get();

                // Rough k=1 cache: within one old list, many vectors share
                // (new_list, subspace, old_subcode). Cache that argmin.
                rough_cache_unique_x.clear();
                for (size_t i = 0; i < list_size; i++) {
                    const idx_t gid = id_ptr[i] + id_off;
                    const size_t x = id_to_new_list[static_cast<size_t>(gid)];
                    if (rough_cache_x_to_pos[x] < 0) {
                        rough_cache_x_to_pos[x] =
                                static_cast<int>(rough_cache_unique_x.size());
                        rough_cache_unique_x.push_back(static_cast<uint32_t>(x));
                    }
                }
                const size_t cache_size = rough_cache_unique_x.size() * M * K;
                rough_k1_cache.resize(cache_size);
                rough_k1_cache_valid.assign(cache_size, 0);

                // Level 3: per vector, rough k=1 then exact kk-neighbor refine.
                for (size_t i = 0; i < list_size; i++) {
                    const idx_t gid = id_ptr[i] + id_off;
                    const size_t x = id_to_new_list[static_cast<size_t>(gid)];
                    const float* centroid_new = merged_ivf.centroids.data() + x * d;
                    const float* vx = vectors + static_cast<size_t>(gid) * d;
                    const uint8_t* old_code = code_ptr + i * code_size;

                    for (size_t t = 0; t < d; t++) {
                        residual[t] = vx[t] - centroid_new[t];
                    }

                    for (size_t m = 0; m < M; m++) {
                        const uint8_t a = old_code[m];
                        const float* prefix_ma =
                                prefix_tab.data() + (m * K + a) * K;
                        const float* p4 = fcode.plus2_ip_new_centroid_q.data() +
                                fcode_list_mk_index(x, m, 0, M, K);
                        const int x_pos = rough_cache_x_to_pos[x];
                        uint8_t c;
                        if (x_pos >= 0) {
                            const size_t key =
                                    (static_cast<size_t>(x_pos) * M + m) * K + a;
                            if (!rough_k1_cache_valid[key]) {
                                rough_k1_cache[key] =
                                        argmin_sum_code(prefix_ma, p4, K);
                                rough_k1_cache_valid[key] = 1;
                            }
                            c = rough_k1_cache[key];
                        } else {
                            c = argmin_sum_code(prefix_ma, p4, K);
                        }
                        const float* res_m = residual.data() + m * dsub;
                        float best_dis = fvec_L2sqr(
                                res_m, pq.get_centroids(m, c), dsub);
                        uint8_t best_code = c;
                        const uint8_t* nb = newcode_nn.data() +
                                (m * K + c) * static_cast<size_t>(neighbor_kk);

                        int u = 0;
                        for (; u + 3 < neighbor_kk; u += 4) {
                            const uint8_t b0 = nb[static_cast<size_t>(u)];
                            const uint8_t b1 = nb[static_cast<size_t>(u + 1)];
                            const uint8_t b2 = nb[static_cast<size_t>(u + 2)];
                            const uint8_t b3 = nb[static_cast<size_t>(u + 3)];
                            float d0, d1, d2, d3;
                            fvec_L2sqr_batch_4(
                                    res_m,
                                    pq.get_centroids(m, b0),
                                    pq.get_centroids(m, b1),
                                    pq.get_centroids(m, b2),
                                    pq.get_centroids(m, b3),
                                    dsub,
                                    d0,
                                    d1,
                                    d2,
                                    d3);
                            const float ds[4] = {d0, d1, d2, d3};
                            const uint8_t bs[4] = {b0, b1, b2, b3};
                            for (int j = 0; j < 4; j++) {
                                if (ds[j] < best_dis) {
                                    best_dis = ds[j];
                                    best_code = bs[j];
                                }
                            }
                        }
                        for (; u < neighbor_kk; u++) {
                            const uint8_t b = nb[static_cast<size_t>(u)];
                            const float dis = fvec_L2sqr(
                                    res_m, pq.get_centroids(m, b), dsub);
                            if (dis < best_dis) {
                                best_dis = dis;
                                best_code = b;
                            }
                        }
                        code_buf[m] = best_code;
                    }

                    const size_t pos =
                            out_pos[x].fetch_add(1, std::memory_order_relaxed);
                    out_ids[x][pos] = gid;
                    std::memcpy(
                            out_codes[x].data() + pos * code_size,
                            code_buf.data(),
                            code_size);
                }

                for (uint32_t x : rough_cache_unique_x) {
                    rough_cache_x_to_pos[x] = -1;
                }

                for (size_t m = 0; m < M; m++) {
                    const float* p3 = fcode.minus2_ip_old_centroid_q.data() +
                            fcode_list_mk_index(global_j, m, 0, M, K);
                    for (size_t a = 0; a < K; a++) {
                        float* row = prefix_tab.data() + (m * K + a) * K;
                        sub_inplace_floats_K(row, p3, K);
                    }
                }
            }
        }
    }
    if (stats) {
        const auto t_encode1 = std::chrono::steady_clock::now();
        stats->ivfpq_reencode_encode_codes_s +=
                std::chrono::duration<double>(t_encode1 - t_encode0).count();
    }

    {
        const auto t_add_entries0 = std::chrono::steady_clock::now();
        for (size_t x = 0; x < nlist; x++) {
            if (!out_ids[x].empty()) {
                ail->add_entries(
                        x, out_ids[x].size(), out_ids[x].data(), out_codes[x].data());
            }
        }
        if (stats) {
            const auto t_add_entries1 = std::chrono::steady_clock::now();
            stats->ivfpq_reencode_add_entries_s +=
                    std::chrono::duration<double>(t_add_entries1 - t_add_entries0).count();
        }
    }

    if (stats) {
        stats->fast_add_num_tables = 4.0;
        stats->fast_add_full_encodes = static_cast<double>(ntotal);
    }

    index->ntotal = static_cast<idx_t>(ntotal);

    if (stats) {
        auto t_build1 = std::chrono::steady_clock::now();
        const double build_total = std::chrono::duration<double>(t_build1 - t_build0).count();
        stats->add_to_index_s += build_total;
        stats->ivfpq_reencode_s += (build_total - fcode_precompute_duration);
    }

    return index;
}

} // namespace

// ── Public API ────────────────────────────────────────────────────────────────

void ivfpq_concat_merge(IndexIVFPQ& dst, IndexIVFPQ& src, idx_t add_id) {
    FAISS_THROW_IF_NOT(&dst != &src);
    FAISS_THROW_IF_NOT(dst.is_trained && src.is_trained);
    FAISS_THROW_IF_NOT(dst.d == src.d);
    FAISS_THROW_IF_NOT(dst.metric_type == src.metric_type);
    FAISS_THROW_IF_NOT(dst.by_residual == src.by_residual);
    FAISS_THROW_IF_NOT(dst.pq.M == src.pq.M);
    FAISS_THROW_IF_NOT(dst.pq.nbits == src.pq.nbits);
    FAISS_THROW_IF_NOT_MSG(dst.direct_map.no() && src.direct_map.no(), "concat merge does not support direct_map");

    const size_t dst_nlist = dst.nlist;
    const size_t src_nlist = src.nlist;
    const size_t new_nlist = dst_nlist + src_nlist;

    std::unique_ptr<Index> new_quantizer(clone_index(dst.quantizer));
    FAISS_THROW_IF_NOT(new_quantizer);

    std::vector<float> centroid(dst.d);
    for (size_t i = 0; i < src_nlist; i++) {
        src.quantizer->reconstruct(i, centroid.data());
        new_quantizer->add(1, centroid.data());
    }

    std::unique_ptr<InvertedLists> new_lists(new ArrayInvertedLists(new_nlist, dst.code_size));

    for (size_t l = 0; l < dst_nlist; l++) {
        const size_t sz = dst.invlists->list_size(l);
        if (sz == 0)
            continue;
        InvertedLists::ScopedIds ids(dst.invlists, l);
        InvertedLists::ScopedCodes codes(dst.invlists, l);
        new_lists->add_entries(l, sz, ids.get(), codes.get());
    }

    for (size_t l = 0; l < src_nlist; l++) {
        const size_t sz = src.invlists->list_size(l);
        if (sz == 0)
            continue;
        InvertedLists::ScopedIds ids(src.invlists, l);
        InvertedLists::ScopedCodes codes(src.invlists, l);
        const size_t new_l = dst_nlist + l;
        if (add_id == 0) {
            new_lists->add_entries(new_l, sz, ids.get(), codes.get());
        } else {
            std::vector<idx_t> new_ids(sz);
            for (size_t i = 0; i < sz; i++)
                new_ids[i] = ids[i] + add_id;
            new_lists->add_entries(new_l, sz, new_ids.data(), codes.get());
        }
    }

    if (dst.own_fields && dst.quantizer && dst.quantizer != src.quantizer) {
        delete dst.quantizer;
    }
    dst.quantizer = new_quantizer.release();
    dst.own_fields = true;
    dst.nlist = new_nlist;
    dst.replace_invlists(new_lists.release(), true);
    dst.ntotal += src.ntotal;
    src.ntotal = 0;
}

std::unique_ptr<Index> merge_ivfpq(const std::vector<IndexIVFPQ*>& indices, const MergeArguments& options) {
    FAISS_THROW_IF_NOT(!indices.empty());

    auto overall0 = std::chrono::steady_clock::now();

    if (options.method == MergeMethod::Concat) {
        std::unique_ptr<IndexShards> shards(new IndexShards(false));
        for (IndexIVFPQ* idx : indices)
            shards->add_shard(clone_index(idx));
        shards->successive_ids = true;
        if (options.run_stats) {
            options.run_stats->total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - overall0).count();
        }
        return shards;
    }

    FAISS_THROW_IF_NOT(options.method == MergeMethod::Merge);

    const MergeOptions merge_opts = options.merge;

    auto pair = concat_ivf_meta_only(indices, options.run_stats);
    IVFDataForMerge data = std::move(pair.first);
    ConcatMeta meta = std::move(pair.second);

    const std::vector<float> old_centroids = data.centroids;

    // Populate data.vectors — raw if available, PQ-decoded otherwise.
    // std::move avoids an extra ntotal*d*sizeof(float) copy on the decode path.
    if (merge_opts.ivf_merge_use_raw && have_full_raw_vectors(merge_opts, data.ntotal)) {
        data.vectors_view = merge_opts.training_vectors;
    } else {
        std::vector<float> decoded_all(data.ntotal * data.d);
        decode_all_vectors(indices, meta, data.d, decoded_all.data(), options.run_stats);
        data.vectors = std::move(decoded_all);
        data.vectors_view = data.vectors.data();
    }

    MergeOptions ivf_opts = merge_opts;
    if (ivf_opts.target_nlist == 0) {
        ivf_opts.target_nlist = data.nlist;
    }
    merge_ivf_data(data, ivf_opts, options.run_stats);

    const size_t target_M = merge_opts.target_M ? merge_opts.target_M : indices[0]->pq.M;
    const size_t target_nbits = merge_opts.target_nbits ? merge_opts.target_nbits : indices[0]->pq.nbits;

    FAISS_THROW_IF_NOT_MSG(
            have_full_raw_vectors(merge_opts, data.ntotal),
            "Merge requires full raw vectors via merge.training_vectors "
            "(n_training_vectors==ntotal) for exact PQ re-encode");

    ProductQuantizer pq(data.d, target_M, target_nbits);
    if (merge_opts.ivfpq_merge_aware_pq_hotstart) {
        train_pq_from_old_codeword_frequencies(
                indices,
                data.d,
                target_M,
                target_nbits,
                options.run_stats,
                &pq);
    }
    train_pq_from_raw_residual_sample(
            data,
            merge_opts.training_vectors,
            target_M,
            target_nbits,
            merge_opts.pq_train_max_pts,
            merge_opts.ivfpq_merge_aware_pq_hotstart,
            merge_opts.ivfpq_merge_aware_pq_niter,
            options.run_stats,
            &pq);

    std::unique_ptr<IndexIVFPQ> out = build_index_from_merged_ivf_and_vectors(
            data, pq, merge_opts.training_vectors, options.run_stats, indices, &meta, merge_opts, old_centroids);

    if (options.run_stats) {
        options.run_stats->total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - overall0).count();
    }

    return out;
}

} // namespace faiss
