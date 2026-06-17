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

// AVX2 SIMD for inner hot-path vector additions.
// Enabled only when compiled with -mavx2; tests intentionally omit this flag.
#ifdef __AVX2__
#include <immintrin.h>
#endif

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

// ── topk_smallest_codes ───────────────────────────────────────────────────────

static constexpr size_t FASTADD_MAX_K = 256;

/// Parallel to MergeRunStats::FASTADD_SUBCODE_RECALL_NUM_K / ivfpq_fast_add_subcode_recall_hits[].
static constexpr int kSubcodeRecallKs[MergeRunStats::FASTADD_SUBCODE_RECALL_NUM_K] = {1, 2, 4, 8, 16, 32, 64};

/// Parallel to MergeRunStats::FASTADD_NEIGHBOR_RECALL_NUM_K.
static constexpr int kSubcodeNeighborRecallKs[MergeRunStats::FASTADD_NEIGHBOR_RECALL_NUM_K] = {1, 2, 4, 8};
static constexpr int kSubcodeNeighborTopCap = 8;

static inline void insert_topk_code(float score, uint8_t code, int k, float* __restrict__ best_scores, uint8_t* __restrict__ best_codes) {
    if (score >= best_scores[static_cast<size_t>(k - 1)])
        return;
    int pos = k - 1;
    while (pos > 0 && score < best_scores[static_cast<size_t>(pos - 1)]) {
        best_scores[static_cast<size_t>(pos)] = best_scores[static_cast<size_t>(pos - 1)];
        best_codes[static_cast<size_t>(pos)] = best_codes[static_cast<size_t>(pos - 1)];
        --pos;
    }
    best_scores[static_cast<size_t>(pos)] = score;
    best_codes[static_cast<size_t>(pos)] = code;
}

static inline void topk_smallest_sum_codes(const float* __restrict__ a, const float* __restrict__ b, size_t K, int k, uint8_t* __restrict__ out_codes) {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(K <= FASTADD_MAX_K);

    const float INF = std::numeric_limits<float>::infinity();
    float best_scores[FASTADD_MAX_K];
    uint8_t best_codes[FASTADD_MAX_K];
    for (int i = 0; i < k; i++) {
        best_scores[static_cast<size_t>(i)] = INF;
        best_codes[static_cast<size_t>(i)] = 0;
    }

    size_t idx = 0;
    for (; idx + 8 <= K; idx += 8) {
        const float s0 = a[idx + 0] + b[idx + 0];
        const float s1 = a[idx + 1] + b[idx + 1];
        const float s2 = a[idx + 2] + b[idx + 2];
        const float s3 = a[idx + 3] + b[idx + 3];
        const float s4 = a[idx + 4] + b[idx + 4];
        const float s5 = a[idx + 5] + b[idx + 5];
        const float s6 = a[idx + 6] + b[idx + 6];
        const float s7 = a[idx + 7] + b[idx + 7];
        insert_topk_code(s0, static_cast<uint8_t>(idx + 0), k, best_scores, best_codes);
        insert_topk_code(s1, static_cast<uint8_t>(idx + 1), k, best_scores, best_codes);
        insert_topk_code(s2, static_cast<uint8_t>(idx + 2), k, best_scores, best_codes);
        insert_topk_code(s3, static_cast<uint8_t>(idx + 3), k, best_scores, best_codes);
        insert_topk_code(s4, static_cast<uint8_t>(idx + 4), k, best_scores, best_codes);
        insert_topk_code(s5, static_cast<uint8_t>(idx + 5), k, best_scores, best_codes);
        insert_topk_code(s6, static_cast<uint8_t>(idx + 6), k, best_scores, best_codes);
        insert_topk_code(s7, static_cast<uint8_t>(idx + 7), k, best_scores, best_codes);
    }
    for (; idx < K; idx++) {
        const float score = a[idx] + b[idx];
        insert_topk_code(score, static_cast<uint8_t>(idx), k, best_scores, best_codes);
    }

    for (int i = 0; i < k; i++)
        out_codes[static_cast<size_t>(i)] = best_codes[static_cast<size_t>(i)];
}

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

static inline uint8_t argmin2_sum_code(
        const float* __restrict__ a,
        const float* __restrict__ b,
        size_t K,
        float* best_out,
        float* second_out) {
    float best_score = a[0] + b[0];
    float second_score = std::numeric_limits<float>::infinity();
    uint8_t best_code = 0;
    for (size_t idx = 1; idx < K; idx++) {
        const float score = a[idx] + b[idx];
        if (score < best_score) {
            second_score = best_score;
            best_score = score;
            best_code = static_cast<uint8_t>(idx);
        } else if (score < second_score) {
            second_score = score;
        }
    }
    *best_out = best_score;
    *second_out = second_score;
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
    double fcode_precompute_duration = 0.0;

    const size_t d = merged_ivf.d;
    const size_t nlist = merged_ivf.nlist;
    const size_t ntotal = merged_ivf.ntotal;
    const size_t M = pq.M;
    const size_t K = pq.ksub;
    const size_t dsub = pq.dsub;
    const size_t code_size = pq.code_size;

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

    const int shortlist_k = merge2.pq_fast_add_k;
    const bool use_fast_add = (shortlist_k > 0 && meta != nullptr && !indices.empty());
    if (merge2.ivfpq_use_subcode_remap) {
        std::fprintf(stderr, "[ivfpq_merge] build index: begin subcode remap path ntotal=%zu nlist=%zu M=%zu K=%zu\n", ntotal, nlist, M, K);
        std::fflush(stderr);
    }
    if (merge2.pq_fast_add_measure_subcode_recall) {
        FAISS_THROW_IF_NOT_MSG(use_fast_add, "pq_fast_add_measure_subcode_recall requires pq_fast_add_k>0 and IVFPQ shard inputs (fast-add path)");
    }
    if (merge2.pq_fast_add_measure_subcode_recall_with_neighbors) {
        FAISS_THROW_IF_NOT_MSG(use_fast_add, "pq_fast_add_measure_subcode_recall_with_neighbors requires pq_fast_add_k>0 and IVFPQ shard inputs (fast-add path)");
    }

    // ── OPTIMIZATION: Analyze centroid changes for smart filtering ──────────
    std::vector<float> centroid_movement(nlist, 0.0f);
    std::vector<bool> needs_reencoding(nlist, true);
    size_t skipped_clusters = 0;
    
    const float movement_threshold = merge2.ivfpq_smart_filter_threshold;
    if (merge2.ivfpq_smart_filtering && !old_centroids.empty() && old_centroids.size() >= nlist * d) {
        for (size_t i = 0; i < nlist; i++) {
            const float* old_c = old_centroids.data() + i * d;
            const float* new_c = merged_ivf.centroids.data() + i * d;
            
            float movement = fvec_L2sqr(old_c, new_c, d);
            centroid_movement[i] = std::sqrt(movement);
            
            if (centroid_movement[i] < movement_threshold) {
                needs_reencoding[i] = false;
                skipped_clusters++;
            }
        }
        
        printf("Smart filtering: %zu/%zu clusters have small centroid movement (%.1f%%), skipping re-encoding\n",
               skipped_clusters, nlist, 100.0 * skipped_clusters / nlist);
    }

    const bool use_subcode_remap = merge2.ivfpq_use_subcode_remap;
    if (use_fast_add) {
        // ── Fast approximate encode: three-level prefix traversal ──────────
        //
        // Traversal order:  shard s  →  old_list j (within s)  →  vector v
        //
        // Level 1 — per shard s
        //   prefix_tab[m][a][b] = (1)[m,b] + (2)[s,m,a,b]
        //   Computed once per shard (M*K*K adds).
        //   Fixing s keeps the (2) table slice hot in L3 for the whole shard,
        //   eliminating the cache thrashing seen when s varies per vector.
        //
        // Level 2 — per old_list j (in-place update + restore)
        //   prefix_tab[m][a][b]  +=  (3)[j,m,b]   for every a
        //   (3)[j,m,b] is the same K-float row for all a, so the update is
        //   M*K SIMD calls of width K.  The table is restored (subtracted)
        //   before advancing to the next list.
        //   Cost per list: 2 * M*K*K ops, amortised over list_size vectors.
        //
        // Level 3 — per vector v  (inner hot loop)
        //   rough_scores[b] = prefix_tab[m][a_m][b] + (4)[x,m,b]   1 add/b
        //   → top-k shortlist → exact L2 on k candidates → best code
        //
        // Memory: prefix_tab layout  [m * K * K + a * K + b]
        //   Each (m,a) row is K contiguous floats → SIMD-friendly, fits in L2
        //   after Level 1 initialisation.
        //
        // Further savings vs. previous new_list-first traversal
        //   • Codes read sequentially from ScopedCodes — no ntotal*code_size
        //     flat id_to_code buffer, no scatter/gather.
        //   • No unordered_map grouping per new_list.
        //   • add_entries called once per new_list (batched) instead of once
        //     per vector; eliminates O(ntotal) push_back calls.

        const auto t_fcode0 = std::chrono::steady_clock::now();
        PQMergeFCodeTables fcode;
        const size_t nlist_old = old_centroids.size() / d;
        precompute_fcode_tables_decomposed(indices, old_centroids, nlist_old, merged_ivf, pq, &fcode, stats);
        fcode_precompute_duration = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_fcode0).count();
        if (use_subcode_remap) {
            std::fprintf(stderr, "[ivfpq_merge] subcode remap: fcode precompute done %.3fs\n", fcode_precompute_duration);
            std::fflush(stderr);
        }

        // Per-subspace, per-new-code nearest-neighbor precompute (new PQ codebook only).
        // Used by fast-add exact refinement: exact set = union(top-k rough codes, their kk neighbors).
        // Layout: nn[(m*K + b)*kk + t] gives t-th nearest neighbor code for (m,b).
        std::vector<uint8_t> newcode_nn;
        const int neighbor_kk = std::min(std::max(0, merge2.pq_fast_add_neighbor_kk), static_cast<int>(K > 0 ? (K - 1) : 0));
        if (neighbor_kk > 0) {
            const auto t_nn0 = std::chrono::steady_clock::now();
            FAISS_THROW_IF_NOT_MSG(K <= 256, "pq_fast_add_measure_subcode_recall_with_neighbors assumes ksub<=256");
            newcode_nn.resize(M * K * static_cast<size_t>(neighbor_kk));

            std::vector<float> best_d(static_cast<size_t>(neighbor_kk));
            std::vector<uint8_t> best_b(static_cast<size_t>(neighbor_kk));
            const float INF = std::numeric_limits<float>::infinity();

            for (size_t m = 0; m < M; m++) {
                for (size_t b = 0; b < K; b++) {
                    for (int t = 0; t < neighbor_kk; t++) {
                        best_d[static_cast<size_t>(t)] = INF;
                        best_b[static_cast<size_t>(t)] = 0;
                    }
                    const float* cb = pq.get_centroids(m, static_cast<uint8_t>(b));
                    for (size_t b2 = 0; b2 < K; b2++) {
                        if (b2 == b)
                            continue;
                        const float dis = fvec_L2sqr(cb, pq.get_centroids(m, static_cast<uint8_t>(b2)), dsub);
                        // insertion into fixed-size sorted list (small kk=32)
                        if (dis >= best_d[static_cast<size_t>(neighbor_kk - 1)])
                            continue;
                        int pos = neighbor_kk - 1;
                        while (pos > 0 && dis < best_d[static_cast<size_t>(pos - 1)]) {
                            best_d[static_cast<size_t>(pos)] = best_d[static_cast<size_t>(pos - 1)];
                            best_b[static_cast<size_t>(pos)] = best_b[static_cast<size_t>(pos - 1)];
                            --pos;
                        }
                        best_d[static_cast<size_t>(pos)] = dis;
                        best_b[static_cast<size_t>(pos)] = static_cast<uint8_t>(b2);
                    }
                    uint8_t* dst = newcode_nn.data() + (m * K + b) * static_cast<size_t>(neighbor_kk);
                    for (int t = 0; t < neighbor_kk; t++) {
                        dst[t] = best_b[static_cast<size_t>(t)];
                    }
                }
            }
            if (stats) {
                const auto t_nn1 = std::chrono::steady_clock::now();
                stats->ivfpq_fast_add_neighbor_precompute_s += std::chrono::duration<double>(t_nn1 - t_nn0).count();
            }
        }

        std::vector<int> subspace_neighbor_kk(M, neighbor_kk);
        if (merge2.ivfpq_subspace_neighbor_schedule && neighbor_kk > 0) {
            std::vector<std::pair<float, size_t>> energy;
            energy.reserve(M);
            for (size_t m = 0; m < M; m++) {
                std::vector<float> mean(dsub, 0.0f);
                for (size_t b = 0; b < K; b++) {
                    const float* c = pq.get_centroids(m, static_cast<uint8_t>(b));
                    for (size_t t = 0; t < dsub; t++) {
                        mean[t] += c[t];
                    }
                }
                for (size_t t = 0; t < dsub; t++) {
                    mean[t] /= static_cast<float>(K);
                }

                float var = 0.0f;
                for (size_t b = 0; b < K; b++) {
                    const float* c = pq.get_centroids(m, static_cast<uint8_t>(b));
                    for (size_t t = 0; t < dsub; t++) {
                        const float diff = c[t] - mean[t];
                        var += diff * diff;
                    }
                }
                energy.emplace_back(var / static_cast<float>(K), m);
            }

            std::sort(
                    energy.begin(),
                    energy.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });

            const float high_fraction =
                    std::max(0.0f, std::min(1.0f, merge2.ivfpq_subspace_neighbor_high_fraction));
            const float mid_fraction =
                    std::max(0.0f, std::min(1.0f, merge2.ivfpq_subspace_neighbor_mid_fraction));
            const size_t high_count = std::min(
                    M,
                    static_cast<size_t>(std::llround(high_fraction * static_cast<float>(M))));
            const size_t mid_count = std::min(
                    M - high_count,
                    static_cast<size_t>(std::llround(mid_fraction * static_cast<float>(M))));
            const int mid_kk = std::min(
                    neighbor_kk,
                    std::max(0, merge2.ivfpq_subspace_neighbor_mid_kk));
            const int low_kk = std::min(
                    neighbor_kk,
                    std::max(0, merge2.ivfpq_subspace_neighbor_low_kk));

            for (size_t rank = 0; rank < M; rank++) {
                const size_t m = energy[rank].second;
                if (rank < high_count) {
                    subspace_neighbor_kk[m] = neighbor_kk;
                } else if (rank < high_count + mid_count) {
                    subspace_neighbor_kk[m] = mid_kk;
                } else {
                    subspace_neighbor_kk[m] = low_kk;
                }
            }

            size_t high_used = 0, mid_used = 0, low_used = 0;
            for (size_t m = 0; m < M; m++) {
                if (subspace_neighbor_kk[m] == neighbor_kk) {
                    high_used++;
                } else if (subspace_neighbor_kk[m] == mid_kk) {
                    mid_used++;
                } else {
                    low_used++;
                }
            }
            std::fprintf(
                    stderr,
                    "[ivfpq_merge] subspace neighbor schedule: high=%zu kk=%d mid=%zu kk=%d low=%zu kk=%d\n",
                    high_used,
                    neighbor_kk,
                    mid_used,
                    mid_kk,
                    low_used,
                    low_kk);
            std::fflush(stderr);
        }

        const bool measure_neighbors = stats != nullptr && merge2.pq_fast_add_measure_subcode_recall_with_neighbors;
        if (measure_neighbors) {
            for (int ji = 0; ji < MergeRunStats::FASTADD_NEIGHBOR_RECALL_NUM_K; ji++) {
                stats->ivfpq_fast_add_subcode_neighbor_recall_hits[ji] = 0;
                stats->ivfpq_fast_add_subcode_neighbor_recall_union_size_sum[ji] = 0;
            }
            stats->ivfpq_fast_add_subcode_neighbor_recall_pairs = 0;
        }

        // Reverse map: global id → new_list assignment.
        // ntotal * 4 bytes (vs. ntotal * code_size bytes for id_to_code).
        std::vector<uint32_t> id_to_new_list(ntotal, 0);
        for (size_t x = 0; x < nlist; x++) {
            for (idx_t id : merged_ivf.lists[x]) {
                id_to_new_list[static_cast<size_t>(id)] = static_cast<uint32_t>(x);
            }
        }
        if (use_subcode_remap) {
            std::fprintf(stderr, "[ivfpq_merge] subcode remap: id_to_new_list done\n");
            std::fflush(stderr);
        }

        // Pre-allocated per-new-list output buffers (sizes known upfront).
        // Atomic position counters let multiple threads write to the same
        // new_list slot without a mutex.
        std::vector<std::vector<idx_t>> out_ids(nlist);
        std::vector<std::vector<uint8_t>> out_codes(nlist);
        for (size_t x = 0; x < nlist; x++) {
            const size_t sz = merged_ivf.lists[x].size();
            out_ids[x].resize(sz);
            out_codes[x].resize(sz * code_size);
        }
        if (use_subcode_remap) {
            std::fprintf(stderr, "[ivfpq_merge] subcode remap: output buffers allocated\n");
            std::fflush(stderr);
        }
        std::vector<std::atomic<size_t>> out_pos(nlist);
        for (auto& p : out_pos)
            p.store(0, std::memory_order_relaxed);

        const int k = std::min(shortlist_k, static_cast<int>(K));
        const size_t S = indices.size();
        const bool time_split = merge2.pq_fast_add_time_split && stats != nullptr;
        const bool use_margin_full_scan =
                merge2.ivfpq_full_scan_margin_ratio > 0.0f ||
                merge2.ivfpq_full_scan_margin_abs > 0.0f;
        const bool use_adaptive_k1_margin =
                merge2.ivfpq_adaptive_k1_by_margin && k == 1 &&
                merge2.ivfpq_adaptive_k1_min_neighbor_kk < neighbor_kk &&
                (merge2.ivfpq_adaptive_k1_high_margin_ratio > 0.0f ||
                 merge2.ivfpq_adaptive_k1_mid_margin_ratio > 0.0f);
        const bool use_rough_k1_cache =
                merge2.ivfpq_cache_rough_k1 && k == 1 &&
                !use_margin_full_scan && !use_adaptive_k1_margin;
        const int adaptive_k1_min_neighbor_kk =
                std::min(
                        std::max(0, merge2.ivfpq_adaptive_k1_min_neighbor_kk),
                        neighbor_kk);
        const bool measure_subcode_recall = stats != nullptr && merge2.pq_fast_add_measure_subcode_recall;
        const bool measure_subcode_recall_neighbors = measure_neighbors;
        double rough_topk_acc_s = 0;
        double exact_l2_acc_s = 0;
        double margin_full_scan_acc = 0;

        if (stats && merge2.pq_fast_add_measure_subcode_recall) {
            for (int ji = 0; ji < MergeRunStats::FASTADD_SUBCODE_RECALL_NUM_K; ++ji) {
                stats->ivfpq_fast_add_subcode_recall_hits[ji] = 0;
            }
            stats->ivfpq_fast_add_subcode_recall_pairs = 0;
        }

#pragma omp parallel reduction(+ : rough_topk_acc_s, exact_l2_acc_s, margin_full_scan_acc)
        {
            // Thread-local prefix table: M * K * K floats.
            // For M=16, K=256: 4 MB — fits in L3, hot in L2 during Level 2.
            // Lifecycle: reinitialised per shard; updated in-place per list,
            // then restored before the next list.
            std::vector<float> prefix_tab(M * K * K);
            // One shortlist row per subspace (same as interleaved pass, without reuse races).
            std::vector<uint8_t> shortlist_all(M * static_cast<size_t>(k));
            std::vector<float> rough_best_score(M, 0.0f);
            std::vector<float> rough_second_score(M, 0.0f);
            std::vector<float> residual(d);
            std::vector<uint8_t> code_buf(code_size);
            uint64_t recall_lh[MergeRunStats::FASTADD_SUBCODE_RECALL_NUM_K] = {0};
            uint64_t recall_lp = 0;
            std::vector<uint8_t> rough_top64(64);

            uint64_t neigh_lh[MergeRunStats::FASTADD_NEIGHBOR_RECALL_NUM_K] = {0};
            uint64_t neigh_union_sum[MergeRunStats::FASTADD_NEIGHBOR_RECALL_NUM_K] = {0};
            uint64_t neigh_lp = 0;
            std::vector<uint8_t> rough_top8(static_cast<size_t>(kSubcodeNeighborTopCap));
            std::vector<int> remap_x_to_pos(nlist, -1);
            std::vector<uint32_t> remap_unique_x;
            std::vector<uint8_t> remap_table;
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

                // ── Level 1: per shard ──────────────────────────────────────
                // prefix_tab[m*K*K + a*K + b] = (1)[m,b] + (2)[si,m,a,b]
                for (size_t m = 0; m < M; m++) {
                    const float* Qm = fcode.norm_sq_new_codebook.data() + m * K;
                    for (size_t a = 0; a < K; a++) {
                        const float* p2 = fcode.minus2_ip_old_new_q.data() + pq_old_new_ip_index(si, m, a, 0, M, K);
                        float* dst = prefix_tab.data() + (m * K + a) * K;
                        add_floats_K(Qm, p2, dst, K);
                    }
                }

                // ── Level 2: per old_list j within shard si ─────────────────
                for (size_t l = 0; l < idx.nlist; l++) {
                    const size_t list_size = idx.invlists->list_size(l);
                    if (list_size == 0)
                        continue;

                    const size_t global_j = list_off + l;

                    // In-place update: prefix_tab[m][a][b] += (3)[j,m,b]
                    // The same p3 row is broadcast across all K rows of
                    // dimension a for each subspace m.
                    for (size_t m = 0; m < M; m++) {
                        const float* p3 = fcode.minus2_ip_old_centroid_q.data() + fcode_list_mk_index(global_j, m, 0, M, K);
                        for (size_t a = 0; a < K; a++) {
                            float* row = prefix_tab.data() + (m * K + a) * K;
                            add_inplace_floats_K(row, p3, K);
                        }
                    }

                    InvertedLists::ScopedIds ids(idx.invlists, l);
                    InvertedLists::ScopedCodes codes(idx.invlists, l);
                    const idx_t* id_ptr = ids.get();
                    const uint8_t* code_ptr = codes.get();

                    if (use_subcode_remap) {
                        if ((global_j % 250) == 0) {
                            std::fprintf(stderr, "[ivfpq_merge] subcode remap: old_list=%zu list_size=%zu\n", global_j, list_size);
                            std::fflush(stderr);
                        }
                        remap_unique_x.clear();
                        for (size_t i = 0; i < list_size; i++) {
                            const idx_t gid = id_ptr[i] + id_off;
                            const size_t x = id_to_new_list[static_cast<size_t>(gid)];
                            if (remap_x_to_pos[x] < 0) {
                                remap_x_to_pos[x] = static_cast<int>(remap_unique_x.size());
                                remap_unique_x.push_back(static_cast<uint32_t>(x));
                            }
                        }

                        remap_table.resize(M * K);
                        const auto t_remap0 = std::chrono::steady_clock::now();
                        for (size_t ux_pos = 0; ux_pos < remap_unique_x.size(); ux_pos++) {
                            const size_t x = remap_unique_x[ux_pos];
                            for (size_t m = 0; m < M; m++) {
                                const float* p4 = fcode.plus2_ip_new_centroid_q.data() + fcode_list_mk_index(x, m, 0, M, K);
                                for (size_t a = 0; a < K; a++) {
                                    const float* prefix_ma = prefix_tab.data() + (m * K + a) * K;
                                    uint8_t best_code = 0;
                                    float best_score = prefix_ma[0] + p4[0];
                                    for (size_t b = 1; b < K; b++) {
                                        const float score = prefix_ma[b] + p4[b];
                                        if (score < best_score) {
                                            best_score = score;
                                            best_code = static_cast<uint8_t>(b);
                                        }
                                    }
                                    remap_table[m * K + a] = best_code;
                                }
                            }

                            for (size_t i = 0; i < list_size; i++) {
                                const idx_t gid = id_ptr[i] + id_off;
                                if (id_to_new_list[static_cast<size_t>(gid)] != x) {
                                    continue;
                                }
                                const uint8_t* old_code = code_ptr + i * code_size;
                                for (size_t m = 0; m < M; m++) {
                                    code_buf[m] = remap_table[m * K + old_code[m]];
                                }

                                const size_t pos = out_pos[x].fetch_add(1, std::memory_order_relaxed);
                                out_ids[x][pos] = gid;
                                std::memcpy(out_codes[x].data() + pos * code_size, code_buf.data(), code_size);
                            }
                        }
                        if (time_split) {
                            const auto t_remap1 = std::chrono::steady_clock::now();
                            rough_topk_acc_s += std::chrono::duration<double>(t_remap1 - t_remap0).count();
                        }

                        for (uint32_t x : remap_unique_x) {
                            remap_x_to_pos[x] = -1;
                        }

                        for (size_t m = 0; m < M; m++) {
                            const float* p3 = fcode.minus2_ip_old_centroid_q.data() + fcode_list_mk_index(global_j, m, 0, M, K);
                            for (size_t a = 0; a < K; a++) {
                                float* row = prefix_tab.data() + (m * K + a) * K;
                                sub_inplace_floats_K(row, p3, K);
                            }
                        }
                        continue;
                    }

                    if (use_rough_k1_cache) {
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
                    }

                    // ── Level 3: per vector v ───────────────────────────────
                    for (size_t i = 0; i < list_size; i++) {
                        const idx_t gid = id_ptr[i] + id_off;
                        const size_t x = id_to_new_list[static_cast<size_t>(gid)];
                        
                        // OPTIMIZATION: Skip re-encoding if centroid movement is small
                        if (!needs_reencoding[x]) {
                            // Directly copy existing codes without re-encoding
                            const uint8_t* old_code = code_ptr + i * code_size;
                            const size_t pos = out_pos[x].fetch_add(1, std::memory_order_relaxed);
                            out_ids[x][pos] = gid;
                            std::memcpy(out_codes[x].data() + pos * code_size, old_code, code_size);
                            continue;
                        }
                        
                        const float* centroid_new = merged_ivf.centroids.data() + x * d;
                        const float* vx = vectors + static_cast<size_t>(gid) * d;

                        for (size_t t = 0; t < d; t++) {
                            residual[t] = vx[t] - centroid_new[t];
                        }

                        // Codes read sequentially from the inverted list —
                        // no random-access scatter/gather via id_to_code.
                        const uint8_t* old_code = code_ptr + i * code_size;

                        if (measure_subcode_recall) {
                            const int k_rough_cap = std::min(64, static_cast<int>(K));
                            for (size_t m = 0; m < M; m++) {
                                const uint8_t a = old_code[m];
                                const float* prefix_ma = prefix_tab.data() + (m * K + a) * K;
                                const float* p4 = fcode.plus2_ip_new_centroid_q.data() + fcode_list_mk_index(x, m, 0, M, K);
                                topk_smallest_sum_codes(prefix_ma, p4, K, k_rough_cap, rough_top64.data());
                                const float* res_m = residual.data() + m * dsub;
                                uint8_t true_best = 0;
                                float best_dis = std::numeric_limits<float>::infinity();
                                for (size_t b = 0; b < K; b++) {
                                    const float dis = fvec_L2sqr(res_m, pq.get_centroids(m, static_cast<uint8_t>(b)), dsub);
                                    if (dis < best_dis) {
                                        best_dis = dis;
                                        true_best = static_cast<uint8_t>(b);
                                    }
                                }
                                for (int ji = 0; ji < MergeRunStats::FASTADD_SUBCODE_RECALL_NUM_K; ji++) {
                                    const int ks = kSubcodeRecallKs[ji];
                                    const int k_check = std::min(ks, static_cast<int>(K));
                                    bool found = false;
                                    for (int t = 0; t < k_check; t++) {
                                        if (rough_top64[static_cast<size_t>(t)] == true_best) {
                                            found = true;
                                            break;
                                        }
                                    }
                                    if (found) {
                                        recall_lh[ji]++;
                                    }
                                }
                                recall_lp++;
                            }
                        }

                        if (measure_subcode_recall_neighbors) {
                            const int kk = std::min(std::max(0, merge2.pq_fast_add_neighbor_kk), static_cast<int>(K > 0 ? (K - 1) : 0));
                            const int top_cap = std::min(kSubcodeNeighborTopCap, static_cast<int>(K));

                            for (size_t m = 0; m < M; m++) {
                                const uint8_t a = old_code[m];
                                const float* prefix_ma = prefix_tab.data() + (m * K + a) * K;
                                const float* p4 = fcode.plus2_ip_new_centroid_q.data() + fcode_list_mk_index(x, m, 0, M, K);
                                topk_smallest_sum_codes(prefix_ma, p4, K, top_cap, rough_top8.data());

                                const float* res_m = residual.data() + m * dsub;
                                uint8_t true_best = 0;
                                float best_dis = std::numeric_limits<float>::infinity();
                                for (size_t b = 0; b < K; b++) {
                                    const float dis = fvec_L2sqr(res_m, pq.get_centroids(m, static_cast<uint8_t>(b)), dsub);
                                    if (dis < best_dis) {
                                        best_dis = dis;
                                        true_best = static_cast<uint8_t>(b);
                                    }
                                }

                                // For k in {1,2,4,8}: union of rough top-k and their kk-neighbors (unique).
                                bool seen[256] = {false};
                                uint32_t union_sz = 0;
                                bool hit = false;

                                for (int ji = 0; ji < MergeRunStats::FASTADD_NEIGHBOR_RECALL_NUM_K; ji++) {
                                    const int rk = std::min(kSubcodeNeighborRecallKs[ji], top_cap);
                                    // Add newly included rough codes (incremental across ji).
                                    const int prev_rk = (ji == 0) ? 0 : std::min(kSubcodeNeighborRecallKs[ji - 1], top_cap);
                                    for (int t = prev_rk; t < rk; t++) {
                                        const uint8_t c = rough_top8[static_cast<size_t>(t)];
                                        if (!seen[c]) {
                                            seen[c] = true;
                                            union_sz++;
                                            if (c == true_best)
                                                hit = true;
                                        }
                                        if (kk > 0) {
                                            const uint8_t* nb = newcode_nn.data() + (m * K + c) * static_cast<size_t>(kk);
                                            for (int u = 0; u < kk; u++) {
                                                const uint8_t nbc = nb[static_cast<size_t>(u)];
                                                if (!seen[nbc]) {
                                                    seen[nbc] = true;
                                                    union_sz++;
                                                    if (nbc == true_best)
                                                        hit = true;
                                                }
                                            }
                                        }
                                    }
                                    neigh_union_sum[ji] += union_sz;
                                    if (hit) {
                                        neigh_lh[ji]++;
                                    }
                                }
                                neigh_lp++;
                            }
                        }

                        auto rough_pass = [&]() {
                            for (size_t m = 0; m < M; m++) {
                                const uint8_t a = old_code[m];
                                const float* prefix_ma = prefix_tab.data() + (m * K + a) * K;
                                const float* p4 = fcode.plus2_ip_new_centroid_q.data() + fcode_list_mk_index(x, m, 0, M, K);
                                uint8_t* out = shortlist_all.data() + m * static_cast<size_t>(k);
                                if (k == 1 && (use_margin_full_scan || use_adaptive_k1_margin)) {
                                    float best_score = 0.0f;
                                    float second_score = 0.0f;
                                    out[0] = argmin2_sum_code(prefix_ma, p4, K, &best_score, &second_score);
                                    rough_best_score[m] = best_score;
                                    rough_second_score[m] = second_score;
                                } else if (k == 1) {
                                    if (use_rough_k1_cache) {
                                        const int x_pos = rough_cache_x_to_pos[x];
                                        if (x_pos >= 0) {
                                            const size_t key =
                                                    (static_cast<size_t>(x_pos) * M + m) * K + a;
                                            if (!rough_k1_cache_valid[key]) {
                                                rough_k1_cache[key] =
                                                        argmin_sum_code(prefix_ma, p4, K);
                                                rough_k1_cache_valid[key] = 1;
                                            }
                                            out[0] = rough_k1_cache[key];
                                        } else {
                                            out[0] = argmin_sum_code(prefix_ma, p4, K);
                                        }
                                    } else {
                                        out[0] = argmin_sum_code(prefix_ma, p4, K);
                                    }
                                } else {
                                    topk_smallest_sum_codes(prefix_ma, p4, K, k, out);
                                }
                            }
                        };
                        auto exact_pass = [&]() {
                            for (size_t m = 0; m < M; m++) {
                                const float* res_m = residual.data() + m * dsub;
                                float best_dis = std::numeric_limits<float>::infinity();
                                uint8_t best_code = 0;
                                const uint8_t* sl = shortlist_all.data() + m * static_cast<size_t>(k);

                                bool do_full_scan = false;
                                if (use_margin_full_scan && k == 1) {
                                    const float gap = rough_second_score[m] - rough_best_score[m];
                                    const float denom = std::fabs(rough_best_score[m]) + 1.0f;
                                    const float ratio = gap / denom;
                                    do_full_scan =
                                            (merge2.ivfpq_full_scan_margin_abs > 0.0f &&
                                             gap <= merge2.ivfpq_full_scan_margin_abs) ||
                                            (merge2.ivfpq_full_scan_margin_ratio > 0.0f &&
                                             ratio <= merge2.ivfpq_full_scan_margin_ratio);
                                }

                                if (do_full_scan) {
                                    for (size_t b = 0; b < K; b++) {
                                        const float dis = fvec_L2sqr(
                                                res_m,
                                                pq.get_centroids(m, static_cast<uint8_t>(b)),
                                                dsub);
                                        if (dis < best_dis) {
                                            best_dis = dis;
                                            best_code = static_cast<uint8_t>(b);
                                        }
                                    }
                                    code_buf[m] = best_code;
                                    margin_full_scan_acc += 1.0;
                                    continue;
                                }
                                
                                // OPTIMIZATION: Adaptive k and neighbor_kk based on rough score analysis
                                int adaptive_k = k;
                                int adaptive_neighbor_kk = subspace_neighbor_kk[m];

                                if (use_adaptive_k1_margin && k == 1) {
                                    const float gap = rough_second_score[m] - rough_best_score[m];
                                    const float denom = std::fabs(rough_best_score[m]) + 1.0f;
                                    const float ratio = gap / denom;
                                    if (merge2.ivfpq_adaptive_k1_high_margin_ratio > 0.0f &&
                                        ratio >= merge2.ivfpq_adaptive_k1_high_margin_ratio) {
                                        adaptive_neighbor_kk = adaptive_k1_min_neighbor_kk;
                                    } else if (merge2.ivfpq_adaptive_k1_mid_margin_ratio > 0.0f &&
                                               ratio >= merge2.ivfpq_adaptive_k1_mid_margin_ratio) {
                                        adaptive_neighbor_kk =
                                                std::max(
                                                        adaptive_k1_min_neighbor_kk,
                                                        (neighbor_kk + adaptive_k1_min_neighbor_kk) / 2);
                                    }
                                }
                                
                                // Analyze rough score distribution to adjust exact candidates
                                if (merge2.ivfpq_adaptive_candidates && k >= 8) {
                                    // Calculate score gaps for adaptive adjustment
                                    std::vector<std::pair<float, uint8_t>> score_pairs;
                                    score_pairs.reserve(K);
                                    
                                    const float* prefix_ma = prefix_tab.data() + (m * K + old_code[m]) * K;
                                    const float* p4 = fcode.plus2_ip_new_centroid_q.data() + fcode_list_mk_index(x, m, 0, M, K);
                                    
                                    for (size_t b = 0; b < K; b++) {
                                        float score = prefix_ma[b] + p4[b];
                                        score_pairs.emplace_back(score, b);
                                    }
                                    
                                    std::sort(score_pairs.begin(), score_pairs.end());
                                    
                                    // Analyze score distribution
                                    if (k < K) {
                                        float top_k_score = score_pairs[k-1].first;
                                        float best_score = score_pairs[0].first;
                                        float total_range = score_pairs[K-1].first - best_score;
                                        
                                        if (total_range > 1e-6f) {
                                            float confidence = (top_k_score - best_score) / total_range;
                                            
                                            if (confidence > 0.8f) {
                                                // High confidence, reduce exact candidates
                                                adaptive_k = std::max(k / 2, 8);
                                                adaptive_neighbor_kk = std::max(neighbor_kk / 2, 8);
                                            } else if (confidence < 0.3f) {
                                                // Low confidence, might need more candidates (but cap it)
                                                adaptive_k = std::min(k * 3 / 2, static_cast<int>(K));
                                            }
                                        }
                                    }
                                }
                                
                                // exact candidates = union(top-k rough codes, neighbors of each rough code)
                                bool seen[256] = {false};
                                for (int ci = 0; ci < adaptive_k && ci < k; ci++) {
                                    const uint8_t c = sl[static_cast<size_t>(ci)];
                                    if (!seen[c]) {
                                        seen[c] = true;
                                        const float dis = fvec_L2sqr(res_m, pq.get_centroids(m, c), dsub);
                                        if (dis < best_dis) {
                                            best_dis = dis;
                                            best_code = c;
                                        }
                                    }
                                    if (adaptive_neighbor_kk > 0) {
                                        const uint8_t* nb = newcode_nn.data() + (m * K + c) * static_cast<size_t>(neighbor_kk);
                                        for (int u = 0; u < adaptive_neighbor_kk; u++) {
                                            const uint8_t b = nb[static_cast<size_t>(u)];
                                            if (seen[b]) {
                                                continue;
                                            }
                                            seen[b] = true;
                                            const float dis = fvec_L2sqr(res_m, pq.get_centroids(m, b), dsub);
                                            if (dis < best_dis) {
                                                best_dis = dis;
                                                best_code = b;
                                            }
                                        }
                                    }
                                }
                                code_buf[m] = best_code;
                            }
                        };
                        if (time_split) {
                            const auto t0 = std::chrono::steady_clock::now();
                            rough_pass();
                            const auto t1 = std::chrono::steady_clock::now();
                            rough_topk_acc_s += std::chrono::duration<double>(t1 - t0).count();
                            const auto t2 = std::chrono::steady_clock::now();
                            exact_pass();
                            const auto t3 = std::chrono::steady_clock::now();
                            exact_l2_acc_s += std::chrono::duration<double>(t3 - t2).count();
                        } else {
                            rough_pass();
                            exact_pass();
                        }

                        // Atomic slot claim in the pre-allocated buffer for x.
                        const size_t pos = out_pos[x].fetch_add(1, std::memory_order_relaxed);
                        out_ids[x][pos] = gid;
                        std::memcpy(out_codes[x].data() + pos * code_size, code_buf.data(), code_size);
                    }

                    if (use_rough_k1_cache) {
                        for (uint32_t x : rough_cache_unique_x) {
                            rough_cache_x_to_pos[x] = -1;
                        }
                    }

                    // Restore prefix_tab: undo (3)[global_j] so the table is
                    // ready for the next list.  Rough-score use only, so minor
                    // FP round-trip error is harmless.
                    for (size_t m = 0; m < M; m++) {
                        const float* p3 = fcode.minus2_ip_old_centroid_q.data() + fcode_list_mk_index(global_j, m, 0, M, K);
                        for (size_t a = 0; a < K; a++) {
                            float* row = prefix_tab.data() + (m * K + a) * K;
                            sub_inplace_floats_K(row, p3, K);
                        }
                    }
                }
            }

            if (measure_subcode_recall) {
#pragma omp critical
                {
                    for (int ji = 0; ji < MergeRunStats::FASTADD_SUBCODE_RECALL_NUM_K; ji++) {
                        stats->ivfpq_fast_add_subcode_recall_hits[ji] += recall_lh[ji];
                    }
                    stats->ivfpq_fast_add_subcode_recall_pairs += recall_lp;
                }
            }
            if (measure_subcode_recall_neighbors) {
#pragma omp critical
                {
                    for (int ji = 0; ji < MergeRunStats::FASTADD_NEIGHBOR_RECALL_NUM_K; ji++) {
                        stats->ivfpq_fast_add_subcode_neighbor_recall_hits[ji] += neigh_lh[ji];
                        stats->ivfpq_fast_add_subcode_neighbor_recall_union_size_sum[ji] += neigh_union_sum[ji];
                    }
                    stats->ivfpq_fast_add_subcode_neighbor_recall_pairs += neigh_lp;
                }
            }
        } // end omp parallel

        // Batched add_entries: one call per new_list (vs. one per vector).
        // Eliminates O(ntotal) dynamic push_back / realloc calls inside
        // ArrayInvertedLists.
        for (size_t x = 0; x < nlist; x++) {
            if (out_ids[x].empty())
                continue;
            ail->add_entries(x, out_ids[x].size(), out_ids[x].data(), out_codes[x].data());
        }

        if (stats) {
            stats->fast_add_num_tables = 4.0;
            stats->fast_add_full_encodes = static_cast<double>(ntotal);
            stats->fast_add_lazy_full_scans = margin_full_scan_acc;
            if (time_split) {
                stats->ivfpq_fast_add_rough_topk_s = rough_topk_acc_s;
                stats->ivfpq_fast_add_exact_l2_s = exact_l2_acc_s;
            }
        }

    } else {
        // ── Full encode path ───────────────────────────────────────────────
        // encode_vectors is read-only on index state; ail->add_entries is safe
        // because each thread owns a unique list_no.

        const size_t batch = 32768;

#pragma omp parallel
        {
            std::vector<float> xb(batch * d);
            std::vector<idx_t> ids(batch);
            std::vector<idx_t> keys(batch);
            std::vector<uint8_t> codes(batch * code_size);

#pragma omp for schedule(dynamic, 1)
            for (int ln_int = 0; ln_int < static_cast<int>(nlist); ln_int++) {
                const size_t list_no = static_cast<size_t>(ln_int);
                const auto& lst = merged_ivf.lists[list_no];
                if (lst.empty())
                    continue;

                for (size_t s = 0; s < lst.size(); s += batch) {
                    const size_t n = std::min(batch, lst.size() - s);
                    for (size_t i = 0; i < n; i++) {
                        const idx_t id = lst[s + i];
                        ids[i] = id;
                        keys[i] = static_cast<idx_t>(list_no);
                        std::memcpy(xb.data() + i * d, vectors + static_cast<size_t>(id) * d, d * sizeof(float));
                    }
                    index->encode_vectors(static_cast<idx_t>(n), xb.data(), keys.data(), codes.data(), false);
                    ail->add_entries(list_no, n, ids.data(), codes.data());
                }
            }
        } // end omp parallel
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
    if (merge_opts.ivfpq_use_subcode_remap) {
        std::fprintf(stderr, "[ivfpq_merge] merge_ivfpq: begin\n");
        std::fflush(stderr);
    }

    auto pair = concat_ivf_meta_only(indices, options.run_stats);
    IVFDataForMerge data = std::move(pair.first);
    ConcatMeta meta = std::move(pair.second);
    if (merge_opts.ivfpq_use_subcode_remap) {
        std::fprintf(stderr, "[ivfpq_merge] merge_ivfpq: concat done ntotal=%zu nlist_old=%zu d=%zu\n", data.ntotal, data.nlist, data.d);
        std::fflush(stderr);
    }

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
    if (merge_opts.ivfpq_use_subcode_remap) {
        std::fprintf(stderr, "[ivfpq_merge] merge_ivfpq: vector materialization done\n");
        std::fflush(stderr);
    }

    MergeOptions ivf_opts = merge_opts;
    if (ivf_opts.target_nlist == 0) {
        ivf_opts.target_nlist = data.nlist;
    }
    merge_ivf_data(data, ivf_opts, options.run_stats);
    if (merge_opts.ivfpq_use_subcode_remap) {
        std::fprintf(stderr, "[ivfpq_merge] merge_ivfpq: ivf merge done nlist_new=%zu\n", data.nlist);
        std::fflush(stderr);
    }

    const size_t target_M = merge_opts.target_M ? merge_opts.target_M : indices[0]->pq.M;
    const size_t target_nbits = merge_opts.target_nbits ? merge_opts.target_nbits : indices[0]->pq.nbits;

    FAISS_THROW_IF_NOT_MSG(
            have_full_raw_vectors(merge_opts, data.ntotal),
            "Merge requires full raw vectors via merge.training_vectors "
            "(n_training_vectors==ntotal) for exact PQ re-encode");

    ProductQuantizer pq(data.d, target_M, target_nbits);
    if (merge_opts.ivfpq_train_pq_on_raw_residuals) {
        train_pq_from_raw_residual_sample(
                data,
                merge_opts.training_vectors,
                target_M,
                target_nbits,
                merge_opts.pq_train_max_pts,
                options.run_stats,
                &pq);
    } else {
        train_pq_from_old_codeword_frequencies(
                indices,
                data.d,
                target_M,
                target_nbits,
                options.run_stats,
                &pq);
    }

    std::unique_ptr<IndexIVFPQ> out = build_index_from_merged_ivf_and_vectors(
            data, pq, merge_opts.training_vectors, options.run_stats, indices, &meta, merge_opts, old_centroids);

    if (options.run_stats) {
        options.run_stats->total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - overall0).count();
    }

    return out;
}

} // namespace faiss
