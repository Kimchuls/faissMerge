#include <faiss/IndexIVFRaBitQMerge.h>

#include <chrono>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexShards.h>
#include <faiss/clone_index.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/RaBitQuantizerMultiBit.h>
#include <faiss/invlists/InvertedLists.h>

namespace faiss {
namespace {

static void ensure_ivfrabitq_compatible(const IndexIVFRaBitQ& idx) {
    FAISS_THROW_IF_NOT(idx.is_trained);
    FAISS_THROW_IF_NOT(idx.quantizer);
    FAISS_THROW_IF_NOT(idx.quantizer->ntotal == idx.nlist);
    FAISS_THROW_IF_NOT(idx.invlists != nullptr);
    FAISS_THROW_IF_NOT_MSG(idx.by_residual, "IVFRaBitQ merge requires by_residual==true");
    FAISS_THROW_IF_NOT_MSG(idx.direct_map.no(), "IVFRaBitQ merge requires direct_map disabled");
    FAISS_THROW_IF_NOT_MSG(idx.metric_type == METRIC_L2, "IVFRaBitQ merge supports METRIC_L2 only");
}

static IVFDataForMerge decode_concat_to_ivf_data(
        const std::vector<IndexIVFRaBitQ*>& indices,
        bool decode_codes,
        bool keep_old_codes,
        MergeRunStats* stats) {
    auto t0 = std::chrono::steady_clock::now();
    FAISS_THROW_IF_NOT(!indices.empty());

    for (const IndexIVFRaBitQ* idx : indices) {
        ensure_ivfrabitq_compatible(*idx);
        FAISS_THROW_IF_NOT(idx->d == indices[0]->d);
        FAISS_THROW_IF_NOT(idx->metric_type == indices[0]->metric_type);
    }

    IVFDataForMerge data;
    data.d = static_cast<size_t>(indices[0]->d);
    data.metric = indices[0]->metric_type;
    for (const IndexIVFRaBitQ* idx : indices) {
        data.nlist += static_cast<size_t>(idx->nlist);
        data.ntotal += static_cast<size_t>(idx->ntotal);
    }
    const bool all_have_stored_t0 = keep_old_codes &&
            std::all_of(indices.begin(), indices.end(), [](const IndexIVFRaBitQ* idx) {
                return idx->has_complete_stored_t0();
            });

    data.centroids.resize(data.nlist * data.d);
    data.lists.assign(data.nlist, {});
    if (decode_codes) {
        data.vectors.resize(data.ntotal * data.d);
    }
    if (keep_old_codes) {
        data.old_code_size = indices[0]->code_size;
        data.old_codes.resize(data.ntotal * data.old_code_size);
        data.old_centroids.resize(data.nlist * data.d);
        data.old_list_for_id.resize(data.ntotal, -1);
        if (all_have_stored_t0) {
            data.old_t0_by_id.resize(data.ntotal, 0.0f);
        }
    }

    size_t list_off = 0;
    idx_t id_off = 0;
    std::vector<float> centroid(data.d);
    std::vector<float> decoded;

    for (const IndexIVFRaBitQ* idx : indices) {
        for (size_t l = 0; l < static_cast<size_t>(idx->nlist); l++) {
            const size_t gl = list_off + l;
            idx->quantizer->reconstruct(static_cast<idx_t>(l), centroid.data());
            std::memcpy(data.centroids.data() + gl * data.d, centroid.data(), data.d * sizeof(float));
            if (keep_old_codes) {
                std::memcpy(data.old_centroids.data() + gl * data.d, centroid.data(), data.d * sizeof(float));
            }

            const size_t list_size = idx->invlists->list_size(l);
            const std::vector<float>* list_t0 = nullptr;
            if (all_have_stored_t0) {
                list_t0 = &idx->stored_t0_by_list[l];
                FAISS_THROW_IF_NOT(list_t0->size() == list_size);
            }
            if (list_size == 0) {
                continue;
            }

            InvertedLists::ScopedIds ids(idx->invlists, l);
            std::unique_ptr<InvertedLists::ScopedCodes> scoped_codes;
            const uint8_t* list_codes = nullptr;
            if (decode_codes || keep_old_codes) {
                scoped_codes = std::make_unique<InvertedLists::ScopedCodes>(idx->invlists, l);
                list_codes = scoped_codes->get();
            }
            if (decode_codes) {
                std::vector<idx_t> list_nos(list_size, static_cast<idx_t>(l));
                decoded.resize(list_size * data.d);
                idx->decode_vectors(static_cast<idx_t>(list_size), list_codes, list_nos.data(), decoded.data());
            }

            auto& out_list = data.lists[gl];
            out_list.reserve(list_size);
            for (size_t i = 0; i < list_size; i++) {
                const idx_t gid = ids[i] + id_off;
                FAISS_THROW_IF_NOT(gid >= 0);
                FAISS_THROW_IF_NOT(static_cast<size_t>(gid) < data.ntotal);
                out_list.push_back(gid);
                if (decode_codes) {
                    std::memcpy(data.vectors.data() + static_cast<size_t>(gid) * data.d,
                                decoded.data() + i * data.d,
                                data.d * sizeof(float));
                }
                if (keep_old_codes) {
                    std::memcpy(data.old_codes.data() + static_cast<size_t>(gid) * data.old_code_size,
                                list_codes + i * idx->code_size,
                                idx->code_size);
                    data.old_list_for_id[static_cast<size_t>(gid)] = static_cast<idx_t>(gl);
                    if (list_t0) {
                        data.old_t0_by_id[static_cast<size_t>(gid)] = (*list_t0)[i];
                    }
                }
            }
        }
        list_off += static_cast<size_t>(idx->nlist);
        id_off += idx->ntotal;
    }

    if (stats) {
        auto t1 = std::chrono::steady_clock::now();
        stats->concat_s = std::chrono::duration<double>(t1 - t0).count();
        stats->extract_data_s = stats->concat_s;
    }
    return data;
}

static std::vector<idx_t> build_assign_from_lists(
        const std::vector<std::vector<idx_t>>& lists,
        size_t ntotal) {
    std::vector<idx_t> assign(ntotal, -1);
    for (size_t list_no = 0; list_no < lists.size(); list_no++) {
        for (idx_t id : lists[list_no]) {
            FAISS_THROW_IF_NOT(id >= 0);
            FAISS_THROW_IF_NOT(static_cast<size_t>(id) < ntotal);
            assign[static_cast<size_t>(id)] = static_cast<idx_t>(list_no);
        }
    }
    for (size_t id = 0; id < assign.size(); id++) {
        FAISS_THROW_IF_NOT_MSG(assign[id] >= 0, "IVFRaBitQ merge: unassigned vector id");
    }
    return assign;
}

static void exact_assign_to_final_centroids(
        IVFDataForMerge& data,
        int batch_size,
        MergeRunStats* stats) {
    auto t0 = std::chrono::steady_clock::now();
    const float* vectors = data.vectors_view ? data.vectors_view : data.vectors.data();
    FAISS_THROW_IF_NOT(vectors != nullptr);
    FAISS_THROW_IF_NOT(data.vectors_view || data.vectors.size() == data.ntotal * data.d);
    FAISS_THROW_IF_NOT(data.centroids.size() == data.nlist * data.d);

    IndexFlatL2 centroid_index(static_cast<int>(data.d));
    centroid_index.add(static_cast<idx_t>(data.nlist), data.centroids.data());

    data.lists.assign(data.nlist, {});
    const size_t bs = static_cast<size_t>(std::max(1, batch_size));
    std::vector<float> dists(bs);
    std::vector<idx_t> labels(bs);
    for (size_t start = 0; start < data.ntotal; start += bs) {
        const size_t cur = std::min(bs, data.ntotal - start);
        centroid_index.search(
                static_cast<idx_t>(cur),
                data.vectors.data() + start * data.d,
                1,
                dists.data(),
                labels.data());
        for (size_t i = 0; i < cur; i++) {
            const idx_t list_no = labels[i];
            FAISS_THROW_IF_NOT(list_no >= 0);
            FAISS_THROW_IF_NOT(static_cast<size_t>(list_no) < data.nlist);
            data.lists[static_cast<size_t>(list_no)].push_back(
                    static_cast<idx_t>(start + i));
        }
    }

    if (stats) {
        auto t1 = std::chrono::steady_clock::now();
        stats->remap_assign_write_s +=
                std::chrono::duration<double>(t1 - t0).count();
        stats->remap_full_reassign_s +=
                std::chrono::duration<double>(t1 - t0).count();
        finalize_merge_run_stats(stats);
    }
}


static std::unique_ptr<IndexIVFRaBitQ> build_rabitq_from_merged_data(
        const IVFDataForMerge& data,
        uint8_t nb_bits,
        uint8_t qb,
        MergeRunStats* stats) {
    auto t0 = std::chrono::steady_clock::now();
    const float* vectors = data.vectors_view ? data.vectors_view : data.vectors.data();
    FAISS_THROW_IF_NOT(vectors != nullptr);
    FAISS_THROW_IF_NOT(data.vectors_view || data.vectors.size() == data.ntotal * data.d);
    FAISS_THROW_IF_NOT(data.centroids.size() == data.nlist * data.d);
    FAISS_THROW_IF_NOT(data.lists.size() == data.nlist);

    auto quantizer = std::make_unique<IndexFlatL2>(static_cast<int>(data.d));
    quantizer->add(static_cast<idx_t>(data.nlist), data.centroids.data());

    auto out = std::make_unique<IndexIVFRaBitQ>(
            quantizer.get(), data.d, data.nlist, data.metric, true, nb_bits);
    out->own_fields = true;
    out->quantizer = quantizer.release();
    out->is_trained = true;
    out->qb = qb;

    const auto assign = build_assign_from_lists(data.lists, data.ntotal);
    out->add_core(
            static_cast<idx_t>(data.ntotal),
            vectors,
            nullptr,
            assign.data());

    if (stats) {
        auto t1 = std::chrono::steady_clock::now();
        stats->ivfpq_reencode_s = std::chrono::duration<double>(t1 - t0).count();
        stats->build_index_s = stats->ivfpq_reencode_s;
    }
    return out;
}

static std::unique_ptr<IndexIVFRaBitQ> build_rabitq_from_merged_data_listwise(
        const IVFDataForMerge& data,
        uint8_t nb_bits,
        uint8_t qb,
        MergeRunStats* stats) {
    auto t0 = std::chrono::steady_clock::now();
    const float* vectors = data.vectors_view ? data.vectors_view : data.vectors.data();
    FAISS_THROW_IF_NOT(vectors != nullptr);
    FAISS_THROW_IF_NOT(data.vectors_view || data.vectors.size() == data.ntotal * data.d);
    FAISS_THROW_IF_NOT(data.centroids.size() == data.nlist * data.d);
    FAISS_THROW_IF_NOT(data.lists.size() == data.nlist);

    auto quantizer = std::make_unique<IndexFlatL2>(static_cast<int>(data.d));
    quantizer->add(static_cast<idx_t>(data.nlist), data.centroids.data());

    auto out = std::make_unique<IndexIVFRaBitQ>(
            quantizer.get(), data.d, data.nlist, data.metric, true, nb_bits);
    out->own_fields = true;
    out->quantizer = quantizer.release();
    out->is_trained = true;
    out->qb = qb;

    for (size_t list_no = 0; list_no < data.nlist; list_no++) {
        const auto& ids_in = data.lists[list_no];
        const size_t n = ids_in.size();
        if (n == 0) {
            continue;
        }
        std::vector<idx_t> ids(n);
        std::vector<idx_t> list_nos(n, static_cast<idx_t>(list_no));
        std::vector<float> x(n * data.d);
        auto t_gather0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < n; i++) {
            const idx_t id = ids_in[i];
            FAISS_THROW_IF_NOT(id >= 0);
            FAISS_THROW_IF_NOT(static_cast<size_t>(id) < data.ntotal);
            ids[i] = id;
            std::memcpy(
                    x.data() + i * data.d,
                    vectors + static_cast<size_t>(id) * data.d,
                    data.d * sizeof(float));
        }
        auto t_gather1 = std::chrono::steady_clock::now();
        std::vector<uint8_t> codes(n * out->code_size);
        auto t_encode0 = std::chrono::steady_clock::now();
        out->encode_vectors(
                static_cast<idx_t>(n),
                x.data(),
                list_nos.data(),
                codes.data(),
                false);
        auto t_encode1 = std::chrono::steady_clock::now();
        auto t_add0 = std::chrono::steady_clock::now();
        out->invlists->add_entries(list_no, n, ids.data(), codes.data());
        auto t_add1 = std::chrono::steady_clock::now();
        if (stats) {
            stats->ivfpq_reencode_id_map_s +=
                    std::chrono::duration<double>(t_gather1 - t_gather0).count();
            stats->ivfpq_reencode_encode_codes_s +=
                    std::chrono::duration<double>(t_encode1 - t_encode0).count();
            stats->ivfpq_reencode_add_entries_s +=
                    std::chrono::duration<double>(t_add1 - t_add0).count();
        }
    }
    out->ntotal = static_cast<idx_t>(data.ntotal);

    if (stats) {
        auto t1 = std::chrono::steady_clock::now();
        stats->ivfpq_reencode_s = std::chrono::duration<double>(t1 - t0).count();
        stats->build_index_s = stats->ivfpq_reencode_s;
    }
    return out;
}


static std::unique_ptr<IndexIVFRaBitQ> build_rabitq_from_merged_data_direct_1bit(
        const IVFDataForMerge& data,
        uint8_t qb,
        MergeRunStats* stats) {
    auto t0 = std::chrono::steady_clock::now();
    FAISS_THROW_IF_NOT(data.metric == METRIC_L2);
    const float* vectors = data.vectors_view ? data.vectors_view : data.vectors.data();
    FAISS_THROW_IF_NOT(vectors != nullptr);
    FAISS_THROW_IF_NOT(data.vectors_view || data.vectors.size() == data.ntotal * data.d);
    FAISS_THROW_IF_NOT(data.centroids.size() == data.nlist * data.d);
    FAISS_THROW_IF_NOT(data.lists.size() == data.nlist);

    auto quantizer = std::make_unique<IndexFlatL2>(static_cast<int>(data.d));
    quantizer->add(static_cast<idx_t>(data.nlist), data.centroids.data());

    auto out = std::make_unique<IndexIVFRaBitQ>(
            quantizer.get(), data.d, data.nlist, data.metric, true, 1);
    out->own_fields = true;
    out->quantizer = quantizer.release();
    out->is_trained = true;
    out->qb = qb;

    const float inv_d_sqrt = data.d == 0
            ? 1.0f
            : 1.0f / std::sqrt(static_cast<float>(data.d));
    constexpr float epsilon = std::numeric_limits<float>::epsilon();

    for (size_t list_no = 0; list_no < data.nlist; list_no++) {
        const auto& ids_in = data.lists[list_no];
        const size_t n = ids_in.size();
        if (n == 0) {
            continue;
        }
        std::vector<idx_t> ids(n);
        std::vector<uint8_t> codes(n * out->code_size);
        const float* centroid = data.centroids.data() + list_no * data.d;
        auto t_encode0 = std::chrono::steady_clock::now();

#pragma omp parallel for if (n > 1000)
        for (int64_t ii = 0; ii < static_cast<int64_t>(n); ii++) {
            const size_t i = static_cast<size_t>(ii);
            const idx_t id = ids_in[i];
            FAISS_THROW_IF_NOT(id >= 0);
            FAISS_THROW_IF_NOT(static_cast<size_t>(id) < data.ntotal);
            ids[i] = id;

            const float* x = vectors + static_cast<size_t>(id) * data.d;
            uint8_t* code = codes.data() + i * out->code_size;

            std::memset(code, 0, out->code_size);

            float norm_L2sqr = 0.0f;
            float dp_oO = 0.0f;
            for (size_t j = 0; j < data.d; j++) {
                const float or_minus_c = x[j] - centroid[j];
                norm_L2sqr += or_minus_c * or_minus_c;
                if (or_minus_c > 0.0f) {
                    dp_oO += or_minus_c;
                    code[j >> 3] |= static_cast<uint8_t>(1u << (j & 7));
                } else {
                    dp_oO -= or_minus_c;
                }
            }

            const float sqrt_norm_L2 = std::sqrt(norm_L2sqr);
            const float inv_norm_L2 =
                    (norm_L2sqr < epsilon) ? 1.0f : (1.0f / sqrt_norm_L2);
            const float normalized_dp = dp_oO * inv_norm_L2 * inv_d_sqrt;
            const float inv_dp_oO =
                    (std::abs(normalized_dp) < epsilon) ? 1.0f : (1.0f / normalized_dp);

            auto* factors = reinterpret_cast<rabitq_utils::SignBitFactors*>(
                    code + (data.d + 7) / 8);
            factors->or_minus_c_l2sqr = norm_L2sqr;
            factors->dp_multiplier = inv_dp_oO * sqrt_norm_L2;
        }
        auto t_encode1 = std::chrono::steady_clock::now();
        auto t_add0 = std::chrono::steady_clock::now();
        out->invlists->add_entries(list_no, n, ids.data(), codes.data());
        auto t_add1 = std::chrono::steady_clock::now();
        if (stats) {
            stats->ivfpq_reencode_encode_codes_s +=
                    std::chrono::duration<double>(t_encode1 - t_encode0).count();
            stats->ivfpq_reencode_add_entries_s +=
                    std::chrono::duration<double>(t_add1 - t_add0).count();
        }
    }
    out->ntotal = static_cast<idx_t>(data.ntotal);

    if (stats) {
        auto t1 = std::chrono::steady_clock::now();
        stats->ivfpq_reencode_s = std::chrono::duration<double>(t1 - t0).count();
        stats->build_index_s = stats->ivfpq_reencode_s;
    }
    return out;
}


struct InRemapRaBitQEncodeContext {
    size_t d = 0;
    size_t nlist = 0;
    size_t ntotal = 0;
    size_t code_size = 0;
    float inv_d_sqrt = 1.0f;
    std::vector<std::vector<idx_t>> ids_by_list;
    std::vector<std::vector<uint8_t>> codes_by_list;
    size_t encoded = 0;
    double encode_s = 0.0;
};

static void encode_in_remap_rabitq_1bit_callback(
        void* user_data,
        const idx_t* ids,
        const float* vectors,
        const idx_t* target_lists,
        const float* target_centroids,
        size_t n) {
    auto* ctx = reinterpret_cast<InRemapRaBitQEncodeContext*>(user_data);
    FAISS_THROW_IF_NOT(ctx != nullptr);
    FAISS_THROW_IF_NOT(ids != nullptr);
    FAISS_THROW_IF_NOT(vectors != nullptr);
    FAISS_THROW_IF_NOT(target_lists != nullptr);
    FAISS_THROW_IF_NOT(target_centroids != nullptr);
    constexpr float epsilon = std::numeric_limits<float>::epsilon();

    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < n; i++) {
        const idx_t id = ids[i];
        const idx_t target = target_lists[i];
        FAISS_THROW_IF_NOT(id >= 0);
        FAISS_THROW_IF_NOT(target >= 0);
        const size_t list_no = static_cast<size_t>(target);
        FAISS_THROW_IF_NOT(list_no < ctx->nlist);

        auto& out_ids = ctx->ids_by_list[list_no];
        auto& out_codes = ctx->codes_by_list[list_no];
        out_ids.push_back(id);
        const size_t code_off = out_codes.size();
        out_codes.resize(code_off + ctx->code_size);
        uint8_t* code = out_codes.data() + code_off;
        std::memset(code, 0, ctx->code_size);

        const float* x = vectors + i * ctx->d;
        const float* centroid = target_centroids + list_no * ctx->d;
        float norm_L2sqr = 0.0f;
        float dp_oO = 0.0f;
        for (size_t j = 0; j < ctx->d; j++) {
            const float or_minus_c = x[j] - centroid[j];
            norm_L2sqr += or_minus_c * or_minus_c;
            if (or_minus_c > 0.0f) {
                dp_oO += or_minus_c;
                code[j >> 3] |= static_cast<uint8_t>(1u << (j & 7));
            } else {
                dp_oO -= or_minus_c;
            }
        }

        const float sqrt_norm_L2 = std::sqrt(norm_L2sqr);
        const float inv_norm_L2 =
                (norm_L2sqr < epsilon) ? 1.0f : (1.0f / sqrt_norm_L2);
        const float normalized_dp = dp_oO * inv_norm_L2 * ctx->inv_d_sqrt;
        const float inv_dp_oO =
                (std::abs(normalized_dp) < epsilon) ? 1.0f : (1.0f / normalized_dp);
        auto* factors = reinterpret_cast<rabitq_utils::SignBitFactors*>(
                code + (ctx->d + 7) / 8);
        factors->or_minus_c_l2sqr = norm_L2sqr;
        factors->dp_multiplier = inv_dp_oO * sqrt_norm_L2;
        ctx->encoded++;
    }
    const auto t1 = std::chrono::steady_clock::now();
    ctx->encode_s += std::chrono::duration<double>(t1 - t0).count();
}

static std::unique_ptr<IndexIVFRaBitQ> build_rabitq_from_in_remap_buffers_direct_1bit(
        const IVFDataForMerge& data,
        InRemapRaBitQEncodeContext& ctx,
        uint8_t qb,
        MergeRunStats* stats) {
    auto t0 = std::chrono::steady_clock::now();
    FAISS_THROW_IF_NOT(ctx.encoded == data.ntotal);
    FAISS_THROW_IF_NOT(ctx.nlist == data.nlist);
    FAISS_THROW_IF_NOT(ctx.d == data.d);
    FAISS_THROW_IF_NOT(data.centroids.size() == data.nlist * data.d);

    auto quantizer = std::make_unique<IndexFlatL2>(static_cast<int>(data.d));
    quantizer->add(static_cast<idx_t>(data.nlist), data.centroids.data());

    auto out = std::make_unique<IndexIVFRaBitQ>(
            quantizer.get(), data.d, data.nlist, data.metric, true, 1);
    out->own_fields = true;
    out->quantizer = quantizer.release();
    out->is_trained = true;
    out->qb = qb;
    FAISS_THROW_IF_NOT(out->code_size == ctx.code_size);

    auto t_add0 = std::chrono::steady_clock::now();
    for (size_t list_no = 0; list_no < data.nlist; list_no++) {
        const size_t n = ctx.ids_by_list[list_no].size();
        if (n == 0) {
            continue;
        }
        FAISS_THROW_IF_NOT(ctx.codes_by_list[list_no].size() == n * out->code_size);
        out->invlists->add_entries(
                list_no,
                n,
                ctx.ids_by_list[list_no].data(),
                ctx.codes_by_list[list_no].data());
    }
    auto t_add1 = std::chrono::steady_clock::now();
    out->ntotal = static_cast<idx_t>(data.ntotal);

    if (stats) {
        auto t1 = std::chrono::steady_clock::now();
        stats->ivfpq_reencode_encode_codes_s += ctx.encode_s;
        stats->ivfpq_reencode_add_entries_s +=
                std::chrono::duration<double>(t_add1 - t_add0).count();
        stats->ivfpq_reencode_s = std::chrono::duration<double>(t1 - t0).count();
        stats->build_index_s = stats->ivfpq_reencode_s;
    }
    return out;
}

static std::unique_ptr<IndexIVFRaBitQ> build_rabitq_from_source_lists_final_assign_direct_1bit(
        const IVFDataForMerge& data,
        const std::vector<std::vector<idx_t>>& source_lists,
        uint8_t qb,
        MergeRunStats* stats) {
    auto t0 = std::chrono::steady_clock::now();
    FAISS_THROW_IF_NOT(data.metric == METRIC_L2);
    const float* vectors = data.vectors_view ? data.vectors_view : data.vectors.data();
    FAISS_THROW_IF_NOT(vectors != nullptr);
    FAISS_THROW_IF_NOT(data.vectors_view || data.vectors.size() == data.ntotal * data.d);
    FAISS_THROW_IF_NOT(data.centroids.size() == data.nlist * data.d);
    FAISS_THROW_IF_NOT(data.final_assign.size() == data.ntotal);

    auto quantizer = std::make_unique<IndexFlatL2>(static_cast<int>(data.d));
    quantizer->add(static_cast<idx_t>(data.nlist), data.centroids.data());

    auto out = std::make_unique<IndexIVFRaBitQ>(
            quantizer.get(), data.d, data.nlist, data.metric, true, 1);
    out->own_fields = true;
    out->quantizer = quantizer.release();
    out->is_trained = true;
    out->qb = qb;

    const float inv_d_sqrt = data.d == 0
            ? 1.0f
            : 1.0f / std::sqrt(static_cast<float>(data.d));
    constexpr float epsilon = std::numeric_limits<float>::epsilon();

    std::vector<size_t> counts(data.nlist, 0);
    for (idx_t list_no : data.final_assign) {
        FAISS_THROW_IF_NOT(list_no >= 0);
        FAISS_THROW_IF_NOT(static_cast<size_t>(list_no) < data.nlist);
        counts[static_cast<size_t>(list_no)]++;
    }

    std::vector<std::vector<idx_t>> ids_by_list(data.nlist);
    std::vector<std::vector<uint8_t>> codes_by_list(data.nlist);
    for (size_t list_no = 0; list_no < data.nlist; list_no++) {
        ids_by_list[list_no].resize(counts[list_no]);
        codes_by_list[list_no].resize(counts[list_no] * out->code_size);
    }

    std::vector<size_t> offsets(data.nlist, 0);
    auto t_encode0 = std::chrono::steady_clock::now();
    for (const auto& src_list : source_lists) {
        for (idx_t id : src_list) {
            FAISS_THROW_IF_NOT(id >= 0);
            const size_t id_sz = static_cast<size_t>(id);
            FAISS_THROW_IF_NOT(id_sz < data.ntotal);
            const size_t list_no = static_cast<size_t>(data.final_assign[id_sz]);
            FAISS_THROW_IF_NOT(list_no < data.nlist);
            const size_t pos = offsets[list_no]++;
            ids_by_list[list_no][pos] = id;

            const float* centroid = data.centroids.data() + list_no * data.d;
            const float* x = vectors + id_sz * data.d;
            uint8_t* code = codes_by_list[list_no].data() + pos * out->code_size;
            std::memset(code, 0, out->code_size);

            float norm_L2sqr = 0.0f;
            float dp_oO = 0.0f;
            for (size_t j = 0; j < data.d; j++) {
                const float or_minus_c = x[j] - centroid[j];
                norm_L2sqr += or_minus_c * or_minus_c;
                if (or_minus_c > 0.0f) {
                    dp_oO += or_minus_c;
                    code[j >> 3] |= static_cast<uint8_t>(1u << (j & 7));
                } else {
                    dp_oO -= or_minus_c;
                }
            }

            const float sqrt_norm_L2 = std::sqrt(norm_L2sqr);
            const float inv_norm_L2 =
                    (norm_L2sqr < epsilon) ? 1.0f : (1.0f / sqrt_norm_L2);
            const float normalized_dp = dp_oO * inv_norm_L2 * inv_d_sqrt;
            const float inv_dp_oO =
                    (std::abs(normalized_dp) < epsilon) ? 1.0f : (1.0f / normalized_dp);

            auto* factors = reinterpret_cast<rabitq_utils::SignBitFactors*>(
                    code + (data.d + 7) / 8);
            factors->or_minus_c_l2sqr = norm_L2sqr;
            factors->dp_multiplier = inv_dp_oO * sqrt_norm_L2;
        }
    }
    auto t_encode1 = std::chrono::steady_clock::now();

    auto t_add0 = std::chrono::steady_clock::now();
    for (size_t list_no = 0; list_no < data.nlist; list_no++) {
        const size_t n = counts[list_no];
        FAISS_THROW_IF_NOT(offsets[list_no] == n);
        if (n == 0) {
            continue;
        }
        out->invlists->add_entries(
                list_no,
                n,
                ids_by_list[list_no].data(),
                codes_by_list[list_no].data());
    }
    auto t_add1 = std::chrono::steady_clock::now();
    out->ntotal = static_cast<idx_t>(data.ntotal);

    if (stats) {
        auto t1 = std::chrono::steady_clock::now();
        stats->ivfpq_reencode_encode_codes_s +=
                std::chrono::duration<double>(t_encode1 - t_encode0).count();
        stats->ivfpq_reencode_add_entries_s +=
                std::chrono::duration<double>(t_add1 - t_add0).count();
        stats->ivfpq_reencode_s = std::chrono::duration<double>(t1 - t0).count();
        stats->build_index_s = stats->ivfpq_reencode_s;
    }
    return out;
}



static void unpack_old_full_codes(
        const uint8_t* old_code,
        size_t d,
        uint8_t nb_bits,
        std::vector<int>& full_code) {
    full_code.resize(d);
    const size_t sign_bytes = (d + 7) / 8;
    const size_t ex_bits = nb_bits - 1;
    if (ex_bits == 0) {
        for (size_t j = 0; j < d; j++) {
            full_code[j] = rabitq_utils::extract_bit_standard(old_code, j) ? 1 : 0;
        }
        return;
    }
    const uint8_t* ex_code = old_code + sign_bytes + sizeof(rabitq_utils::SignBitFactorsWithError);
    const int half = 1 << ex_bits;
    for (size_t j = 0; j < d; j++) {
        const int ex = rabitq_utils::extract_code_inline(ex_code, j, ex_bits);
        const bool sign = rabitq_utils::extract_bit_standard(old_code, j);
        full_code[j] = ex + (sign ? half : 0);
    }
}

static float residual_norm_from_ptr(const float* residual, size_t d) {
    double s = 0.0;
    for (size_t i = 0; i < d; i++) {
        s += static_cast<double>(residual[i]) * residual[i];
    }
    return static_cast<float>(std::sqrt(s));
}

static float old_state_objective(
        const std::vector<int>& full_code,
        const float* residual,
        size_t d,
        uint8_t nb_bits) {
    const int max_code = (1 << nb_bits) - 1;
    const double center = static_cast<double>(max_code) * 0.5;
    double dot = 0.0;
    double qnorm2 = 0.0;
    double rnorm2 = 0.0;
    for (size_t i = 0; i < d; i++) {
        const double q = static_cast<double>(full_code[i]) - center;
        const double r = residual[i];
        dot += q * r;
        qnorm2 += q * q;
        rnorm2 += r * r;
    }
    return (qnorm2 > 0.0 && rnorm2 > 0.0)
            ? static_cast<float>(dot / std::sqrt(qnorm2 * rnorm2))
            : 0.0f;
}

struct TDigestStats {
    size_t n = 0;
    double sum = 0.0;
    std::vector<double> values;

    void add(double v) {
        if (!std::isfinite(v)) {
            return;
        }
        n++;
        sum += v;
        values.push_back(v);
    }

    double mean() const {
        return n == 0 ? 0.0 : sum / static_cast<double>(n);
    }

    double q(double p) {
        if (values.empty()) {
            return 0.0;
        }
        std::sort(values.begin(), values.end());
        const double pos = p * static_cast<double>(values.size() - 1);
        const size_t lo = static_cast<size_t>(std::floor(pos));
        const size_t hi = std::min(values.size() - 1, lo + 1);
        const double a = pos - static_cast<double>(lo);
        return values[lo] * (1.0 - a) + values[hi] * a;
    }
};

static void append_stat_json(std::ostringstream& os, const char* name, TDigestStats& st) {
    os << "    \"" << name << "\": {"
       << "\"n\": " << st.n
       << ", \"mean\": " << st.mean()
       << ", \"p50\": " << st.q(0.50)
       << ", \"p90\": " << st.q(0.90)
       << ", \"p95\": " << st.q(0.95)
       << ", \"p99\": " << st.q(0.99)
       << "}";
}

static void make_tpred_full_code(
        const float* residual,
        size_t d,
        uint8_t nb_bits,
        float tpred,
        std::vector<int>& full_code) {
    full_code.resize(d);
    if (nb_bits == 1) {
        for (size_t i = 0; i < d; i++) {
            full_code[i] = residual[i] >= 0.0f ? 1 : 0;
        }
        return;
    }
    const int ex_bits = nb_bits - 1;
    const int half = 1 << ex_bits;
    const int max_ex = half - 1;
    const float norm = residual_norm_from_ptr(residual, d);
    for (size_t i = 0; i < d; i++) {
        int mag = 0;
        if (norm > 1e-10f) {
            mag = std::min(
                    static_cast<int>(tpred * std::abs(residual[i]) / norm + 1e-5f),
                    max_ex);
        }
        full_code[i] = residual[i] >= 0.0f ? half + mag : ((~mag) & max_ex);
    }
}

struct AdaptiveTResult {
    float t = 0.0f;
    double objective = 0.0;
    int steps = 0;
    int evaluations = 0;
    bool hit_limit = false;
};

static AdaptiveTResult adaptive_t_search(
        const float* residual,
        size_t d,
        uint8_t nb_bits,
        float start_t,
        float factor,
        int max_steps,
        int patience,
        double min_window_gain,
        std::vector<int>& scratch) {
    auto evaluate = [&](float t) {
        make_tpred_full_code(residual, d, nb_bits, t, scratch);
        return static_cast<double>(old_state_objective(scratch, residual, d, nb_bits));
    };
    AdaptiveTResult out;
    if (!(start_t > 0.0f) || !(factor > 1.0f) || max_steps <= 0) {
        return out;
    }
    out.t = start_t;
    out.objective = evaluate(start_t);
    out.evaluations = 1;
    const double lower_j = evaluate(start_t / factor);
    const double upper_j = evaluate(start_t * factor);
    out.evaluations += 2;
    int direction = 0;
    if (lower_j > out.objective || upper_j > out.objective) {
        direction = lower_j > upper_j ? -1 : 1;
        out.t = direction < 0 ? start_t / factor : start_t * factor;
        out.objective = std::max(lower_j, upper_j);
        out.steps = 1;
    }
    if (direction == 0) {
        return out;
    }
    double window_best = out.objective;
    int window_steps = 0;
    for (int radius = 2; radius <= max_steps; ++radius) {
        const float candidate_t = start_t * std::pow(factor, direction * radius);
        const double candidate_j = evaluate(candidate_t);
        ++out.evaluations;
        if (candidate_j > out.objective) {
            out.objective = candidate_j;
            out.t = candidate_t;
            out.steps = radius;
        }
        ++window_steps;
        if (window_steps >= patience) {
            if (out.objective - window_best < min_window_gain) {
                return out;
            }
            window_best = out.objective;
            window_steps = 0;
        }
    }
    out.hit_limit = true;
    return out;
}

static void pack_full_code_for_rabitq(
        const float* x,
        const float* centroid,
        const std::vector<int>& full_code,
        size_t d,
        uint8_t nb_bits,
        MetricType metric,
        uint8_t* out_code,
        std::vector<int>& ex_tmp,
        std::vector<float>& residual) {
    const size_t sign_bytes = (d + 7) / 8;
    const size_t ex_bits = nb_bits - 1;
    const int half = 1 << ex_bits;
    std::memset(out_code, 0, sign_bytes);
    residual.resize(d);

    float norm_l2sqr = 0.0f;
    float or_l2sqr = 0.0f;
    float dp_oo = 0.0f;
    for (size_t j = 0; j < d; j++) {
        const float r = x[j] - centroid[j];
        residual[j] = r;
        norm_l2sqr += r * r;
        or_l2sqr += x[j] * x[j];
        if (r > 0.0f) {
            dp_oo += r;
        } else {
            dp_oo -= r;
        }
        if (full_code[j] >= half) {
            rabitq_utils::set_bit_standard(out_code, j);
        }
    }

    auto factors = rabitq_utils::compute_factors_from_intermediates(
            norm_l2sqr,
            or_l2sqr,
            dp_oo,
            d,
            metric,
            ex_bits > 0);

    if (ex_bits == 0) {
        auto* base_factors = reinterpret_cast<rabitq_utils::SignBitFactors*>(
                out_code + sign_bytes);
        base_factors->or_minus_c_l2sqr = factors.or_minus_c_l2sqr;
        base_factors->dp_multiplier = factors.dp_multiplier;
        return;
    }

    auto* base_factors = reinterpret_cast<rabitq_utils::SignBitFactorsWithError*>(
            out_code + sign_bytes);
    *base_factors = factors;

    uint8_t* ex_code = out_code + sign_bytes + sizeof(rabitq_utils::SignBitFactorsWithError);
    auto* ex_factors = reinterpret_cast<rabitq_utils::ExtraBitsFactors*>(
            ex_code + (d * ex_bits + 7) / 8);
    ex_tmp.resize(d);
    const int max_ex = (1 << ex_bits) - 1;
    const float norm = std::sqrt(norm_l2sqr);
    double ipnorm = 0.0;
    if (norm > 1e-10f) {
        for (size_t j = 0; j < d; j++) {
            const int ex = full_code[j] & max_ex;
            ex_tmp[j] = ex;
            const int mag = residual[j] >= 0.0f ? ex : ((~ex) & max_ex);
            ipnorm += (static_cast<double>(mag) + 0.5) * std::abs(residual[j]) / norm;
        }
    } else {
        std::fill(ex_tmp.begin(), ex_tmp.end(), 0);
    }
    rabitq_multibit::pack_multibit_codes(ex_tmp.data(), ex_code, d, nb_bits);
    rabitq_multibit::compute_ex_factors(
            residual.data(),
            centroid,
            d,
            norm,
            ipnorm,
            *ex_factors,
            metric);
}

static float const_median_t_for_dim_bits(size_t d, uint8_t nb_bits) {
    if (d == 96) {
        if (nb_bits == 2) {
            return 12.09640837f;
        }
        if (nb_bits == 4) {
            return 35.64824677f;
        }
        if (nb_bits == 8) {
            return 386.3062744f;
        }
    }
    return 0.0f;
}

static float caq_tight_start_t_for_residual(
        const float* residual,
        size_t d,
        uint8_t nb_bits) {
    static constexpr float k_tight_start[9] =
            {0.0f, 0.15f, 0.20f, 0.52f, 0.59f, 0.71f, 0.75f, 0.77f, 0.81f};
    if (nb_bits <= 1 || nb_bits > 9) {
        return 0.0f;
    }
    const size_t ex_bits = nb_bits - 1;
    const int max_code = (1 << ex_bits) - 1;
    const float norm = residual_norm_from_ptr(residual, d);
    if (!(norm > 1e-10f)) {
        return 0.0f;
    }
    float max_z = 0.0f;
    for (size_t i = 0; i < d; i++) {
        max_z = std::max(max_z, std::abs(residual[i]) / norm);
    }
    if (!(max_z > 0.0f)) {
        return 0.0f;
    }
    const float t_end = static_cast<float>(max_code + 10) / max_z;
    return t_end * k_tight_start[ex_bits];
}

static void encode_old_state_k0_code(
        const float* x,
        const float* old_centroid,
        const float* new_centroid,
        const uint8_t* old_code,
        float precomputed_old_t0,
        size_t d,
        uint8_t nb_bits,
        MetricType metric,
        bool mirror_only,
        const std::string& t_init_mode,
        int local_t_steps,
        float local_t_step,
        bool adaptive_t,
        float adaptive_t_factor,
        int adaptive_t_max_steps,
        int adaptive_t_patience,
        float adaptive_t_min_gain,
        uint8_t* out_code,
        std::vector<int>& old_full_code,
        std::vector<int>& mirror_code,
        std::vector<int>& tpred_code,
        std::vector<int>& ex_tmp,
        std::vector<float>& old_residual,
        std::vector<float>& new_residual) {
    unpack_old_full_codes(old_code, d, nb_bits, old_full_code);
    old_residual.resize(d);
    new_residual.resize(d);
    for (size_t j = 0; j < d; j++) {
        old_residual[j] = x[j] - old_centroid[j];
        new_residual[j] = x[j] - new_centroid[j];
    }

    mirror_code = old_full_code;
    const int max_full = (1 << nb_bits) - 1;
    const int half = 1 << (nb_bits - 1);
    for (size_t j = 0; j < d; j++) {
        const bool want_pos = new_residual[j] >= 0.0f;
        const bool have_pos = mirror_code[j] >= half;
        if (want_pos != have_pos) {
            mirror_code[j] = max_full - mirror_code[j];
        }
    }

    const float old_norm = residual_norm_from_ptr(old_residual.data(), d);
    const float new_norm = residual_norm_from_ptr(new_residual.data(), d);
    float tpred = 0.0f;
    if (nb_bits > 1 && old_norm > 1e-10f && new_norm > 1e-10f) {
        float old_t0 = precomputed_old_t0;
        if (!(old_t0 > 0.0f)) {
            std::vector<float> old_abs(d);
            for (size_t j = 0; j < d; j++) {
                old_abs[j] = std::abs(old_residual[j]) / old_norm;
            }
            old_t0 = rabitq_multibit::compute_optimal_scaling_factor(
                    old_abs.data(), d, nb_bits);
        }
        if (t_init_mode == "old_t") {
            tpred = old_t0;
        } else if (t_init_mode == "const_median_048") {
            tpred = const_median_t_for_dim_bits(d, nb_bits);
        } else if (t_init_mode == "caq_tight_start") {
            tpred = caq_tight_start_t_for_residual(new_residual.data(), d, nb_bits);
        } else {
            tpred = old_t0 / old_norm * new_norm;
        }
    }
    make_tpred_full_code(new_residual.data(), d, nb_bits, tpred, tpred_code);

    const std::vector<int>* chosen_ptr = &mirror_code;
    if (!mirror_only) {
        const float mirror_j = old_state_objective(mirror_code, new_residual.data(), d, nb_bits);
        const float tpred_j = old_state_objective(tpred_code, new_residual.data(), d, nb_bits);
        float best_j = mirror_j;
        if (tpred_j > best_j) {
            best_j = tpred_j;
            mirror_code = tpred_code;
        }
        // Keep the selected code stable while tpred_code is reused as scratch.
        chosen_ptr = &mirror_code;
        if (nb_bits > 1 && tpred > 0.0f && adaptive_t) {
            AdaptiveTResult adaptive = adaptive_t_search(
                    new_residual.data(), d, nb_bits, tpred,
                    adaptive_t_factor, adaptive_t_max_steps,
                    adaptive_t_patience, adaptive_t_min_gain, tpred_code);
            if (adaptive.objective > mirror_j) {
                make_tpred_full_code(
                        new_residual.data(), d, nb_bits, adaptive.t, mirror_code);
            }
            chosen_ptr = &mirror_code;
        } else if (nb_bits > 1 && tpred > 0.0f && local_t_steps > 0 && local_t_step > 0.0f) {
            auto evaluate_scale = [&](int step, float sign) {
                const float scale = 1.0f + sign * local_t_step * step;
                if (scale <= 0.0f) {
                    return -std::numeric_limits<float>::infinity();
                }
                make_tpred_full_code(
                        new_residual.data(), d, nb_bits, tpred * scale, tpred_code);
                const float cand_j = old_state_objective(
                        tpred_code, new_residual.data(), d, nb_bits);
                if (cand_j > best_j) {
                    best_j = cand_j;
                    mirror_code = tpred_code;
                }
                return cand_j;
            };

            // Probe two steps on both sides, then spend the remaining budget
            // only in the direction with the better window candidate.
            float left_best = -std::numeric_limits<float>::infinity();
            float right_best = -std::numeric_limits<float>::infinity();
            const int probe_steps = std::min(local_t_steps, 2);
            for (int step = 1; step <= probe_steps; step++) {
                left_best = std::max(left_best, evaluate_scale(step, -1.0f));
                right_best = std::max(right_best, evaluate_scale(step, 1.0f));
            }
            if (local_t_steps > probe_steps) {
                const float direction = right_best > left_best ? 1.0f : -1.0f;
                for (int step = probe_steps + 1; step <= local_t_steps; step++) {
                    evaluate_scale(step, direction);
                }
            }
        }
    }
    const std::vector<int>& chosen = *chosen_ptr;
    pack_full_code_for_rabitq(
            x,
            new_centroid,
            chosen,
            d,
            nb_bits,
            metric,
            out_code,
            ex_tmp,
            new_residual);
}

static void write_rabitq_t_diagnostic(
        const IVFDataForMerge& data,
        uint8_t nb_bits,
        const std::string& path,
        size_t max_vectors,
        const float* queries,
        size_t n_queries) {
    if (path.empty() || nb_bits <= 1) {
        return;
    }
    const float* vectors = data.vectors_view ? data.vectors_view : data.vectors.data();
    FAISS_THROW_IF_NOT(vectors != nullptr);
    FAISS_THROW_IF_NOT(data.old_codes.size() == data.ntotal * data.old_code_size);
    FAISS_THROW_IF_NOT(data.old_centroids.size() % data.d == 0);
    FAISS_THROW_IF_NOT(data.old_list_for_id.size() == data.ntotal);
    FAISS_THROW_IF_NOT(data.old_t0_by_id.empty() || data.old_t0_by_id.size() == data.ntotal);
    FAISS_THROW_IF_NOT(data.centroids.size() == data.nlist * data.d);

    std::vector<idx_t> assign = build_assign_from_lists(data.lists, data.ntotal);
    std::vector<float> old_residual(data.d), new_residual(data.d), abs_norm(data.d);
    std::vector<int> pred_code, raw_code, cand_code, best_local_code;
    TDigestStats hop, frac_hop, abs_log_ratio, j_ratio, pred_j, raw_j, cos_gap, shift_ratio;
    TDigestStats old_t_gap, old_t_hop, old_t_abslog;
    TDigestStats const_median_gap, const_median_hop, const_median_abslog;
    TDigestStats const_half_gap, const_half_hop, const_half_abslog;
    TDigestStats const_double_gap, const_double_hop, const_double_abslog;
    TDigestStats signed_log_ratio, pred_hit_1e3, pred_hit_1e4, pred_left_of_opt;
    TDigestStats adapt1_gap, adapt1_steps, adapt1_evals, adapt1_hit_1e3, adapt1_hit_1e4, adapt1_limit;
    TDigestStats adapt3_gap, adapt3_steps, adapt3_evals, adapt3_hit_1e3, adapt3_hit_1e4, adapt3_limit;
    TDigestStats adapt4_gap, adapt4_steps, adapt4_evals, adapt4_hit_1e3, adapt4_hit_1e4, adapt4_limit;
    const std::vector<int> local_range_steps{0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32};
    const float local_range_step_size = 1.0f / 128.0f;
    std::vector<TDigestStats> local_range_gap(local_range_steps.size());
    std::vector<TDigestStats> local_range_relative_gap(local_range_steps.size());
    std::vector<TDigestStats> local_range_distance_are(local_range_steps.size());
    TDigestStats optimal_distance_are;
    size_t seen = 0;
    const size_t stride = (max_vectors > 0 && data.ntotal > max_vectors)
            ? std::max<size_t>(1, data.ntotal / max_vectors)
            : 1;
    for (size_t id = 0; id < data.ntotal; id += stride) {
        if (max_vectors > 0 && seen >= max_vectors) {
            break;
        }
        const idx_t new_list = assign[id];
        const idx_t old_list = data.old_list_for_id[id];
        if (new_list < 0 || old_list < 0) {
            continue;
        }
        const float* x = vectors + id * data.d;
        const float* old_c = data.old_centroids.data() + static_cast<size_t>(old_list) * data.d;
        const float* new_c = data.centroids.data() + static_cast<size_t>(new_list) * data.d;
        double old_norm2 = 0.0, new_norm2 = 0.0, shift_norm2 = 0.0;
        for (size_t j = 0; j < data.d; j++) {
            old_residual[j] = x[j] - old_c[j];
            new_residual[j] = x[j] - new_c[j];
            old_norm2 += static_cast<double>(old_residual[j]) * old_residual[j];
            new_norm2 += static_cast<double>(new_residual[j]) * new_residual[j];
            const double dc = static_cast<double>(old_c[j]) - new_c[j];
            shift_norm2 += dc * dc;
        }
        const float old_norm = static_cast<float>(std::sqrt(old_norm2));
        const float new_norm = static_cast<float>(std::sqrt(new_norm2));
        if (!(old_norm > 1e-10f) || !(new_norm > 1e-10f)) {
            continue;
        }
        float old_t0 = data.old_t0_by_id.empty() ? 0.0f : data.old_t0_by_id[id];
        if (!(old_t0 > 0.0f)) {
            for (size_t j = 0; j < data.d; j++) {
                abs_norm[j] = std::abs(old_residual[j]) / old_norm;
            }
            old_t0 = rabitq_multibit::compute_optimal_scaling_factor(
                    abs_norm.data(), data.d, nb_bits);
        }
        const float tpred = old_t0 / old_norm * new_norm;
        for (size_t j = 0; j < data.d; j++) {
            abs_norm[j] = std::abs(new_residual[j]) / new_norm;
        }
        const float traw = rabitq_multibit::compute_optimal_scaling_factor(
                abs_norm.data(), data.d, nb_bits);
        make_tpred_full_code(new_residual.data(), data.d, nb_bits, tpred, pred_code);
        make_tpred_full_code(new_residual.data(), data.d, nb_bits, traw, raw_code);
        size_t h = 0;
        for (size_t j = 0; j < data.d; j++) {
            if (pred_code[j] != raw_code[j]) {
                h++;
            }
        }
        const double jp = old_state_objective(pred_code, new_residual.data(), data.d, nb_bits);
        const double jr = old_state_objective(raw_code, new_residual.data(), data.d, nb_bits);
        const double initial_gap = jr - jp;
        double best_local_objective = jp;
        best_local_code = pred_code;
        std::vector<float> query_residual(data.d);
        double true_distance = 0.0;
        double query_norm2 = 0.0;
        if (queries != nullptr && n_queries > 0) {
            const float* query = queries + (seen % n_queries) * data.d;
            for (size_t j = 0; j < data.d; j++) {
                query_residual[j] = query[j] - new_c[j];
                const double delta = query_residual[j] - new_residual[j];
                true_distance += delta * delta;
                query_norm2 += query_residual[j] * query_residual[j];
            }
        }
        auto record_distance_are = [&](const std::vector<int>& code, TDigestStats& stat) {
            if (!(true_distance > 1e-12)) {
                return;
            }
            const double center = ((1 << nb_bits) - 1) * 0.5;
            double code_dot_r = 0.0;
            double code_dot_q = 0.0;
            for (size_t j = 0; j < data.d; j++) {
                const double value = code[j] - center;
                code_dot_r += value * new_residual[j];
                code_dot_q += value * query_residual[j];
            }
            if (std::abs(code_dot_r) <= 1e-12) {
                return;
            }
            const double estimated_ip = new_norm2 * code_dot_q / code_dot_r;
            const double estimated_distance = query_norm2 + new_norm2 - 2.0 * estimated_ip;
            stat.add(std::abs(estimated_distance - true_distance) / true_distance);
        };
        auto record_local_gap = [&](size_t ki) {
            const double gap = std::max(0.0, jr - best_local_objective);
            local_range_gap[ki].add(gap);
            if (jr > 1e-12) {
                local_range_relative_gap[ki].add(gap / jr);
            }
            record_distance_are(best_local_code, local_range_distance_are[ki]);
        };
        auto evaluate_local_scale = [&](int step, float direction) {
            const float scale =
                    1.0f + direction * local_range_step_size * step;
            make_tpred_full_code(
                    new_residual.data(),
                    data.d,
                    nb_bits,
                    tpred * scale,
                    cand_code);
            const double value = old_state_objective(
                    cand_code, new_residual.data(), data.d, nb_bits);
            if (value > best_local_objective) {
                best_local_objective = value;
                best_local_code = cand_code;
            }
            return value;
        };
        record_local_gap(0);
        double left_best = -std::numeric_limits<double>::infinity();
        double right_best = -std::numeric_limits<double>::infinity();
        size_t next_range_index = 1;
        for (int step = 1; step <= 2; step++) {
            left_best =
                    std::max(left_best, evaluate_local_scale(step, -1.0f));
            right_best =
                    std::max(right_best, evaluate_local_scale(step, 1.0f));
            record_local_gap(next_range_index++);
        }
        const float local_direction = right_best > left_best ? 1.0f : -1.0f;
        for (int step = 3; step <= local_range_steps.back(); step++) {
            evaluate_local_scale(step, local_direction);
            if (next_range_index < local_range_steps.size() &&
                local_range_steps[next_range_index] == step) {
                record_local_gap(next_range_index++);
            }
        }
        record_distance_are(raw_code, optimal_distance_are);
        const double signed_log = std::log(static_cast<double>(tpred) / traw);
        signed_log_ratio.add(signed_log);
        pred_left_of_opt.add(tpred < traw ? 1.0 : 0.0);
        pred_hit_1e3.add(initial_gap <= 1e-3 ? 1.0 : 0.0);
        pred_hit_1e4.add(initial_gap <= 1e-4 ? 1.0 : 0.0);
        auto add_adaptive = [&](double min_gain, TDigestStats& gap, TDigestStats& steps,
                                TDigestStats& evals, TDigestStats& hit3,
                                TDigestStats& hit4, TDigestStats& limit) {
            AdaptiveTResult adaptive = adaptive_t_search(
                    new_residual.data(), data.d, nb_bits, tpred,
                    std::exp(1.0f / 32.0f), 32, 4, min_gain, cand_code);
            const double adaptive_gap = jr - adaptive.objective;
            gap.add(adaptive_gap);
            steps.add(adaptive.steps);
            evals.add(adaptive.evaluations);
            hit3.add(adaptive_gap <= 1e-3 ? 1.0 : 0.0);
            hit4.add(adaptive_gap <= 1e-4 ? 1.0 : 0.0);
            limit.add(adaptive.hit_limit ? 1.0 : 0.0);
        };
        add_adaptive(1e-5, adapt1_gap, adapt1_steps, adapt1_evals, adapt1_hit_1e3, adapt1_hit_1e4, adapt1_limit);
        add_adaptive(3e-5, adapt3_gap, adapt3_steps, adapt3_evals, adapt3_hit_1e3, adapt3_hit_1e4, adapt3_limit);
        add_adaptive(1e-4, adapt4_gap, adapt4_steps, adapt4_evals, adapt4_hit_1e3, adapt4_hit_1e4, adapt4_limit);
        auto add_candidate_stats = [&](
                float tcand,
                TDigestStats& gap_stat,
                TDigestStats& hop_stat,
                TDigestStats& abslog_stat) {
            if (!(tcand > 0.0f) || !(traw > 0.0f)) {
                return;
            }
            make_tpred_full_code(new_residual.data(), data.d, nb_bits, tcand, cand_code);
            size_t ch = 0;
            for (size_t j = 0; j < data.d; j++) {
                if (cand_code[j] != raw_code[j]) {
                    ch++;
                }
            }
            const double jc = old_state_objective(cand_code, new_residual.data(), data.d, nb_bits);
            gap_stat.add(jr - jc);
            hop_stat.add(static_cast<double>(ch));
            abslog_stat.add(std::abs(std::log(static_cast<double>(tcand) / traw)));
        };
        float const_median_t = 0.0f;
        if (data.d == 96) {
            if (nb_bits == 2) {
                const_median_t = 12.09640837f;
            } else if (nb_bits == 4) {
                const_median_t = 35.64824677f;
            } else if (nb_bits == 8) {
                const_median_t = 386.3062744f;
            }
        }
        add_candidate_stats(old_t0, old_t_gap, old_t_hop, old_t_abslog);
        if (const_median_t > 0.0f) {
            add_candidate_stats(const_median_t, const_median_gap, const_median_hop, const_median_abslog);
            add_candidate_stats(const_median_t * 0.5f, const_half_gap, const_half_hop, const_half_abslog);
            add_candidate_stats(const_median_t * 2.0f, const_double_gap, const_double_hop, const_double_abslog);
        }
        hop.add(static_cast<double>(h));
        frac_hop.add(static_cast<double>(h) / static_cast<double>(data.d));
        if (tpred > 0.0f && traw > 0.0f) {
            abs_log_ratio.add(std::abs(std::log(static_cast<double>(tpred) / traw)));
        }
        pred_j.add(jp);
        raw_j.add(jr);
        if (std::abs(jr) > 1e-12) {
            j_ratio.add(jp / jr);
        }
        cos_gap.add(jr - jp);
        shift_ratio.add(std::sqrt(shift_norm2) / static_cast<double>(old_norm));
        seen++;
    }

    std::ofstream out(path);
    FAISS_THROW_IF_NOT_MSG(out, "failed to open RaBitQ t diagnostic output");
    std::ostringstream os;
    os << "{\n"
       << "  \"nbits\": " << static_cast<int>(nb_bits) << ",\n"
       << "  \"sampled_vectors\": " << seen << ",\n"
       << "  \"dimension\": " << data.d << ",\n"
       << "  \"stats\": {\n";
    append_stat_json(os, "hop_count", hop); os << ",\n";
    append_stat_json(os, "hop_fraction", frac_hop); os << ",\n";
    append_stat_json(os, "abs_log_tpred_over_traw", abs_log_ratio); os << ",\n";
    append_stat_json(os, "objective_pred", pred_j); os << ",\n";
    append_stat_json(os, "objective_raw", raw_j); os << ",\n";
    append_stat_json(os, "objective_ratio_pred_over_raw", j_ratio); os << ",\n";
    append_stat_json(os, "objective_gap_raw_minus_pred", cos_gap); os << ",\n";
    append_stat_json(os, "objective_gap_raw_minus_old_t_no_norm_ratio", old_t_gap); os << ",\n";
    append_stat_json(os, "hop_count_old_t_no_norm_ratio", old_t_hop); os << ",\n";
    append_stat_json(os, "abs_log_old_t_over_traw", old_t_abslog); os << ",\n";
    append_stat_json(os, "objective_gap_raw_minus_const_median_048", const_median_gap); os << ",\n";
    append_stat_json(os, "hop_count_const_median_048", const_median_hop); os << ",\n";
    append_stat_json(os, "abs_log_const_median_over_traw", const_median_abslog); os << ",\n";
    append_stat_json(os, "objective_gap_raw_minus_const_half_median_048", const_half_gap); os << ",\n";
    append_stat_json(os, "hop_count_const_half_median_048", const_half_hop); os << ",\n";
    append_stat_json(os, "objective_gap_raw_minus_const_double_median_048", const_double_gap); os << ",\n";
    append_stat_json(os, "hop_count_const_double_median_048", const_double_hop); os << ",\n";
    append_stat_json(os, "signed_log_tpred_over_topt", signed_log_ratio); os << ",\n";
    append_stat_json(os, "tpred_left_of_topt_fraction", pred_left_of_opt); os << ",\n";
    append_stat_json(os, "tpred_hit_1e-3", pred_hit_1e3); os << ",\n";
    append_stat_json(os, "tpred_hit_1e-4", pred_hit_1e4); os << ",\n";
    append_stat_json(os, "adaptive_1e-5_gap", adapt1_gap); os << ",\n";
    append_stat_json(os, "adaptive_1e-5_steps", adapt1_steps); os << ",\n";
    append_stat_json(os, "adaptive_1e-5_evaluations", adapt1_evals); os << ",\n";
    append_stat_json(os, "adaptive_1e-5_hit_1e-3", adapt1_hit_1e3); os << ",\n";
    append_stat_json(os, "adaptive_1e-5_hit_1e-4", adapt1_hit_1e4); os << ",\n";
    append_stat_json(os, "adaptive_1e-5_hit_limit", adapt1_limit); os << ",\n";
    append_stat_json(os, "adaptive_3e-5_gap", adapt3_gap); os << ",\n";
    append_stat_json(os, "adaptive_3e-5_steps", adapt3_steps); os << ",\n";
    append_stat_json(os, "adaptive_3e-5_evaluations", adapt3_evals); os << ",\n";
    append_stat_json(os, "adaptive_3e-5_hit_1e-3", adapt3_hit_1e3); os << ",\n";
    append_stat_json(os, "adaptive_3e-5_hit_1e-4", adapt3_hit_1e4); os << ",\n";
    append_stat_json(os, "adaptive_3e-5_hit_limit", adapt3_limit); os << ",\n";
    append_stat_json(os, "adaptive_1e-4_gap", adapt4_gap); os << ",\n";
    append_stat_json(os, "adaptive_1e-4_steps", adapt4_steps); os << ",\n";
    append_stat_json(os, "adaptive_1e-4_evaluations", adapt4_evals); os << ",\n";
    append_stat_json(os, "adaptive_1e-4_hit_1e-3", adapt4_hit_1e3); os << ",\n";
    append_stat_json(os, "adaptive_1e-4_hit_1e-4", adapt4_hit_1e4); os << ",\n";
    append_stat_json(os, "adaptive_1e-4_hit_limit", adapt4_limit); os << ",\n";
    append_stat_json(os, "centroid_shift_over_old_residual_norm", shift_ratio); os << "\n";
    os << "  },\n"
       << "  \"local_t_range\": {\n"
       << "    \"step_size\": " << local_range_step_size << ",\n"
       << "    \"definition\": \"J(t_opt) - max J(t) among directional local-t candidates\",\n"
       << "    \"direction_rule\": \"probe both sides through step 2, then continue in the better direction\",\n"
       << "    \"objective_regret_by_max_step\": {\n";
    for (size_t ki = 0; ki < local_range_steps.size(); ki++) {
        const std::string name = std::to_string(local_range_steps[ki]);
        append_stat_json(os, name.c_str(), local_range_gap[ki]);
        os << (ki + 1 == local_range_steps.size() ? "\n" : ",\n");
    }
    os << "    },\n"
       << "    \"relative_definition\": \"(J(t_opt) - J(t_K)) / J(t_opt), computed per vector\",\n"
       << "    \"relative_objective_regret_by_max_step\": {\n";
    for (size_t ki = 0; ki < local_range_steps.size(); ki++) {
        const std::string name = std::to_string(local_range_steps[ki]);
        append_stat_json(os, name.c_str(), local_range_relative_gap[ki]);
        os << (ki + 1 == local_range_steps.size() ? "\n" : ",\n");
    }
    os << "    }\n"
       << "  },\n"
       << "  \"distance_estimation\": {\n"
       << "    \"definition\": \"mean abs(estimated squared L2 - true squared L2) / true squared L2 over paired queries\",\n"
       << "    \"query_pairing\": \"sampled database vector i paired with query i modulo query count, using the database vector merged centroid\",\n"
       << "    \"are_by_max_step\": {\n";
    for (size_t ki = 0; ki < local_range_steps.size(); ki++) {
        const std::string name = std::to_string(local_range_steps[ki]);
        append_stat_json(os, name.c_str(), local_range_distance_are[ki]);
        os << (ki + 1 == local_range_steps.size() ? "\n" : ",\n");
    }
    os << "    },\n";
    append_stat_json(os, "t_opt", optimal_distance_are);
    os << "\n  }\n}\n";
    out << os.str();
}

static std::unique_ptr<IndexIVFRaBitQ> build_rabitq_from_old_state_k0(
        const IVFDataForMerge& data,
        uint8_t nb_bits,
        uint8_t qb,
        MergeRunStats* stats,
        bool mirror_only,
        const std::string& t_init_mode,
        int local_t_steps,
        float local_t_step,
        bool adaptive_t,
        float adaptive_t_factor,
        int adaptive_t_max_steps,
        int adaptive_t_patience,
        float adaptive_t_min_gain) {
    auto t0 = std::chrono::steady_clock::now();
    FAISS_THROW_IF_NOT(data.metric == METRIC_L2);
    const float* vectors = data.vectors_view ? data.vectors_view : data.vectors.data();
    FAISS_THROW_IF_NOT(vectors != nullptr);
    FAISS_THROW_IF_NOT(data.vectors_view || data.vectors.size() == data.ntotal * data.d);
    FAISS_THROW_IF_NOT(data.old_code_size > 0);
    FAISS_THROW_IF_NOT(data.old_codes.size() == data.ntotal * data.old_code_size);
    FAISS_THROW_IF_NOT(data.old_centroids.size() % data.d == 0);
    FAISS_THROW_IF_NOT(data.old_list_for_id.size() == data.ntotal);
    FAISS_THROW_IF_NOT(data.old_t0_by_id.empty() || data.old_t0_by_id.size() == data.ntotal);
    FAISS_THROW_IF_NOT(data.centroids.size() == data.nlist * data.d);
    FAISS_THROW_IF_NOT(data.lists.size() == data.nlist);

    auto quantizer = std::make_unique<IndexFlatL2>(static_cast<int>(data.d));
    quantizer->add(static_cast<idx_t>(data.nlist), data.centroids.data());

    auto out = std::make_unique<IndexIVFRaBitQ>(
            quantizer.get(), data.d, data.nlist, data.metric, true, nb_bits);
    out->own_fields = true;
    out->quantizer = quantizer.release();
    out->is_trained = true;
    out->qb = qb;

    for (size_t list_no = 0; list_no < data.nlist; list_no++) {
        const auto& ids_in = data.lists[list_no];
        const size_t n = ids_in.size();
        if (n == 0) {
            continue;
        }
        std::vector<idx_t> ids(n);
        std::vector<uint8_t> codes(n * out->code_size);
        std::vector<int> old_full_code;
        std::vector<int> mirror_code;
        std::vector<int> tpred_code;
        std::vector<int> ex_tmp;
        std::vector<float> old_residual;
        std::vector<float> new_residual;
        const float* centroid = data.centroids.data() + list_no * data.d;
        auto t_encode0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < n; i++) {
            const idx_t id = ids_in[i];
            FAISS_THROW_IF_NOT(id >= 0);
            const size_t id_sz = static_cast<size_t>(id);
            FAISS_THROW_IF_NOT(id_sz < data.ntotal);
            ids[i] = id;
            uint8_t* code = codes.data() + i * out->code_size;
            std::memset(code, 0, out->code_size);
            const idx_t old_list = data.old_list_for_id[id_sz];
            FAISS_THROW_IF_NOT(old_list >= 0);
            const float* old_centroid = data.old_centroids.data() + static_cast<size_t>(old_list) * data.d;
            encode_old_state_k0_code(
                    vectors + id_sz * data.d,
                    old_centroid,
                    centroid,
                    data.old_codes.data() + id_sz * data.old_code_size,
                    data.old_t0_by_id.empty() ? 0.0f : data.old_t0_by_id[id_sz],
                    data.d,
                    nb_bits,
                    data.metric,
                    mirror_only,
                    t_init_mode,
                    local_t_steps,
                    local_t_step,
                    adaptive_t,
                    adaptive_t_factor,
                    adaptive_t_max_steps,
                    adaptive_t_patience,
                    adaptive_t_min_gain,
                    code,
                    old_full_code,
                    mirror_code,
                    tpred_code,
                    ex_tmp,
                    old_residual,
                    new_residual);
        }
        auto t_encode1 = std::chrono::steady_clock::now();
        auto t_add0 = std::chrono::steady_clock::now();
        out->invlists->add_entries(list_no, n, ids.data(), codes.data());
        auto t_add1 = std::chrono::steady_clock::now();
        if (stats) {
            stats->ivfpq_reencode_encode_codes_s +=
                    std::chrono::duration<double>(t_encode1 - t_encode0).count();
            stats->ivfpq_reencode_add_entries_s +=
                    std::chrono::duration<double>(t_add1 - t_add0).count();
        }
    }
    out->ntotal = static_cast<idx_t>(data.ntotal);

    if (stats) {
        auto t1 = std::chrono::steady_clock::now();
        stats->ivfpq_reencode_s = std::chrono::duration<double>(t1 - t0).count();
        stats->build_index_s = stats->ivfpq_reencode_s;
    }
    return out;
}


static std::unique_ptr<IndexIVFRaBitQ> build_rabitq_from_final_assign_direct_1bit(
        const IVFDataForMerge& data,
        uint8_t qb,
        MergeRunStats* stats) {
    auto t0 = std::chrono::steady_clock::now();
    FAISS_THROW_IF_NOT(data.metric == METRIC_L2);
    const float* vectors = data.vectors_view ? data.vectors_view : data.vectors.data();
    FAISS_THROW_IF_NOT(vectors != nullptr);
    FAISS_THROW_IF_NOT(data.vectors_view || data.vectors.size() == data.ntotal * data.d);
    FAISS_THROW_IF_NOT(data.centroids.size() == data.nlist * data.d);
    FAISS_THROW_IF_NOT(data.final_assign.size() == data.ntotal);

    auto quantizer = std::make_unique<IndexFlatL2>(static_cast<int>(data.d));
    quantizer->add(static_cast<idx_t>(data.nlist), data.centroids.data());

    auto out = std::make_unique<IndexIVFRaBitQ>(
            quantizer.get(), data.d, data.nlist, data.metric, true, 1);
    out->own_fields = true;
    out->quantizer = quantizer.release();
    out->is_trained = true;
    out->qb = qb;

    const float inv_d_sqrt = data.d == 0
            ? 1.0f
            : 1.0f / std::sqrt(static_cast<float>(data.d));
    constexpr float epsilon = std::numeric_limits<float>::epsilon();

    std::vector<size_t> counts(data.nlist, 0);
    for (idx_t list_no : data.final_assign) {
        FAISS_THROW_IF_NOT(list_no >= 0);
        FAISS_THROW_IF_NOT(static_cast<size_t>(list_no) < data.nlist);
        counts[static_cast<size_t>(list_no)]++;
    }

    std::vector<std::vector<idx_t>> ids_by_list(data.nlist);
    std::vector<std::vector<uint8_t>> codes_by_list(data.nlist);
    for (size_t list_no = 0; list_no < data.nlist; list_no++) {
        ids_by_list[list_no].resize(counts[list_no]);
        codes_by_list[list_no].resize(counts[list_no] * out->code_size);
    }

    std::vector<size_t> offsets(data.nlist, 0);
    auto t_encode0 = std::chrono::steady_clock::now();
    for (size_t id_sz = 0; id_sz < data.ntotal; id_sz++) {
        const idx_t id = static_cast<idx_t>(id_sz);
        const size_t list_no = static_cast<size_t>(data.final_assign[id_sz]);
        const size_t pos = offsets[list_no]++;
        ids_by_list[list_no][pos] = id;

        const float* centroid = data.centroids.data() + list_no * data.d;
        const float* x = vectors + id_sz * data.d;
        uint8_t* code = codes_by_list[list_no].data() + pos * out->code_size;
        std::memset(code, 0, out->code_size);

        float norm_L2sqr = 0.0f;
        float dp_oO = 0.0f;
        for (size_t j = 0; j < data.d; j++) {
            const float or_minus_c = x[j] - centroid[j];
            norm_L2sqr += or_minus_c * or_minus_c;
            if (or_minus_c > 0.0f) {
                dp_oO += or_minus_c;
                code[j >> 3] |= static_cast<uint8_t>(1u << (j & 7));
            } else {
                dp_oO -= or_minus_c;
            }
        }

        const float sqrt_norm_L2 = std::sqrt(norm_L2sqr);
        const float inv_norm_L2 =
                (norm_L2sqr < epsilon) ? 1.0f : (1.0f / sqrt_norm_L2);
        const float normalized_dp = dp_oO * inv_norm_L2 * inv_d_sqrt;
        const float inv_dp_oO =
                (std::abs(normalized_dp) < epsilon) ? 1.0f : (1.0f / normalized_dp);

        auto* factors = reinterpret_cast<rabitq_utils::SignBitFactors*>(
                code + (data.d + 7) / 8);
        factors->or_minus_c_l2sqr = norm_L2sqr;
        factors->dp_multiplier = inv_dp_oO * sqrt_norm_L2;
    }
    auto t_encode1 = std::chrono::steady_clock::now();

    auto t_add0 = std::chrono::steady_clock::now();
    for (size_t list_no = 0; list_no < data.nlist; list_no++) {
        const size_t n = counts[list_no];
        if (n == 0) {
            continue;
        }
        out->invlists->add_entries(
                list_no,
                n,
                ids_by_list[list_no].data(),
                codes_by_list[list_no].data());
    }
    auto t_add1 = std::chrono::steady_clock::now();
    out->ntotal = static_cast<idx_t>(data.ntotal);

    if (stats) {
        auto t1 = std::chrono::steady_clock::now();
        stats->ivfpq_reencode_encode_codes_s +=
                std::chrono::duration<double>(t_encode1 - t_encode0).count();
        stats->ivfpq_reencode_add_entries_s +=
                std::chrono::duration<double>(t_add1 - t_add0).count();
        stats->ivfpq_reencode_s = std::chrono::duration<double>(t1 - t0).count();
        stats->build_index_s = stats->ivfpq_reencode_s;
    }
    return out;
}


} // namespace

std::unique_ptr<Index> merge_ivfrabitq(
        const std::vector<IndexIVFRaBitQ*>& indices,
        const MergeArguments& options) {
    FAISS_THROW_IF_NOT(!indices.empty());
    for (const IndexIVFRaBitQ* idx : indices) {
        ensure_ivfrabitq_compatible(*idx);
    }

    if (options.method == MergeMethod::Concat) {
        auto shards = std::make_unique<IndexShards>(indices[0]->d, false, true);
        for (const IndexIVFRaBitQ* idx : indices) {
            shards->add_shard(clone_index(idx));
        }
        return shards;
    }

    auto wall0 = std::chrono::steady_clock::now();
    MergeRunStats local_stats;
    MergeRunStats* stats = options.run_stats ? options.run_stats : &local_stats;
    MergeOptions merge_options = options.merge;
    InRemapRaBitQEncodeContext in_remap_ctx;
    size_t total_ntotal = 0;
    for (const IndexIVFRaBitQ* idx : indices) {
        total_ntotal += static_cast<size_t>(idx->ntotal);
    }
    const bool have_raw_vectors = merge_options.ivf_merge_use_raw &&
            merge_options.training_vectors != nullptr &&
            merge_options.n_training_vectors == total_ntotal;

    // A 1-bit RaBitQ code has no scaling t or extra-bit state to reuse.
    // Avoid the multi-bit old-state adaptation path and its needless decoding.
    const bool keep_old_codes = merge_options.use_old_state_rabitq_reencode &&
            merge_options.target_nbits > 1;
    IVFDataForMerge data = decode_concat_to_ivf_data(indices, !have_raw_vectors, keep_old_codes, stats);
    if (keep_old_codes && merge_options.old_state_t0_by_id != nullptr) {
        FAISS_THROW_IF_NOT(merge_options.n_old_state_t0 == data.ntotal);
        data.old_t0_by_id.assign(
                merge_options.old_state_t0_by_id,
                merge_options.old_state_t0_by_id + data.ntotal);
    }
    if (have_raw_vectors) {
        if (merge_options.use_ivf_merge_lists_direct_reencode) {
            data.vectors_view = merge_options.training_vectors;
        } else {
            data.vectors.assign(
                    merge_options.training_vectors,
                    merge_options.training_vectors + data.ntotal * data.d);
            data.vectors_view = data.vectors.data();
        }
    }

    std::vector<std::vector<idx_t>> source_lists_for_reencode;
    if (merge_options.use_source_list_order_rabitq_reencode) {
        source_lists_for_reencode = data.lists;
    }

    const bool use_in_remap_for_this_nbits =
            merge_options.use_in_remap_rabitq_encode &&
            merge_options.target_nbits == 1;
    if (merge_options.target_nbits != 1) {
        merge_options.return_final_assign_without_lists = false;
        merge_options.use_direct_1bit_rabitq_reencode = false;
    }

    if (use_in_remap_for_this_nbits) {
        FAISS_THROW_IF_NOT_MSG(
                data.vectors_view != nullptr || !data.vectors.empty(),
                "use_in_remap_rabitq_encode requires a vector source");
        in_remap_ctx.d = data.d;
        in_remap_ctx.nlist = merge_options.target_nlist;
        in_remap_ctx.ntotal = data.ntotal;
        in_remap_ctx.code_size = (data.d + 7) / 8 + sizeof(rabitq_utils::SignBitFactors);
        in_remap_ctx.inv_d_sqrt = data.d == 0
                ? 1.0f
                : 1.0f / std::sqrt(static_cast<float>(data.d));
        in_remap_ctx.ids_by_list.assign(in_remap_ctx.nlist, {});
        in_remap_ctx.codes_by_list.assign(in_remap_ctx.nlist, {});
        if (merge_options.reserve_in_remap_rabitq_buffers && in_remap_ctx.nlist > 0) {
            const size_t reserve_n = std::max<size_t>(1, data.ntotal / in_remap_ctx.nlist + 8);
            for (size_t list_no = 0; list_no < in_remap_ctx.nlist; list_no++) {
                in_remap_ctx.ids_by_list[list_no].reserve(reserve_n);
                in_remap_ctx.codes_by_list[list_no].reserve(reserve_n * in_remap_ctx.code_size);
            }
        }
        merge_options.return_final_assign_without_lists = true;
        merge_options.remap_batch_callback = encode_in_remap_rabitq_1bit_callback;
        merge_options.remap_batch_callback_user_data = &in_remap_ctx;
    }

    auto ivf0 = std::chrono::steady_clock::now();
    if (merge_options.skip_ivf_merge_use_reference_centroids) {
        FAISS_THROW_IF_NOT_MSG(
                have_raw_vectors,
                "skip_ivf_merge_use_reference_centroids requires full raw vectors");
        FAISS_THROW_IF_NOT_MSG(
                merge_options.reference_centroids != nullptr &&
                        merge_options.n_reference_centroids > 0,
                "skip_ivf_merge_use_reference_centroids requires reference centroids");
        data.nlist = merge_options.n_reference_centroids;
        data.centroids.assign(
                merge_options.reference_centroids,
                merge_options.reference_centroids + data.nlist * data.d);
        data.lists.assign(data.nlist, {});
    } else {
        merge_ivf_data(data, merge_options, stats);
        if (merge_options.reference_centroids &&
            merge_options.n_reference_centroids == data.nlist) {
            data.centroids.assign(
                    merge_options.reference_centroids,
                    merge_options.reference_centroids + data.nlist * data.d);
        }
    }
    auto ivf1 = std::chrono::steady_clock::now();
    stats->ivf_merge_s = std::chrono::duration<double>(ivf1 - ivf0).count();

    if (merge_options.use_ivf_merge_lists_direct_reencode) {
        FAISS_THROW_IF_NOT_MSG(
                data.vectors_view != nullptr || !data.vectors.empty(),
                "use_ivf_merge_lists_direct_reencode requires a vector source");
        if (have_raw_vectors) {
            data.vectors_view = merge_options.training_vectors;
        }
    } else {
        exact_assign_to_final_centroids(data, merge_options.batch_size, stats);
    }

    const uint8_t nb_bits = merge_options.target_nbits > 0
            ? static_cast<uint8_t>(merge_options.target_nbits)
            : indices[0]->rabitq.nb_bits;
    std::unique_ptr<IndexIVFRaBitQ> out;
    if (!merge_options.rabitq_t_diagnostic_path.empty()) {
        write_rabitq_t_diagnostic(
                data,
                nb_bits,
                merge_options.rabitq_t_diagnostic_path,
                merge_options.rabitq_t_diagnostic_max_vectors,
                merge_options.rabitq_distance_queries,
                merge_options.n_rabitq_distance_queries);
        if (merge_options.rabitq_t_diagnostic_only) {
            auto quantizer = std::make_unique<IndexFlatL2>(static_cast<int>(data.d));
            quantizer->add(static_cast<idx_t>(data.nlist), data.centroids.data());
            out = std::make_unique<IndexIVFRaBitQ>(
                    quantizer.get(), data.d, data.nlist, data.metric, true, nb_bits);
            out->own_fields = true;
            out->quantizer = quantizer.release();
            out->is_trained = true;
            out->qb = indices[0]->qb;
            auto wall1 = std::chrono::steady_clock::now();
            stats->total_s = std::chrono::duration<double>(wall1 - wall0).count();
            finalize_merge_run_stats(stats);
            return out;
        }
    }
    if (use_in_remap_for_this_nbits) {
        FAISS_THROW_IF_NOT_MSG(nb_bits == 1, "in-remap RaBitQ encode requires nb_bits=1");
        out = build_rabitq_from_in_remap_buffers_direct_1bit(
                data, in_remap_ctx, indices[0]->qb, stats);
    } else if (merge_options.return_final_assign_without_lists) {
        FAISS_THROW_IF_NOT_MSG(nb_bits == 1, "fused assign RaBitQ reencode requires nb_bits=1");
        if (merge_options.use_source_list_order_rabitq_reencode) {
            out = build_rabitq_from_source_lists_final_assign_direct_1bit(
                    data, source_lists_for_reencode, indices[0]->qb, stats);
        } else {
            out = build_rabitq_from_final_assign_direct_1bit(data, indices[0]->qb, stats);
        }
    } else if (merge_options.use_old_state_rabitq_reencode && nb_bits > 1) {
        out = build_rabitq_from_old_state_k0(
                data, nb_bits, indices[0]->qb, stats,
                merge_options.old_state_mirror_only,
                merge_options.old_state_t_init_mode,
                merge_options.old_state_local_t_steps,
                merge_options.old_state_local_t_step,
                merge_options.old_state_adaptive_t,
                merge_options.old_state_adaptive_t_factor,
                merge_options.old_state_adaptive_t_max_steps,
                merge_options.old_state_adaptive_t_patience,
                merge_options.old_state_adaptive_t_min_gain);
    } else if (merge_options.use_direct_1bit_rabitq_reencode) {
        FAISS_THROW_IF_NOT_MSG(nb_bits == 1, "direct 1-bit RaBitQ reencode requires nb_bits=1");
        out = build_rabitq_from_merged_data_direct_1bit(data, indices[0]->qb, stats);
    } else if (merge_options.use_listwise_rabitq_reencode) {
        out = build_rabitq_from_merged_data_listwise(data, nb_bits, indices[0]->qb, stats);
    } else {
        out = build_rabitq_from_merged_data(data, nb_bits, indices[0]->qb, stats);
    }

    auto wall1 = std::chrono::steady_clock::now();
    stats->total_s = std::chrono::duration<double>(wall1 - wall0).count();
    finalize_merge_run_stats(stats);
    return out;
}

} // namespace faiss
