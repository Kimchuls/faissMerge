#include <faiss/IndexIVFFlatMerge.h>
#include <faiss/IndexIVFFlatMergeMT.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexShards.h>
#include <faiss/clone_index.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/utils/distances.h>
#include <cstdlib>

namespace {

using faiss::idx_t;

struct VectorLoc {
    uint32_t list_no = 0;
    uint32_t off = 0;
};

struct IVFData {
    size_t d = 0;
    size_t nlist = 0;
    size_t ntotal = 0;
    faiss::MetricType metric = faiss::METRIC_L2;
    std::vector<float> centroids;
    std::vector<std::vector<idx_t>> lists;
    /// Dense storage used by merge_ivf_data (IVFPQ path). Empty when using index view.
    std::vector<float> vectors;
    const float* vectors_view = nullptr;
    std::unique_ptr<faiss::IndexIVFFlat> index_owned;
    std::vector<idx_t> final_assign;
    faiss::IndexIVFFlat* index_view = nullptr;
    std::vector<VectorLoc> id_loc;
};

static const float* ivfdata_vector_at(const IVFData& data, idx_t id) {
    if (data.index_view != nullptr) {
        FAISS_THROW_IF_NOT(data.index_view->invlists != nullptr);
        FAISS_THROW_IF_NOT(static_cast<size_t>(id) < data.id_loc.size());
        const VectorLoc& loc = data.id_loc[static_cast<size_t>(id)];
        const uint8_t* codes = data.index_view->invlists->get_codes(loc.list_no);
        FAISS_THROW_IF_NOT(codes != nullptr);
        return reinterpret_cast<const float*>(
                codes + static_cast<size_t>(loc.off) * data.index_view->code_size);
    }
    if (!data.vectors.empty()) {
        FAISS_THROW_IF_NOT(static_cast<size_t>(id) < data.ntotal);
        return data.vectors.data() + static_cast<size_t>(id) * data.d;
    }
    if (data.vectors_view != nullptr) {
        FAISS_THROW_IF_NOT(static_cast<size_t>(id) < data.ntotal);
        return data.vectors_view + static_cast<size_t>(id) * data.d;
    }
    FAISS_THROW_MSG("ivfdata_vector_at: no vector source");
}

static void ivfdata_copy_vector(const IVFData& data, idx_t id, float* dst) {
    const float* src = ivfdata_vector_at(data, id);
    std::memcpy(dst, src, data.d * sizeof(float));
}

static const float* ivfdata_contiguous_vector_slice(
        const IVFData& data,
        const std::vector<idx_t>& ids,
        size_t begin,
        size_t count,
        const uint8_t** codes_out,
        size_t* list_no_out) {
    *codes_out = nullptr;
    *list_no_out = 0;
    if (data.index_view == nullptr || count == 0) {
        return nullptr;
    }
    FAISS_THROW_IF_NOT(data.index_view->invlists != nullptr);
    FAISS_THROW_IF_NOT(begin + count <= ids.size());
    const idx_t first_id = ids[begin];
    if (first_id < 0 || static_cast<size_t>(first_id) >= data.id_loc.size()) {
        return nullptr;
    }
    const VectorLoc first = data.id_loc[static_cast<size_t>(first_id)];
    for (size_t i = 0; i < count; i++) {
        const idx_t id = ids[begin + i];
        if (id < 0 || static_cast<size_t>(id) >= data.id_loc.size()) {
            return nullptr;
        }
        const VectorLoc loc = data.id_loc[static_cast<size_t>(id)];
        if (loc.list_no != first.list_no ||
            loc.off != first.off + static_cast<uint32_t>(i)) {
            return nullptr;
        }
    }
    const uint8_t* codes = data.index_view->invlists->get_codes(first.list_no);
    FAISS_THROW_IF_NOT(codes != nullptr);
    *codes_out = codes;
    *list_no_out = first.list_no;
    return reinterpret_cast<const float*>(
            codes + static_cast<size_t>(first.off) * data.index_view->code_size);
}

static void ensure_ivfflat_compatible(const faiss::IndexIVFFlat& idx) {
    FAISS_THROW_IF_NOT(idx.is_trained);
    FAISS_THROW_IF_NOT(idx.quantizer);
    FAISS_THROW_IF_NOT(idx.quantizer->ntotal == idx.nlist);
    FAISS_THROW_IF_NOT(idx.code_size == idx.d * sizeof(float));
    FAISS_THROW_IF_NOT_MSG(idx.by_residual == false, "merge optimizer requires by_residual == false");
    FAISS_THROW_IF_NOT_MSG(idx.direct_map.no(), "merge requires direct_map disabled");
    FAISS_THROW_IF_NOT_MSG(idx.metric_type == faiss::METRIC_L2, "merge optimizer currently supports METRIC_L2 only");
}

static void concat_indices_into_ivfdata(
        const std::vector<faiss::IndexIVFFlat*>& indices,
        IVFData& out) {
    FAISS_THROW_IF_NOT(!indices.empty());
    for (const auto* idx : indices) {
        ensure_ivfflat_compatible(*idx);
    }

    const size_t d = static_cast<size_t>(indices[0]->d);
    size_t total_nlist = 0;
    size_t total_ntotal = 0;
    for (const auto* idx : indices) {
        total_nlist += static_cast<size_t>(idx->nlist);
        total_ntotal += static_cast<size_t>(idx->ntotal);
    }

    std::vector<const faiss::InvertedLists*> ils_ptrs;
    ils_ptrs.reserve(indices.size());
    for (const auto* idx : indices) {
        FAISS_THROW_IF_NOT(idx->invlists != nullptr);
        ils_ptrs.push_back(idx->invlists);
    }

    out.d = d;
    out.nlist = total_nlist;
    out.ntotal = total_ntotal;
    out.metric = indices[0]->metric_type;
    out.centroids.resize(total_nlist * d);
    out.lists.assign(total_nlist, {});
    out.id_loc.assign(total_ntotal, {});

    // Reconstruct each shard's centroids once, in a single batched call per
    // shard, straight into the final out.centroids buffer (IndexFlat's
    // reconstruct_n decodes a contiguous block in one shot, unlike looping
    // over reconstruct() one list at a time). The quantizer below is then
    // built from that same buffer instead of reconstructing everything a
    // second time.
    {
        size_t centroid_offset = 0;
        for (const auto* idx : indices) {
            idx->quantizer->reconstruct_n(
                    0,
                    static_cast<faiss::idx_t>(idx->nlist),
                    out.centroids.data() + centroid_offset * d);
            centroid_offset += static_cast<size_t>(idx->nlist);
        }
        FAISS_THROW_IF_NOT(centroid_offset == total_nlist);
    }

    auto quantizer = std::make_unique<faiss::IndexFlatL2>(static_cast<int>(d));
    quantizer->add(static_cast<faiss::idx_t>(total_nlist), out.centroids.data());
    FAISS_THROW_IF_NOT(static_cast<size_t>(quantizer->ntotal) == total_nlist);

    // VStack concatenates cluster lists across shards (total_nlist);
    // HStack would keep per-shard nlist and break replace_invlists / vector lookup.
    auto stacked_invlists = std::make_unique<faiss::VStackInvertedLists>(
            static_cast<int>(ils_ptrs.size()), ils_ptrs.data());

    auto index = std::make_unique<faiss::IndexIVFFlat>(
            quantizer.get(),
            static_cast<int>(d),
            static_cast<int>(total_nlist),
            indices[0]->metric_type,
            false);
    index->quantizer = quantizer.release();
    index->own_fields = true;
    index->replace_invlists(stacked_invlists.release(), true);
    index->ntotal = static_cast<faiss::idx_t>(total_ntotal);
    index->is_trained = true;

    out.index_owned = std::move(index);
    out.index_view = out.index_owned.get();
    out.vectors.clear();

    size_t global_list_no = 0;
    idx_t id_offset = 0;
    for (const auto* idx : indices) {
        for (size_t local_list = 0; local_list < static_cast<size_t>(idx->nlist);
             local_list++, global_list_no++) {
            const size_t list_size = idx->invlists->list_size(local_list);
            out.lists[global_list_no].reserve(list_size);
            if (list_size == 0) {
                continue;
            }
            faiss::InvertedLists::ScopedIds ids(idx->invlists, local_list);
            for (size_t i = 0; i < list_size; i++) {
                const idx_t global_id = ids[i] + id_offset;
                FAISS_THROW_IF_NOT(
                        global_id >= 0 &&
                        static_cast<size_t>(global_id) < total_ntotal);
                out.lists[global_list_no].push_back(global_id);
                out.id_loc[static_cast<size_t>(global_id)] = VectorLoc{
                        static_cast<uint32_t>(global_list_no),
                        static_cast<uint32_t>(i)};
            }
        }
        id_offset += idx->ntotal;
    }
}

static void ivfdata_from_dense_vectors(IVFData& out) {
    out.index_owned.reset();
    out.index_view = nullptr;
    out.id_loc.clear();
    FAISS_THROW_IF_NOT(!out.vectors.empty());
    FAISS_THROW_IF_NOT(out.vectors.size() == out.ntotal * out.d);
}

static std::unique_ptr<faiss::IndexIVFFlat> sync_ivfdata_to_index(IVFData& data);

static std::vector<idx_t> build_assign_from_lists(
        const std::vector<std::vector<idx_t>>& lists,
        size_t ntotal) {
    std::vector<idx_t> assign(ntotal, -1);
    for (size_t cid = 0; cid < lists.size(); cid++) {
        for (idx_t id : lists[cid]) {
            FAISS_THROW_IF_NOT(id >= 0 && static_cast<size_t>(id) < ntotal);
            assign[static_cast<size_t>(id)] = static_cast<idx_t>(cid);
        }
    }
    for (size_t i = 0; i < assign.size(); i++) {
        FAISS_THROW_IF_NOT_MSG(assign[i] >= 0, "unassigned vector id");
    }
    return assign;
}

static void rebuild_lists_from_assign(
        std::vector<std::vector<idx_t>>& lists,
        const std::vector<idx_t>& assign) {
    const size_t nlist = lists.size();
    std::vector<size_t> counts(nlist, 0);
    for (idx_t cid : assign) {
        FAISS_THROW_IF_NOT(cid >= 0 && static_cast<size_t>(cid) < nlist);
        counts[static_cast<size_t>(cid)]++;
    }
    std::vector<std::vector<idx_t>> new_lists(nlist);
    for (size_t cid = 0; cid < nlist; cid++) {
        new_lists[cid].reserve(counts[cid]);
    }
    for (size_t id = 0; id < assign.size(); id++) {
        const idx_t cid = assign[id];
        new_lists[static_cast<size_t>(cid)].push_back(static_cast<idx_t>(id));
    }
    lists.swap(new_lists);
}

static void log_ivfdata_list_stats(const IVFData& data, const char* tag) {
    std::vector<size_t> sizes;
    sizes.reserve(data.lists.size());
    size_t empty = 0;
    size_t le1 = 0;
    size_t le10 = 0;
    double sum = 0.0;
    for (const auto& list : data.lists) {
        const size_t sz = list.size();
        sizes.push_back(sz);
        sum += static_cast<double>(sz);
        if (sz == 0) empty++;
        if (sz <= 1) le1++;
        if (sz <= 10) le10++;
    }
    std::sort(sizes.begin(), sizes.end());
    const double mean = sizes.empty() ? 0.0 : sum / static_cast<double>(sizes.size());
    double var = 0.0;
    for (size_t sz : sizes) {
        const double diff = static_cast<double>(sz) - mean;
        var += diff * diff;
    }
    var = sizes.empty() ? 0.0 : var / static_cast<double>(sizes.size());
    auto pct = [&](double q) -> size_t {
        if (sizes.empty()) return 0;
        size_t pos = static_cast<size_t>(q * static_cast<double>(sizes.size() - 1));
        if (pos >= sizes.size()) pos = sizes.size() - 1;
        return sizes[pos];
    };
    fprintf(stderr,
            "stage1_list_stats tag=%s nlist=%zu ntotal=%zu mean=%.2f cv=%.6f empty=%zu le1=%zu le10=%zu p50=%zu p90=%zu p95=%zu p99=%zu max=%zu\n",
            tag,
            data.lists.size(),
            data.ntotal,
            mean,
            mean > 0.0 ? std::sqrt(var) / mean : 0.0,
            empty,
            le1,
            le10,
            pct(0.50),
            pct(0.90),
            pct(0.95),
            pct(0.99),
            sizes.empty() ? 0 : sizes.back());
}

static void compress_empty_clusters(
        IVFData& data,
        std::vector<idx_t>& assign) {
    const size_t nlist = data.lists.size();
    std::vector<int> valid;
    valid.reserve(nlist);
    for (size_t i = 0; i < nlist; i++) {
        if (!data.lists[i].empty()) {
            valid.push_back(static_cast<int>(i));
        }
    }
    if (valid.size() == nlist) {
        return;
    }

    std::vector<int> remap(nlist, -1);
    for (size_t i = 0; i < valid.size(); i++) {
        remap[static_cast<size_t>(valid[i])] = static_cast<int>(i);
    }

    std::vector<float> new_centroids(valid.size() * data.d);
    std::vector<std::vector<idx_t>> new_lists(valid.size());
    for (size_t i = 0; i < valid.size(); i++) {
        const size_t old = static_cast<size_t>(valid[i]);
        std::copy(
                data.centroids.begin() + old * data.d,
                data.centroids.begin() + (old + 1) * data.d,
                new_centroids.begin() + i * data.d);
        new_lists[i].swap(data.lists[old]);
    }
    data.centroids.swap(new_centroids);
    data.lists.swap(new_lists);
    data.nlist = data.lists.size();

    for (size_t id = 0; id < assign.size(); id++) {
        const int mapped = remap[static_cast<size_t>(assign[id])];
        FAISS_THROW_IF_NOT(mapped >= 0);
        assign[id] = mapped;
    }
}

#if 0  // Sample-only Stage 1 is disabled in the default merge path.
static void compress_empty_clusters_sample_only(IVFData& data) {
    const size_t nlist = data.lists.size();
    std::vector<int> valid;
    valid.reserve(nlist);
    for (size_t i = 0; i < nlist; i++) {
        if (!data.lists[i].empty()) {
            valid.push_back(static_cast<int>(i));
        }
    }
    if (valid.size() == nlist) {
        return;
    }
    FAISS_THROW_IF_NOT_MSG(!valid.empty(), "stage1_sample_only has no sampled vectors in any list");
    std::vector<float> new_centroids(valid.size() * data.d);
    std::vector<std::vector<idx_t>> new_lists(valid.size());
    for (size_t i = 0; i < valid.size(); i++) {
        const size_t old = static_cast<size_t>(valid[i]);
        std::copy(
                data.centroids.begin() + old * data.d,
                data.centroids.begin() + (old + 1) * data.d,
                new_centroids.begin() + i * data.d);
        new_lists[i].swap(data.lists[old]);
    }
    data.centroids.swap(new_centroids);
    data.lists.swap(new_lists);
    data.nlist = data.lists.size();
}

static IVFData make_stage1_sample_only_data(
        const IVFData& data,
        const faiss::MergeOptions& options) {
    FAISS_THROW_IF_NOT_MSG(
            options.training_ids != nullptr && options.n_training_vectors > 0,
            "stage1_sample_only requires training_ids");

    IVFData sample;
    sample.d = data.d;
    sample.nlist = data.nlist;
    sample.ntotal = data.ntotal;
    sample.metric = data.metric;
    sample.centroids = data.centroids;
    sample.lists.assign(data.nlist, {});
    sample.vectors_view = data.vectors.empty() ? data.vectors_view : data.vectors.data();
    sample.index_view = data.index_view;
    sample.id_loc = data.id_loc;

    std::vector<int> id_to_list(data.ntotal, -1);
    for (size_t list_no = 0; list_no < data.lists.size(); list_no++) {
        for (idx_t id : data.lists[list_no]) {
            FAISS_THROW_IF_NOT(id >= 0 && static_cast<size_t>(id) < data.ntotal);
            id_to_list[static_cast<size_t>(id)] = static_cast<int>(list_no);
        }
    }

    for (size_t i = 0; i < options.n_training_vectors; i++) {
        const idx_t id = options.training_ids[i];
        if (id < 0 || static_cast<size_t>(id) >= data.ntotal) {
            continue;
        }
        const int list_no = id_to_list[static_cast<size_t>(id)];
        if (list_no >= 0) {
            sample.lists[static_cast<size_t>(list_no)].push_back(id);
        }
    }
    compress_empty_clusters_sample_only(sample);
    return sample;
}
#endif

#if 0  // Disabled merge-threshold ablation; default merge skips close-centroid dedup.
struct DSU {
    std::vector<int> p;
    std::vector<int> r;
    explicit DSU(int n = 0) : p(n), r(n, 0) {
        std::iota(p.begin(), p.end(), 0);
    }
    int find(int x) {
        while (p[x] != x) {
            p[x] = p[p[x]];
            x = p[x];
        }
        return x;
    }
    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) {
            return;
        }
        if (r[a] < r[b]) {
            std::swap(a, b);
        }
        p[b] = a;
        if (r[a] == r[b]) {
            r[a]++;
        }
    }
};

struct NeighborGraph {
    std::vector<std::vector<int>> neighbors;
    std::vector<std::vector<float>> neighbors_dist2;
    std::vector<float> delta2;
};

static NeighborGraph prepare_centroid_neighbors_delta2_and_dists(
        const std::vector<float>& centroids,
        size_t nlist,
        size_t d,
        int neighbor_k) {
    FAISS_THROW_IF_NOT_MSG(neighbor_k > 0, "neighbor_k must be > 0");
    if (neighbor_k >= static_cast<int>(nlist)) {
        neighbor_k = static_cast<int>(nlist) - 1;
    }

    faiss::IndexFlatL2 index(static_cast<int>(d));
    index.add(static_cast<faiss::idx_t>(nlist), centroids.data());

    const int k = neighbor_k + 1;
    std::vector<faiss::idx_t> labels(nlist * k);
    std::vector<float> distances(nlist * k);
    index.search(
            static_cast<faiss::idx_t>(nlist),
            centroids.data(),
            k,
            distances.data(),
            labels.data());

    NeighborGraph g;
    g.neighbors.resize(nlist);
    g.neighbors_dist2.resize(nlist);
    g.delta2.assign(nlist, 0.0f);

    for (size_t i = 0; i < nlist; i++) {
        g.neighbors[i].reserve(neighbor_k);
        g.neighbors_dist2[i].reserve(neighbor_k);

        for (int j = 0; j < k; j++) {
            const faiss::idx_t id = labels[i * k + j];
            if (id < 0 || static_cast<size_t>(id) == i) {
                continue;
            }
            g.neighbors[i].push_back(static_cast<int>(id));
            g.neighbors_dist2[i].push_back(distances[i * k + j]);
            if (g.neighbors[i].size() == static_cast<size_t>(neighbor_k)) {
                g.delta2[i] = distances[i * k + j];
                break;
            }
        }
        if (g.neighbors[i].empty()) {
            g.delta2[i] = std::numeric_limits<float>::infinity();
        } else if (g.delta2[i] == 0.0f) {
            g.delta2[i] = g.neighbors_dist2[i].back();
        }
    }
    return g;
}

static inline bool is_mutual_neighbor(
        const std::vector<int>& a,
        int x) {
    for (int v : a) {
        if (v == x) {
            return true;
        }
    }
    return false;
}

static std::pair<bool, int> dedup_close_centroids_mutual_knn(
        IVFData& data,
        float beta,
        int neighbor_k) {
    if (data.nlist <= 1) {
        return {false, 0};
    }
    FAISS_THROW_IF_NOT_MSG(data.metric == faiss::METRIC_L2, "dedup supports METRIC_L2 only");
    if (!(beta > 0.0f)) {
        return {false, 0};
    }

    const size_t C = data.nlist;
    auto g = prepare_centroid_neighbors_delta2_and_dists(data.centroids, C, data.d, neighbor_k);

    DSU dsu(static_cast<int>(C));

    for (size_t i = 0; i < C; i++) {
        const float di = g.delta2[i];
        if (!std::isfinite(di) || di <= 0.0f) {
            continue;
        }
        const auto& ni = g.neighbors[i];
        const auto& ndi = g.neighbors_dist2[i];
        for (size_t t = 0; t < ni.size(); t++) {
            const int j = ni[t];
            if (j < 0) {
                continue;
            }
            const float dij = ndi[t];
            const float dj = g.delta2[static_cast<size_t>(j)];
            if (!std::isfinite(dj) || dj <= 0.0f) {
                continue;
            }
            if (!is_mutual_neighbor(g.neighbors[static_cast<size_t>(j)], static_cast<int>(i))) {
                continue;
            }
            const float thr = beta * std::min(di, dj);
            if (dij < thr) {
                dsu.unite(static_cast<int>(i), j);
            }
        }
    }

    std::unordered_map<int, int> root_to_gid;
    root_to_gid.reserve(C * 2);

    std::vector<int> root(static_cast<size_t>(C));
    for (int i = 0; i < static_cast<int>(C); i++) {
        root[static_cast<size_t>(i)] = dsu.find(i);
    }

    int newC = 0;
    for (int i = 0; i < static_cast<int>(C); i++) {
        const int r = root[static_cast<size_t>(i)];
        if (root_to_gid.find(r) == root_to_gid.end()) {
            root_to_gid[r] = newC++;
        }
    }

    if (static_cast<size_t>(newC) == C) {
        return {false, 0};
    }

    std::vector<std::vector<int>> groups(static_cast<size_t>(newC));
    for (int i = 0; i < static_cast<int>(C); i++) {
        const int gid = root_to_gid[root[static_cast<size_t>(i)]];
        groups[static_cast<size_t>(gid)].push_back(i);
    }

    std::vector<float> new_centroids(static_cast<size_t>(newC) * data.d, 0.0f);
    std::vector<std::vector<idx_t>> new_lists(static_cast<size_t>(newC));

    for (int gid = 0; gid < newC; gid++) {
        double wsum = 0.0;
        float* outc = new_centroids.data() + static_cast<size_t>(gid) * data.d;

        size_t merged_size = 0;
        for (int old : groups[static_cast<size_t>(gid)]) {
            const size_t sz = data.lists[static_cast<size_t>(old)].size();
            merged_size += sz;
        }
        new_lists[static_cast<size_t>(gid)].reserve(merged_size);

        for (int old : groups[static_cast<size_t>(gid)]) {
            const size_t sz = data.lists[static_cast<size_t>(old)].size();
            const double w = (sz > 0) ? static_cast<double>(sz) : 1.0;
            const float* c = data.centroids.data() + static_cast<size_t>(old) * data.d;
            for (size_t j = 0; j < data.d; j++) {
                outc[j] += static_cast<float>(w) * c[j];
            }
            wsum += w;

            auto& lst = data.lists[static_cast<size_t>(old)];
            if (!lst.empty()) {
                new_lists[static_cast<size_t>(gid)].insert(
                        new_lists[static_cast<size_t>(gid)].end(),
                        lst.begin(),
                        lst.end());
            }
        }

        if (wsum > 0.0) {
            const float inv = 1.0f / static_cast<float>(wsum);
            for (size_t j = 0; j < data.d; j++) {
                outc[j] *= inv;
            }
        }
    }

    data.centroids.swap(new_centroids);
    data.lists.swap(new_lists);
    data.nlist = data.lists.size();

    int merged = static_cast<int>(C - data.nlist);
    return {merged > 0, merged};
}

#endif

static void reservoir_sample_ids(
        const std::vector<idx_t>& ids,
        size_t sample_n,
        std::mt19937& rng,
        std::vector<idx_t>& out) {
    out.clear();
    const size_t n = ids.size();
    if (sample_n >= n) {
        out = ids;
        return;
    }
    out.reserve(sample_n);
    for (size_t i = 0; i < sample_n; i++) {
        out.push_back(ids[i]);
    }
    for (size_t i = sample_n; i < n; i++) {
        std::uniform_int_distribution<size_t> dist(0, i);
        const size_t j = dist(rng);
        if (j < sample_n) {
            out[j] = ids[i];
        }
    }
}

struct SplitResult {
    std::vector<float> centroids;
    std::vector<std::vector<idx_t>> lists;
};

/// Splits one cluster into two via k-means k=2. Returns true on success (list size >= 2).
static bool split_one_cluster_to_two(
        const IVFData& data,
        size_t cid,
        int seed,
        SplitResult& out) {
    const auto& ids = data.lists[cid];
    const size_t sz = ids.size();
    if (sz < 2) {
        return false;
    }
    const size_t d = data.d;
    std::vector<float> X(sz * d);
    for (size_t i = 0; i < sz; i++) {
        ivfdata_copy_vector(data, ids[i], X.data() + i * d);
    }
    faiss::Clustering clus(static_cast<int>(d), 2);
    clus.niter = 2;
    clus.nredo = 1;
    clus.seed = seed;
    faiss::IndexFlatL2 assigner(static_cast<int>(d));
    clus.train(static_cast<faiss::idx_t>(sz), X.data(), assigner);
    if (clus.centroids.size() != 2 * d) {
        return false;
    }
    faiss::IndexFlatL2 cindex(static_cast<int>(d));
    cindex.add(2, clus.centroids.data());
    std::vector<float> dists(sz);
    std::vector<faiss::idx_t> labels(sz);
    cindex.search(static_cast<faiss::idx_t>(sz), X.data(), 1, dists.data(), labels.data());
    out.centroids = clus.centroids;
    out.lists.resize(2);
    for (size_t i = 0; i < sz; i++) {
        int lab = static_cast<int>(labels[i]);
        lab = (lab <= 0) ? 0 : ((lab >= 1) ? 1 : lab);
        out.lists[static_cast<size_t>(lab)].push_back(ids[i]);
    }
    return true;
}

/// Replace cluster `cid` with two clusters from k-means k=2; `nlist` increases by 1.
static bool ivfdata_split_cluster_at(IVFData& data, size_t cid, int seed) {
    if (cid >= data.nlist || data.lists[cid].size() < 2) {
        return false;
    }
    SplitResult sr;
    if (!split_one_cluster_to_two(data, cid, seed, sr)) {
        return false;
    }
    const size_t d = data.d;
    std::vector<float> new_centroids((data.nlist + 1) * d);
    std::vector<std::vector<idx_t>> new_lists;
    new_lists.reserve(data.nlist + 1);
    size_t w = 0;
    for (size_t j = 0; j < data.nlist; j++) {
        if (j == cid) {
            for (size_t t = 0; t < 2; t++) {
                std::copy(
                        sr.centroids.begin() + t * d,
                        sr.centroids.begin() + (t + 1) * d,
                        new_centroids.begin() + w * d);
                new_lists.push_back(std::move(sr.lists[t]));
                w++;
            }
        } else {
            std::copy(
                    data.centroids.begin() + j * d,
                    data.centroids.begin() + (j + 1) * d,
                    new_centroids.begin() + w * d);
            new_lists.push_back(std::move(data.lists[j]));
            w++;
        }
    }
    FAISS_THROW_IF_NOT(w == data.nlist + 1);
    data.centroids.swap(new_centroids);
    data.lists.swap(new_lists);
    data.nlist = data.lists.size();
    return true;
}

static bool ivfdata_split_largest_nonempty_cluster(IVFData& data, int seed) {
    size_t best_cid = SIZE_MAX;
    size_t best_sz = 0;
    for (size_t i = 0; i < data.nlist; i++) {
        const size_t sz = data.lists[i].size();
        if (sz >= 2 && sz > best_sz) {
            best_sz = sz;
            best_cid = i;
        }
    }
    if (best_cid == SIZE_MAX) {
        return false;
    }
    return ivfdata_split_cluster_at(data, best_cid, seed);
}

static std::pair<bool, int> quota_split_clusters_kmeans(
        IVFData& data,
        size_t target_nlist,
        int niter,
        int nredo,
        int seed,
        size_t user_batch_size,
        int max_k_per_cluster) {
    if (data.nlist >= target_nlist) {
        return {false, 0};
    }
    if (data.nlist == 0 || data.ntotal == 0) {
        return {false, 0};
    }
    FAISS_THROW_IF_NOT_MSG(data.metric == faiss::METRIC_L2, "quota split supports METRIC_L2 only");

    const size_t C = data.nlist;
    const size_t T = target_nlist;
    // const size_t N = data.ntotal;  // Retained only for disabled split ablations.

    std::vector<size_t> sizes(C, 0);
    for (size_t i = 0; i < C; i++) {
        sizes[i] = data.lists[i].size();
    }

#if 0  // SSE-weighted split quota ablation is disabled.
    std::vector<double> split_scores(C, 0.0);
    double score_sum = 0.0;
    const bool use_sse_quota = split_quota_sse_alpha > 0.0;
    std::vector<float> tmp_vec(data.d);
    for (size_t i = 0; i < C; i++) {
        const size_t sz = sizes[i];
        double score = static_cast<double>(sz);
        if (use_sse_quota && sz > 0) {
            const float* c = data.centroids.data() + i * data.d;
            double sse = 0.0;
            for (idx_t id : data.lists[i]) {
                ivfdata_copy_vector(data, id, tmp_vec.data());
                for (size_t j = 0; j < data.d; j++) {
                    const double diff =
                            static_cast<double>(tmp_vec[j]) - static_cast<double>(c[j]);
                    sse += diff * diff;
                }
            }
            const double radius = std::sqrt(sse / static_cast<double>(sz));
            score = static_cast<double>(sz) *
                    std::pow(std::max(radius, 1e-12), split_quota_sse_alpha);
            if (!std::isfinite(score) || score <= 0.0) {
                score = static_cast<double>(sz);
            }
        }
        split_scores[i] = score;
        score_sum += score;
    }
#endif
    std::vector<double> split_scores(C, 0.0);
    double score_sum = 0.0;
    for (size_t i = 0; i < C; i++) {
        split_scores[i] = static_cast<double>(sizes[i]);
        score_sum += split_scores[i];
    }
    if (score_sum <= 0.0) {
        for (size_t i = 0; i < C; i++) {
            split_scores[i] = static_cast<double>(sizes[i]);
            score_sum += split_scores[i];
        }
    }

    std::vector<int> k_per(C, 1);

    max_k_per_cluster = std::max(1, max_k_per_cluster);
    for (size_t i = 0; i < C; i++) {
        const size_t sz = sizes[i];
        if (sz < 2) {
            k_per[i] = 1;
            continue;
        }
        const double ideal = (split_scores[i] * static_cast<double>(T)) / score_sum;
        int k = static_cast<int>(std::floor(ideal + 0.5));
        k = std::max(1, k);
        k = std::min(k, max_k_per_cluster);
        k = std::min<int>(k, static_cast<int>(sz));
        k_per[i] = k;
    }

    auto sum_k = [&]() -> size_t {
        size_t s = 0;
        for (int v : k_per) {
            s += static_cast<size_t>(v);
        }
        return s;
    };

    size_t cur = sum_k();

    if (cur < T) {
        const size_t need = T - cur;
        std::vector<size_t> order(C);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return split_scores[a] > split_scores[b];
        });
        size_t left = need;
        while (left > 0) {
            bool progressed = false;
            for (size_t idx = 0; idx < C && left > 0; idx++) {
                const size_t i = order[idx];
                const int cap = std::min<int>(max_k_per_cluster, static_cast<int>(sizes[i]));
                if (k_per[i] < cap) {
                    k_per[i]++;
                    left--;
                    progressed = true;
                }
            }
            if (!progressed) {
                break;
            }
        }
    } else if (cur > T) {
        size_t extra = cur - T;
        std::vector<size_t> order(C);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return split_scores[a] < split_scores[b];
        });
        size_t left = extra;
        while (left > 0) {
            bool progressed = false;
            for (size_t idx = 0; idx < C && left > 0; idx++) {
                const size_t i = order[idx];
                if (k_per[i] > 1) {
                    k_per[i]--;
                    left--;
                    progressed = true;
                }
            }
            if (!progressed) {
                break;
            }
        }
    }

    cur = sum_k();

#if 0  // Split debug output is disabled.
    if (split_debug) {
        size_t capped_by_max = 0;
        size_t capped_by_size = 0;
        size_t nontrivial = 0;
        int max_assigned = 0;
        for (size_t i = 0; i < C; i++) {
            max_assigned = std::max(max_assigned, k_per[i]);
            if (k_per[i] > 1) {
                nontrivial++;
            }
            if (sizes[i] >= static_cast<size_t>(max_k_per_cluster) &&
                k_per[i] == max_k_per_cluster) {
                capped_by_max++;
            }
            if (sizes[i] < static_cast<size_t>(max_k_per_cluster) &&
                k_per[i] == static_cast<int>(sizes[i])) {
                capped_by_size++;
            }
        }
        fprintf(stderr,
                "split_quota pass=%d before=%zu target=%zu quota_sum=%zu max_k=%d max_assigned=%d nontrivial=%zu capped_by_max=%zu capped_by_size=%zu\n",
                quota_pass, C, T, cur, max_k_per_cluster, max_assigned,
                nontrivial, capped_by_max, capped_by_size);
    }
#endif
    if (cur != T) {
        fprintf(stderr,
                "Error: quota split failed to match target_nlist (got %zu, target %zu)\n",
                cur, T);
    }

    const size_t assign_batch = std::max<size_t>(256, std::min<size_t>(user_batch_size, 2048));

    std::vector<SplitResult> results(C);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (long long cid_ll = 0; cid_ll < static_cast<long long>(C); cid_ll++) {
        const size_t cid = static_cast<size_t>(cid_ll);
        const int k = k_per[cid];
        const auto& ids = data.lists[cid];
        const size_t sz = ids.size();

        SplitResult res;

        if (k <= 1 || sz < 2) {
            res.centroids.resize(data.d);
            std::copy(
                    data.centroids.begin() + cid * data.d,
                    data.centroids.begin() + (cid + 1) * data.d,
                    res.centroids.begin());
            res.lists.resize(1);
            res.lists[0] = ids;
            results[cid] = std::move(res);
            continue;
        }

        std::mt19937 rng(static_cast<uint32_t>(seed + static_cast<int>(cid) * 1337));

        const size_t sample_per_center = 128;
        const size_t max_sample = 8192;
        size_t sample_n = sample_per_center * static_cast<size_t>(k);
        sample_n = std::min(sample_n, sz);
        sample_n = std::min(sample_n, max_sample);
        sample_n = std::max<size_t>(sample_n, static_cast<size_t>(k));

        std::vector<idx_t> sample_ids;
        reservoir_sample_ids(ids, sample_n, rng, sample_ids);

        std::vector<float> X(sample_ids.size() * data.d);
        for (size_t i = 0; i < sample_ids.size(); i++) {
            ivfdata_copy_vector(data, sample_ids[i], X.data() + i * data.d);
        }

        faiss::Clustering clus(static_cast<int>(data.d), k);
        clus.niter = std::max(1, niter);
        clus.nredo = std::max(1, nredo);
        clus.seed = seed + static_cast<int>(cid) * 17;

        faiss::IndexFlatL2 assigner(static_cast<int>(data.d));
        clus.train(static_cast<faiss::idx_t>(sample_ids.size()), X.data(), assigner);

        std::vector<float> centers;
        if (clus.centroids.size() == static_cast<size_t>(k) * data.d) {
            centers = clus.centroids;
        } else {
            centers.resize(static_cast<size_t>(k) * data.d);
            for (int t = 0; t < k; t++) {
                const idx_t id = sample_ids[static_cast<size_t>(t) % sample_ids.size()];
                ivfdata_copy_vector(
                        data, id, centers.data() + static_cast<size_t>(t) * data.d);
            }
        }

        faiss::IndexFlatL2 cindex(static_cast<int>(data.d));
        cindex.add(static_cast<faiss::idx_t>(k), centers.data());

        std::vector<std::vector<idx_t>> sublists(static_cast<size_t>(k));
        for (auto& sl : sublists) {
            sl.reserve(sz / static_cast<size_t>(k) + 8);
        }

        std::vector<float> sums(static_cast<size_t>(k) * data.d, 0.0f);
        std::vector<size_t> counts(static_cast<size_t>(k), 0);

        std::vector<float> Xb;
        std::vector<faiss::idx_t> labels;
        std::vector<float> dists;

        Xb.resize(assign_batch * data.d);
        labels.resize(assign_batch);
        dists.resize(assign_batch);

        for (size_t s = 0; s < sz; s += assign_batch) {
            const size_t e = std::min(sz, s + assign_batch);
            const size_t bc = e - s;

            for (size_t bi = 0; bi < bc; bi++) {
                const idx_t id = ids[s + bi];
                ivfdata_copy_vector(data, id, Xb.data() + bi * data.d);
            }

            cindex.search(
                    static_cast<faiss::idx_t>(bc),
                    Xb.data(),
                    1,
                    dists.data(),
                    labels.data());

            for (size_t bi = 0; bi < bc; bi++) {
                const int lab = static_cast<int>(labels[bi]);
                const idx_t id = ids[s + bi];
                const size_t l = static_cast<size_t>(std::max(0, std::min(k - 1, lab)));
                sublists[l].push_back(id);
                counts[l]++;
                const float* x = Xb.data() + bi * data.d;
                float* ss = sums.data() + l * data.d;
                for (size_t j = 0; j < data.d; j++) {
                    ss[j] += x[j];
                }
            }
        }

        auto largest_nonempty = [&]() -> int {
            size_t best = 0;
            int best_i = -1;
            for (int t = 0; t < k; t++) {
                if (counts[static_cast<size_t>(t)] > best) {
                    best = counts[static_cast<size_t>(t)];
                    best_i = t;
                }
            }
            return best_i;
        };

        for (int t = 0; t < k; t++) {
            if (counts[static_cast<size_t>(t)] != 0) {
                continue;
            }
            int src = largest_nonempty();
            if (src < 0 || counts[static_cast<size_t>(src)] <= 1) {
                break;
            }
            idx_t moved = sublists[static_cast<size_t>(src)].back();
            sublists[static_cast<size_t>(src)].pop_back();
            counts[static_cast<size_t>(src)]--;

            std::vector<float> moved_vec(data.d);
            ivfdata_copy_vector(data, moved, moved_vec.data());
            float* ssrc = sums.data() + static_cast<size_t>(src) * data.d;
            for (size_t j = 0; j < data.d; j++) {
                ssrc[j] -= moved_vec[j];
            }

            sublists[static_cast<size_t>(t)].push_back(moved);
            counts[static_cast<size_t>(t)] = 1;
            float* st = sums.data() + static_cast<size_t>(t) * data.d;
            for (size_t j = 0; j < data.d; j++) {
                st[j] += moved_vec[j];
            }
        }

        res.centroids.resize(static_cast<size_t>(k) * data.d, 0.0f);
        for (int t = 0; t < k; t++) {
            const size_t ct = counts[static_cast<size_t>(t)];
            float* outc = res.centroids.data() + static_cast<size_t>(t) * data.d;
            if (ct == 0) {
                const float* fallback = data.centroids.data() + cid * data.d;
                std::copy(fallback, fallback + data.d, outc);
                continue;
            }
            const float inv = 1.0f / static_cast<float>(ct);
            const float* ss = sums.data() + static_cast<size_t>(t) * data.d;
            for (size_t j = 0; j < data.d; j++) {
                outc[j] = ss[j] * inv;
            }
        }

        res.lists = std::move(sublists);
        results[cid] = std::move(res);
    }

    std::vector<float> new_centroids(T * data.d);
    std::vector<std::vector<idx_t>> new_lists;
    new_lists.reserve(T);

    size_t out_c = 0;
    for (size_t cid = 0; cid < C; cid++) {
        auto& res = results[cid];
        const size_t k = res.lists.size();
        FAISS_THROW_IF_NOT(k >= 1);
        for (size_t t = 0; t < k; t++) {
            FAISS_THROW_IF_NOT(out_c < T);
            std::copy(
                    res.centroids.begin() + t * data.d,
                    res.centroids.begin() + (t + 1) * data.d,
                    new_centroids.begin() + out_c * data.d);
            new_lists.push_back(std::move(res.lists[t]));
            out_c++;
        }
    }
    if (out_c < T) {
        new_centroids.resize(out_c * data.d);
    }

    data.centroids.swap(new_centroids);
    data.lists.swap(new_lists);
    data.nlist = data.lists.size();

    int created = static_cast<int>(out_c - C);
#if 0  // Split debug output is disabled.
    if (split_debug) {
        fprintf(stderr,
                "split_quota_result pass=%d before=%zu after=%zu created=%d target=%zu\n",
                quota_pass, C, data.nlist, created, T);
    }
#endif
    return {created > 0, created};
}

/// When `data.nlist < target`, repeatedly run `quota_split_clusters_kmeans`, then split
/// the largest list with at least two vectors until `data.nlist == target` or throw.
static void ensure_ivfdata_reaches_target_nlist(
        IVFData& data,
        size_t target,
        int split_kmeans_niter,
        int split_kmeans_nredo,
        int random_state,
        size_t batch_size,
        int split_max_k_per_cluster) {
    if (data.nlist >= target) {
        return;
    }
    int quota_pass = 0;
#if 0  // Split debug output is disabled.
    if (split_debug) {
        log_ivfdata_list_stats(data, "before_quota_split");
    }
#endif
    while (data.nlist < target && quota_pass < 256) {
        const size_t n_before = data.nlist;
        quota_split_clusters_kmeans(
                data,
                target,
                split_kmeans_niter,
                split_kmeans_nredo,
                random_state + quota_pass * 7919,
                batch_size,
                split_max_k_per_cluster);
        quota_pass++;
        if (data.nlist == n_before) {
            break;
        }
    }
#if 0  // Split debug output is disabled.
    if (split_debug) {
        log_ivfdata_list_stats(data, "after_quota_before_fallback");
    }
#endif
    int split_pass = 0;
    // const size_t fallback_start_nlist = data.nlist;  // Debug-only.
    while (data.nlist < target) {
        const bool ok = ivfdata_split_largest_nonempty_cluster(
                data, random_state + split_pass * 65537);
        if (!ok) {
            FAISS_THROW_FMT(
                    "IVF merge optimizer: cannot reach target_nlist=%zu, stuck at nlist=%zu "
                    "(no cluster has at least 2 vectors to split further)",
                    target,
                    data.nlist);
        }
        split_pass++;
    }
#if 0  // Split debug output is disabled.
    if (split_debug) {
        fprintf(stderr,
                "split_fallback start=%zu passes=%d final=%zu target=%zu\n",
                fallback_start_nlist, split_pass, data.nlist, target);
        log_ivfdata_list_stats(data, "after_fallback");
    }
#endif
}

static std::pair<bool, int> reduce_centroids_to_target_kmeans(
        IVFData& data,
        size_t target_nlist,
        int niter,
        int nredo,
        int seed,
        size_t user_batch_size,
        const float* training_vectors = nullptr,
        size_t n_training_vectors = 0,
        const std::string& weight_mode = "list_size",
        size_t weight_train_max = 0) {
    if (data.nlist <= target_nlist) {
        return {false, 0};
    }
    if (target_nlist == 0) {
        return {false, 0};
    }
    FAISS_THROW_IF_NOT_MSG(data.metric == faiss::METRIC_L2, "reduce supports METRIC_L2 only");

    const size_t C = data.nlist;
    const size_t T = target_nlist;
    const bool train_on_vectors =
            training_vectors != nullptr && n_training_vectors >= T;

    std::vector<float> weighted_centroid_train;
    if (!train_on_vectors && weight_mode != "none") {
        std::vector<double> scores(C, 1.0);
        double score_sum = 0.0;
        for (size_t i = 0; i < C; i++) {
            const double sz = static_cast<double>(std::max<size_t>(1, data.lists[i].size()));
            if (weight_mode == "list_size") {
                scores[i] = sz;
            } else if (weight_mode == "sqrt_list_size") {
                scores[i] = std::sqrt(sz);
            } else {
                FAISS_THROW_FMT("unknown stage1 reduce weight mode: %s", weight_mode.c_str());
            }
            score_sum += scores[i];
        }

        const size_t max_train = weight_train_max > 0 ? weight_train_max : C;
        const size_t desired_train = std::max(C, std::min(max_train, std::max(C, T * static_cast<size_t>(12))));
        std::vector<int> reps(C, 1);
        size_t rep_sum = C;
        for (size_t i = 0; i < C; i++) {
            const double ideal = scores[i] * static_cast<double>(desired_train) / score_sum;
            reps[i] = std::max(1, static_cast<int>(std::floor(ideal + 0.5)));
        }
        rep_sum = 0;
        for (int r : reps) {
            rep_sum += static_cast<size_t>(r);
        }
        while (rep_sum > desired_train) {
            size_t best = SIZE_MAX;
            int best_rep = 1;
            for (size_t i = 0; i < C; i++) {
                if (reps[i] > best_rep) {
                    best_rep = reps[i];
                    best = i;
                }
            }
            if (best == SIZE_MAX) {
                break;
            }
            reps[best]--;
            rep_sum--;
        }
        while (rep_sum < desired_train) {
            size_t best = 0;
            double best_score = -1.0;
            for (size_t i = 0; i < C; i++) {
                const double ratio = scores[i] / static_cast<double>(reps[i] + 1);
                if (ratio > best_score) {
                    best_score = ratio;
                    best = i;
                }
            }
            reps[best]++;
            rep_sum++;
        }

        weighted_centroid_train.reserve(rep_sum * data.d);
        for (size_t i = 0; i < C; i++) {
            const float* c = data.centroids.data() + i * data.d;
            for (int r = 0; r < reps[i]; r++) {
                weighted_centroid_train.insert(weighted_centroid_train.end(), c, c + data.d);
            }
        }
    }

    const float* train_x = train_on_vectors
            ? training_vectors
            : (!weighted_centroid_train.empty() ? weighted_centroid_train.data() : data.centroids.data());
    const size_t train_n = train_on_vectors
            ? n_training_vectors
            : (!weighted_centroid_train.empty() ? weighted_centroid_train.size() / data.d : C);

    faiss::Clustering clus(static_cast<int>(data.d), static_cast<int>(T));
    clus.niter = std::max(1, niter);
    clus.nredo = std::max(1, nredo);
    clus.seed = seed;

    faiss::IndexFlatL2 assigner(static_cast<int>(data.d));
    clus.train(static_cast<faiss::idx_t>(train_n), train_x, assigner);

    FAISS_THROW_IF_NOT(clus.centroids.size() == T * data.d);

    faiss::IndexFlatL2 cindex(static_cast<int>(data.d));
    cindex.add(static_cast<faiss::idx_t>(T), clus.centroids.data());

    const size_t batch = std::max<size_t>(256, std::min<size_t>(user_batch_size, 4096));
    std::vector<float> Xb(batch * data.d);
    std::vector<faiss::idx_t> labels(batch);
    std::vector<float> dists(batch);

    std::vector<int> map_old_to_new(C, 0);

    for (size_t s = 0; s < C; s += batch) {
        const size_t e = std::min(C, s + batch);
        const size_t bc = e - s;
        std::memcpy(Xb.data(), data.centroids.data() + s * data.d, bc * data.d * sizeof(float));
        cindex.search(
                static_cast<faiss::idx_t>(bc),
                Xb.data(),
                1,
                dists.data(),
                labels.data());
        for (size_t i = 0; i < bc; i++) {
            int lab = static_cast<int>(labels[i]);
            if (lab < 0) {
                lab = 0;
            }
            if (static_cast<size_t>(lab) >= T) {
                lab = static_cast<int>(T - 1);
            }
            map_old_to_new[s + i] = lab;
        }
    }

    std::vector<std::vector<idx_t>> new_lists(T);
    for (size_t old = 0; old < C; old++) {
        const int t = map_old_to_new[old];
        auto& dst = new_lists[static_cast<size_t>(t)];
        auto& src = data.lists[old];
        if (!src.empty()) {
            dst.insert(dst.end(), src.begin(), src.end());
        }
    }

    std::vector<float> new_centroids = clus.centroids;

    if (!train_on_vectors) {
        std::vector<double> wsum(T, 0.0);
        std::vector<double> acc(T * data.d, 0.0);

        for (size_t old = 0; old < C; old++) {
            const int t = map_old_to_new[old];
            const size_t tid = static_cast<size_t>(t);
            const size_t sz = data.lists[old].size();
            const double w = (sz > 0) ? static_cast<double>(sz) : 1.0;
            const float* oc = data.centroids.data() + old * data.d;
            double* a = acc.data() + tid * data.d;
            for (size_t j = 0; j < data.d; j++) {
                a[j] += w * static_cast<double>(oc[j]);
            }
            wsum[tid] += w;
        }

        for (size_t t = 0; t < T; t++) {
            const double w = wsum[t];
            if (w <= 0.0) {
                continue;
            }
            const double inv = 1.0 / w;
            const double* a = acc.data() + t * data.d;
            float* nc = new_centroids.data() + t * data.d;
            for (size_t j = 0; j < data.d; j++) {
                nc[j] = static_cast<float>(a[j] * inv);
            }
        }
    }

    data.centroids.swap(new_centroids);
    data.lists.swap(new_lists);
    data.nlist = data.lists.size();

    int reduced = static_cast<int>(C - T);
    return {reduced > 0, reduced};
}

static std::vector<idx_t> sample_ids_random_fraction(
        const IVFData& data,
        float fraction,
        int random_state);

#if 0  // Per-shard and training-vector Stage 1 reduction are disabled.
static std::pair<bool, int> reduce_centroids_per_shard_to_target_kmeans(
        IVFData& data,
        size_t target_nlist,
        size_t num_shards,
        int niter,
        int nredo,
        int seed,
        size_t user_batch_size,
        const float* training_vectors = nullptr,
        size_t n_training_vectors = 0) {
    if (data.nlist <= target_nlist) {
        return {false, 0};
    }
    FAISS_THROW_IF_NOT_MSG(num_shards > 0, "per-shard reduce requires num_shards > 0");
    FAISS_THROW_IF_NOT_MSG(
            data.nlist % num_shards == 0,
            "per-shard reduce requires source nlist divisible by num_shards");
    FAISS_THROW_IF_NOT_MSG(
            target_nlist >= num_shards,
            "per-shard reduce requires target_nlist >= num_shards");

    const size_t source_per_shard = data.nlist / num_shards;
    const size_t target_base = target_nlist / num_shards;
    const size_t target_rem = target_nlist % num_shards;

    std::vector<float> out_centroids;
    out_centroids.reserve(target_nlist * data.d);
    std::vector<std::vector<idx_t>> out_lists;
    out_lists.reserve(target_nlist);

    size_t total_out = 0;
    for (size_t shard = 0; shard < num_shards; shard++) {
        const size_t source_begin = shard * source_per_shard;
        const size_t local_target = target_base + (shard < target_rem ? 1 : 0);
        FAISS_THROW_IF_NOT_MSG(
                local_target > 0 && local_target <= source_per_shard,
                "invalid per-shard reduce target");

        IVFData sub;
        sub.d = data.d;
        sub.nlist = source_per_shard;
        sub.ntotal = data.ntotal;
        sub.metric = data.metric;
        sub.vectors = data.vectors;
        sub.vectors_view = data.vectors.empty() ? data.vectors_view : data.vectors.data();
        sub.index_view = data.index_view;
        sub.id_loc = data.id_loc;
        sub.centroids.resize(source_per_shard * data.d);
        std::copy(
                data.centroids.begin() + source_begin * data.d,
                data.centroids.begin() + (source_begin + source_per_shard) * data.d,
                sub.centroids.begin());
        sub.lists.reserve(source_per_shard);
        for (size_t i = 0; i < source_per_shard; i++) {
            sub.lists.push_back(data.lists[source_begin + i]);
        }

        reduce_centroids_to_target_kmeans(
                sub,
                local_target,
                niter,
                nredo,
                seed + static_cast<int>(shard) * 1009,
                user_batch_size,
                training_vectors,
                n_training_vectors);

        FAISS_THROW_IF_NOT(sub.nlist == local_target);
        out_centroids.insert(out_centroids.end(), sub.centroids.begin(), sub.centroids.end());
        for (auto& lst : sub.lists) {
            out_lists.push_back(std::move(lst));
        }
        total_out += sub.nlist;
    }

    FAISS_THROW_IF_NOT(total_out == target_nlist);
    data.centroids.swap(out_centroids);
    data.lists.swap(out_lists);
    data.nlist = data.lists.size();

    const int reduced = static_cast<int>(source_per_shard * num_shards - target_nlist);
    return {reduced > 0, reduced};
}

#endif

static void merge_stage1_adjust_nlist(
        IVFData& data,
        const faiss::MergeOptions& options,
        faiss::MergeRunStats* stats) {
    if (data.nlist == 0) {
        return;
    }

    if (options.target_nlist == 0) {
        return;
    }

    {
        std::vector<idx_t> assign = build_assign_from_lists(data.lists, data.ntotal);
        compress_empty_clusters(data, assign);
        rebuild_lists_from_assign(data.lists, assign);
    }
#if 0  // Split debug output is disabled.
    if (options.split_debug) {
        log_ivfdata_list_stats(data, "after_compress_before_dedup");
    }
#endif

    auto t0 = std::chrono::steady_clock::now();

    // Close-centroid dedup is intentionally disabled in the default method.
    if (stats) {
        stats->merge_close_s = 0.0;
    }

    {
        auto t_split0 = std::chrono::steady_clock::now();

        if (data.nlist > options.target_nlist) {
#if 0  // Alternate Stage 1 reduction inputs and per-shard mode are disabled.
            std::vector<float> stage1_training_vectors;
            const float* stage1_train_ptr = nullptr;
            size_t stage1_train_count = 0;
            if (options.stage1_reduce_use_training_vectors) {
                if (options.training_vectors != nullptr &&
                    options.n_training_vectors >= options.target_nlist) {
                    stage1_train_ptr = options.training_vectors;
                    stage1_train_count = options.n_training_vectors;
                } else {
                    std::vector<idx_t> sample_ids =
                            sample_ids_random_fraction(data, options.sample_fraction, options.random_state);
                    stage1_training_vectors.resize(sample_ids.size() * data.d);
                    for (size_t i = 0; i < sample_ids.size(); i++) {
                        ivfdata_copy_vector(
                                data,
                                sample_ids[i],
                                stage1_training_vectors.data() + i * data.d);
                    }
                    stage1_train_ptr = stage1_training_vectors.data();
                    stage1_train_count = sample_ids.size();
                }
            }
            if (options.stage1_reduce_per_shard) {
                reduce_centroids_per_shard_to_target_kmeans(
                        data,
                        options.target_nlist,
                        options.stage1_reduce_num_shards,
                        5,
                        1,
                        options.random_state,
                        static_cast<size_t>(options.batch_size),
                        stage1_train_ptr,
                        stage1_train_count);
            } else {
                reduce_centroids_to_target_kmeans(
                        data,
                        options.target_nlist,
                        5,
                        1,
                        options.random_state,
                        static_cast<size_t>(options.batch_size),
                        stage1_train_ptr,
                        stage1_train_count,
                        options.stage1_reduce_weight_mode,
                        options.stage1_reduce_weight_train_max);
            }
#else
            reduce_centroids_to_target_kmeans(
                    data,
                    options.target_nlist,
                    5,
                    1,
                    options.random_state,
                    static_cast<size_t>(options.batch_size),
                    nullptr,
                    0,
                    "list_size",
                    options.stage1_reduce_weight_train_max);
#endif
        } else if (data.nlist < options.target_nlist) {
            ensure_ivfdata_reaches_target_nlist(
                    data,
                    options.target_nlist,
                    5,
                    1,
                    options.random_state,
                    static_cast<size_t>(options.batch_size),
                    options.split_max_k_per_cluster);
        }
#if 0  // Split debug output is disabled.
        if (options.split_debug) {
            log_ivfdata_list_stats(data, "after_stage1_adjust");
        }
#endif

        auto t_split1 = std::chrono::steady_clock::now();
        if (stats) {
            stats->split_s = std::chrono::duration<double>(t_split1 - t_split0).count();
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    if (stats) {
        stats->optimize_s = std::chrono::duration<double>(t1 - t0).count();
    }

    FAISS_THROW_IF_NOT_MSG(
            data.nlist == options.target_nlist,
            "merge_stage1_adjust_nlist: nlist != target_nlist after optimization");
}

static std::vector<idx_t> sample_ids_random_fraction(
        const IVFData& data,
        float fraction,
        int random_state) {
    FAISS_THROW_IF_NOT(fraction > 0.0f && fraction <= 1.0f);
    const size_t n = static_cast<size_t>(data.ntotal);
    const size_t want = std::max<size_t>(1, static_cast<size_t>(n * fraction));

    if (want >= n) {
        std::vector<idx_t> all(n);
        std::iota(all.begin(), all.end(), static_cast<idx_t>(0));
        return all;
    }

    // Floyd's algorithm: draws `want` distinct ids out of [0, n) in O(want)
    // time/space, instead of materializing and shuffling all n ids just to
    // keep a small (e.g. default 5%) random subset.
    std::mt19937 rng(static_cast<uint32_t>(random_state));
    std::unordered_set<idx_t> selected;
    selected.reserve(want * 2);
    std::vector<idx_t> out;
    out.reserve(want);
    for (size_t j = n - want; j < n; j++) {
        std::uniform_int_distribution<idx_t> dist(0, static_cast<idx_t>(j));
        idx_t t = dist(rng);
        if (!selected.insert(t).second) {
            t = static_cast<idx_t>(j);
            selected.insert(t);
        }
        out.push_back(t);
    }
    return out;
}

static void train_kmeans_centroids_with_init(
        const std::vector<float>& train_x,
        size_t n_train,
        size_t d,
        size_t nlist,
        const std::vector<float>& init_centroids,
        int random_state,
        int niter,
        std::vector<float>& centroids_out) {
    FAISS_THROW_IF_NOT(n_train > 0);
    FAISS_THROW_IF_NOT(init_centroids.size() == nlist * d);
    faiss::Clustering clus(static_cast<int>(d), nlist);
    clus.verbose = false;
    clus.niter = std::max(1, niter);
    clus.nredo = 1;
    clus.seed = random_state;
    clus.centroids = init_centroids;
    faiss::IndexFlatL2 clus_index(static_cast<int>(d));
    clus.train(static_cast<faiss::idx_t>(n_train), train_x.data(), clus_index);
    FAISS_THROW_IF_NOT(clus.centroids.size() == nlist * d);
    centroids_out = clus.centroids;
}

static std::vector<std::vector<int>> build_src_to_tgt_neighbor_map(
        const std::vector<float>& src_centroids,
        size_t n_src,
        const std::vector<float>& tgt_centroids,
        size_t n_tgt,
        size_t d,
        int neighbor_k) {
    faiss::IndexFlatL2 tgt_index(static_cast<int>(d));
    tgt_index.add(static_cast<faiss::idx_t>(n_tgt), tgt_centroids.data());
    const int k = std::min(neighbor_k, static_cast<int>(n_tgt));
    std::vector<faiss::idx_t> labels(n_src * k);
    std::vector<float> dists(n_src * k);
    tgt_index.search(
            static_cast<faiss::idx_t>(n_src),
            src_centroids.data(),
            k,
            dists.data(),
            labels.data());
    std::vector<std::vector<int>> out(n_src);
    for (size_t i = 0; i < n_src; i++) {
        out[i].reserve(static_cast<size_t>(k));
        for (int j = 0; j < k; j++) {
            const faiss::idx_t id = labels[i * k + j];
            if (id >= 0) {
                out[i].push_back(static_cast<int>(id));
            }
        }
        if (out[i].empty()) {
            out[i].push_back(0);
        }
    }
    return out;
}

static void reassign_all_via_src_to_tgt_neighbors(
        IVFData& data,
        const std::vector<std::vector<int>>& src_to_tgt,
        const std::vector<float>& tgt_centroids,
        size_t n_tgt,
        int batch_size,
        bool return_final_assign_without_lists,
        faiss::MergeRemapBatchCallback batch_callback,
        void* batch_callback_user_data,
        faiss::MergeRunStats* stats = nullptr) {
    FAISS_THROW_IF_NOT(src_to_tgt.size() == data.nlist);
    FAISS_THROW_IF_NOT(tgt_centroids.size() == n_tgt * data.d);
    std::vector<idx_t> assign(data.ntotal, -1);
    const size_t bs = static_cast<size_t>(std::max(1, batch_size));
    std::vector<float> batch_vectors;
    std::vector<faiss::idx_t> batch_labels;
    std::vector<float> batch_dists;
    std::vector<float> cand_centroids;

    for (size_t src_cid = 0; src_cid < data.nlist; src_cid++) {
        const auto& lst = data.lists[src_cid];
        if (lst.empty()) {
            continue;
        }
        const auto& tgt_ids = src_to_tgt[src_cid];
        const auto t_pack0 = std::chrono::steady_clock::now();
        cand_centroids.resize(tgt_ids.size() * data.d);
        for (size_t i = 0; i < tgt_ids.size(); i++) {
            const int tid = tgt_ids[i];
            FAISS_THROW_IF_NOT(tid >= 0 && static_cast<size_t>(tid) < n_tgt);
            std::copy(
                    tgt_centroids.begin() + static_cast<size_t>(tid) * data.d,
                    tgt_centroids.begin() + (static_cast<size_t>(tid) + 1) * data.d,
                    cand_centroids.begin() + i * data.d);
        }
        faiss::IndexFlatL2 cand_index(static_cast<int>(data.d));
        cand_index.add(
                static_cast<faiss::idx_t>(tgt_ids.size()),
                cand_centroids.data());
        const auto t_pack1 = std::chrono::steady_clock::now();
        if (stats) {
            stats->remap_pack_s +=
                    std::chrono::duration<double>(t_pack1 - t_pack0).count();
        }

        for (size_t s = 0; s < lst.size(); s += bs) {
            const size_t e = std::min(lst.size(), s + bs);
            const size_t bc = e - s;
            const auto t_copy0 = std::chrono::steady_clock::now();
            batch_labels.resize(bc);
            batch_dists.resize(bc);
            const uint8_t* borrowed_codes = nullptr;
            size_t borrowed_list_no = 0;
            const float* search_vectors = ivfdata_contiguous_vector_slice(
                    data, lst, s, bc, &borrowed_codes, &borrowed_list_no);
            if (search_vectors == nullptr) {
                batch_vectors.resize(bc * data.d);
                for (size_t bi = 0; bi < bc; bi++) {
                    const idx_t id = lst[s + bi];
                    ivfdata_copy_vector(
                            data, id, batch_vectors.data() + bi * data.d);
                }
                search_vectors = batch_vectors.data();
            }
            const auto t_copy1 = std::chrono::steady_clock::now();
            if (stats) {
                stats->remap_vector_copy_s +=
                        std::chrono::duration<double>(t_copy1 - t_copy0).count();
            }

            const auto t_knn0 = std::chrono::steady_clock::now();
            cand_index.search(
                    static_cast<faiss::idx_t>(bc),
                    search_vectors,
                    1,
                    batch_dists.data(),
                    batch_labels.data());
            const auto t_knn1 = std::chrono::steady_clock::now();
            if (stats) {
                stats->remap_knn_s +=
                        std::chrono::duration<double>(t_knn1 - t_knn0).count();
            }

            const auto t_assign0 = std::chrono::steady_clock::now();
            std::vector<idx_t> batch_target_lists;
            if (batch_callback) {
                batch_target_lists.resize(bc);
            }
            for (size_t bi = 0; bi < bc; bi++) {
                const idx_t id = lst[s + bi];
                const int best_tgt =
                        tgt_ids[static_cast<size_t>(batch_labels[bi])];
                assign[static_cast<size_t>(id)] = static_cast<idx_t>(best_tgt);
                if (batch_callback) {
                    batch_target_lists[bi] = static_cast<idx_t>(best_tgt);
                }
            }
            if (batch_callback) {
                batch_callback(
                        batch_callback_user_data,
                        lst.data() + s,
                        search_vectors,
                        batch_target_lists.data(),
                        tgt_centroids.data(),
                        bc);
            }
            if (borrowed_codes != nullptr) {
                data.index_view->invlists->release_codes(
                        borrowed_list_no, borrowed_codes);
            }
            const auto t_assign1 = std::chrono::steady_clock::now();
            if (stats) {
                stats->remap_assign_write_s +=
                        std::chrono::duration<double>(t_assign1 - t_assign0).count();
            }
        }
    }

    data.nlist = n_tgt;
    data.centroids = tgt_centroids;
    data.final_assign = std::move(assign);
    if (return_final_assign_without_lists) {
        data.lists.clear();
    } else {
        data.lists.assign(n_tgt, {});
        rebuild_lists_from_assign(data.lists, data.final_assign);
    }
}


static std::vector<float> gather_sample_vectors(
        const IVFData& data,
        float sample_fraction,
        int random_state) {
    auto sample_ids_vec = sample_ids_random_fraction(data, sample_fraction, random_state);
    std::vector<float> train_x(sample_ids_vec.size() * data.d);
    for (size_t i = 0; i < sample_ids_vec.size(); i++) {
        ivfdata_copy_vector(
                data,
                sample_ids_vec[i],
                train_x.data() + i * data.d);
    }
    return train_x;
}

static std::vector<float> get_stage2_training_vectors(
        const IVFData& data,
        const faiss::MergeOptions& options) {
    if (options.sample_stage2_from_training_vectors) {
        FAISS_THROW_IF_NOT_MSG(
                options.training_vectors != nullptr &&
                        options.n_training_vectors == data.ntotal,
                "sample_stage2_from_training_vectors requires the full dense database");
        return gather_sample_vectors(
                data, options.sample_fraction, options.random_state);
    }
    if (options.training_vectors && options.n_training_vectors > 0) {
        const size_t n = options.n_training_vectors;
        return std::vector<float>(
                options.training_vectors,
                options.training_vectors + n * data.d);
    }
    return gather_sample_vectors(data, options.sample_fraction, options.random_state);
}

#if 0  // Centroid snap ablation is disabled.
static void snap_centroids_to_nearest_sample_points(
        const std::vector<float>& warm_centroids,
        size_t nlist,
        size_t d,
        const std::vector<float>& sample_x,
        size_t n_sample,
        std::vector<float>& snapped_out) {
    FAISS_THROW_IF_NOT(n_sample > 0);
    FAISS_THROW_IF_NOT(warm_centroids.size() == nlist * d);
    faiss::IndexFlatL2 sample_index(static_cast<int>(d));
    sample_index.add(static_cast<faiss::idx_t>(n_sample), sample_x.data());
    snapped_out.resize(nlist * d);
    std::vector<faiss::idx_t> labels(nlist);
    std::vector<float> dists(nlist);
    sample_index.search(
            static_cast<faiss::idx_t>(nlist),
            warm_centroids.data(),
            1,
            dists.data(),
            labels.data());
    for (size_t i = 0; i < nlist; i++) {
        const faiss::idx_t sid = labels[i];
        FAISS_THROW_IF_NOT(sid >= 0 && static_cast<size_t>(sid) < n_sample);
        std::copy(
                sample_x.begin() + static_cast<size_t>(sid) * d,
                sample_x.begin() + (static_cast<size_t>(sid) + 1) * d,
                snapped_out.begin() + i * d);
    }
}
#endif
// Remap-candidate diagnostics are disabled.
// static void write_remap_candidate_diagnostic(
//         const IVFData& data,
//         const std::vector<float>& source_centroids,
//         const std::vector<float>& target_centroids,
//         size_t target_nlist,
//         const std::vector<int>& raw_ks,
//         size_t max_vectors,
//         int random_state,
//         const std::string& out_path) {
//     if (out_path.empty() || max_vectors == 0 || raw_ks.empty()) {
//         return;
//     }
//     FAISS_THROW_IF_NOT(data.nlist > 0);
//     FAISS_THROW_IF_NOT(target_nlist > 0);
//     FAISS_THROW_IF_NOT(source_centroids.size() == data.nlist * data.d);
//     FAISS_THROW_IF_NOT(target_centroids.size() == target_nlist * data.d);
//
//     std::vector<int> ks;
//     ks.reserve(raw_ks.size());
//     int max_k = 1;
//     for (int k : raw_ks) {
//         k = std::max(1, std::min(k, static_cast<int>(target_nlist)));
//         ks.push_back(k);
//         max_k = std::max(max_k, k);
//     }
//
//     struct SampleItem {
//         idx_t id;
//         uint32_t src;
//     };
//
//     std::vector<SampleItem> samples;
//     samples.reserve(std::min(max_vectors, static_cast<size_t>(data.ntotal)));
//     std::mt19937 rng(static_cast<uint32_t>(random_state));
//     size_t seen = 0;
//     for (size_t src = 0; src < data.nlist; src++) {
//         for (idx_t id : data.lists[src]) {
//             seen++;
//             if (samples.size() < max_vectors) {
//                 samples.push_back({id, static_cast<uint32_t>(src)});
//             } else {
//                 std::uniform_int_distribution<size_t> dist(0, seen - 1);
//                 const size_t j = dist(rng);
//                 if (j < max_vectors) {
//                     samples[j] = {id, static_cast<uint32_t>(src)};
//                 }
//             }
//         }
//     }
//
//     faiss::IndexFlatL2 target_index(static_cast<int>(data.d));
//     target_index.add(static_cast<faiss::idx_t>(target_nlist), target_centroids.data());
//
//     std::vector<faiss::idx_t> src_top_labels(data.nlist * static_cast<size_t>(max_k));
//     std::vector<float> src_top_dists(data.nlist * static_cast<size_t>(max_k));
//     target_index.search(
//             static_cast<faiss::idx_t>(data.nlist),
//             source_centroids.data(),
//             max_k,
//             src_top_dists.data(),
//             src_top_labels.data());
//
//     std::vector<size_t> covered(ks.size(), 0);
//     std::vector<int> ranks;
//     ranks.reserve(samples.size());
//     size_t not_found = 0;
//     std::vector<faiss::idx_t> exact_for_sample(samples.size(), -1);
//
//     const size_t batch = 4096;
//     std::vector<float> xb(batch * data.d);
//     std::vector<faiss::idx_t> exact_labels(batch);
//     std::vector<float> exact_dists(batch);
//
//     for (size_t s = 0; s < samples.size(); s += batch) {
//         const size_t e = std::min(samples.size(), s + batch);
//         const size_t bs = e - s;
//         for (size_t i = 0; i < bs; i++) {
//             ivfdata_copy_vector(data, samples[s + i].id, xb.data() + i * data.d);
//         }
//         target_index.search(
//                 static_cast<faiss::idx_t>(bs),
//                 xb.data(),
//                 1,
//                 exact_dists.data(),
//                 exact_labels.data());
//         for (size_t i = 0; i < bs; i++) {
//             const faiss::idx_t exact = exact_labels[i];
//             exact_for_sample[s + i] = exact;
//             const size_t src = static_cast<size_t>(samples[s + i].src);
//             int rank = max_k + 1;
//             const faiss::idx_t* row = src_top_labels.data() + src * static_cast<size_t>(max_k);
//             for (int j = 0; j < max_k; j++) {
//                 if (row[j] == exact) {
//                     rank = j + 1;
//                     break;
//                 }
//             }
//             if (rank > max_k) {
//                 not_found++;
//             } else {
//                 ranks.push_back(rank);
//             }
//             for (size_t ki = 0; ki < ks.size(); ki++) {
//                 if (rank <= ks[ki]) {
//                     covered[ki]++;
//                 }
//             }
//         }
//     }
//
//     std::sort(ranks.begin(), ranks.end());
//     auto quantile_rank = [&](double q) -> int {
//         if (ranks.empty()) {
//             return 0;
//         }
//         const size_t pos = std::min(
//                 ranks.size() - 1,
//                 static_cast<size_t>(std::floor(q * static_cast<double>(ranks.size() - 1))));
//         return ranks[pos];
//     };
//     double mean_rank = 0.0;
//     for (int r : ranks) {
//         mean_rank += static_cast<double>(r);
//     }
//     if (!ranks.empty()) {
//         mean_rank /= static_cast<double>(ranks.size());
//     }
//
//
//     auto summarize_int_vector_json = [](std::vector<int> vals) -> std::string {
//         std::ostringstream os;
//         if (vals.empty()) {
//             return "{\"mean\":0,\"p50\":0,\"p90\":0,\"p95\":0,\"p99\":0,\"max\":0}";
//         }
//         std::sort(vals.begin(), vals.end());
//         double mean = 0.0;
//         for (int v : vals) {
//             mean += static_cast<double>(v);
//         }
//         mean /= static_cast<double>(vals.size());
//         auto qv = [&](double q) -> int {
//             const size_t pos = std::min(
//                     vals.size() - 1,
//                     static_cast<size_t>(std::floor(q * static_cast<double>(vals.size() - 1))));
//             return vals[pos];
//         };
//         os << "{\"mean\":" << mean
//            << ",\"p50\":" << qv(0.50)
//            << ",\"p90\":" << qv(0.90)
//            << ",\"p95\":" << qv(0.95)
//            << ",\"p99\":" << qv(0.99)
//            << ",\"max\":" << vals.back() << "}";
//         return os.str();
//     };
//
//     std::ostringstream union_json;
//     const std::vector<int> sample_per_list_values = {8, 16, 32, 64};
//     const std::vector<int> base_values = {50, 100};
//     const std::vector<int> cap_values = {150, 200};
//     const int vector_topk = 5;
//     union_json << "  \"sample_union_diagnostic\": {\n";
//     union_json << "    \"vector_topk\": " << vector_topk << ",\n";
//     union_json << "    \"by_sample_per_list\": {\n";
//     for (size_t spi = 0; spi < sample_per_list_values.size(); spi++) {
//         const int sample_per_list = sample_per_list_values[spi];
//         std::vector<std::vector<faiss::idx_t>> list_top_union(data.nlist);
//         std::vector<int> union_sizes;
//         union_sizes.reserve(data.nlist);
//         std::mt19937 list_rng(static_cast<uint32_t>(random_state + sample_per_list * 1009));
//         std::vector<float> one_x(static_cast<size_t>(std::max<size_t>(1, sample_per_list)) * data.d);
//         std::vector<faiss::idx_t> vec_labels(static_cast<size_t>(std::max<size_t>(1, sample_per_list)) * vector_topk);
//         std::vector<float> vec_dists(static_cast<size_t>(std::max<size_t>(1, sample_per_list)) * vector_topk);
//         for (size_t src = 0; src < data.nlist; src++) {
//             std::vector<idx_t> picked;
//             reservoir_sample_ids(
//                     data.lists[src],
//                     std::min<size_t>(static_cast<size_t>(sample_per_list), data.lists[src].size()),
//                     list_rng,
//                     picked);
//             std::unordered_set<faiss::idx_t> uniq;
//             uniq.reserve(static_cast<size_t>(sample_per_list * vector_topk * 2 + 16));
//             if (!picked.empty()) {
//                 for (size_t i = 0; i < picked.size(); i++) {
//                     ivfdata_copy_vector(data, picked[i], one_x.data() + i * data.d);
//                 }
//                 target_index.search(
//                         static_cast<faiss::idx_t>(picked.size()),
//                         one_x.data(),
//                         vector_topk,
//                         vec_dists.data(),
//                         vec_labels.data());
//                 for (size_t i = 0; i < picked.size(); i++) {
//                     for (int j = 0; j < vector_topk; j++) {
//                         const faiss::idx_t lid = vec_labels[i * vector_topk + j];
//                         if (lid >= 0) {
//                             uniq.insert(lid);
//                         }
//                     }
//                 }
//             }
//             list_top_union[src].assign(uniq.begin(), uniq.end());
//             union_sizes.push_back(static_cast<int>(list_top_union[src].size()));
//         }
//
//         union_json << "      \"" << sample_per_list << "\": {\n";
//         union_json << "        \"union_size\": " << summarize_int_vector_json(union_sizes) << ",\n";
//         union_json << "        \"by_base_and_cap\": {\n";
//         bool first_combo = true;
//         for (int base_k : base_values) {
//             for (int cap_k : cap_values) {
//                 if (!first_combo) {
//                     union_json << ",\n";
//                 }
//                 first_combo = false;
//                 size_t cov = 0;
//                 std::vector<int> final_sizes;
//                 final_sizes.reserve(data.nlist);
//                 std::vector<std::unordered_set<faiss::idx_t>> final_sets(data.nlist);
//                 for (size_t src = 0; src < data.nlist; src++) {
//                     auto& fs = final_sets[src];
//                     fs.reserve(static_cast<size_t>(cap_k * 2));
//                     const faiss::idx_t* row = src_top_labels.data() + src * static_cast<size_t>(max_k);
//                     for (int j = 0; j < std::min(base_k, cap_k); j++) {
//                         if (row[j] >= 0) {
//                             fs.insert(row[j]);
//                         }
//                     }
//                     for (faiss::idx_t cand : list_top_union[src]) {
//                         if (static_cast<int>(fs.size()) >= cap_k) {
//                             break;
//                         }
//                         fs.insert(cand);
//                     }
//                     final_sizes.push_back(static_cast<int>(fs.size()));
//                 }
//                 for (size_t i = 0; i < samples.size(); i++) {
//                     const auto exact = exact_for_sample[i];
//                     const size_t src = static_cast<size_t>(samples[i].src);
//                     if (exact >= 0 && final_sets[src].find(exact) != final_sets[src].end()) {
//                         cov++;
//                     }
//                 }
//                 const double coverage = samples.empty() ? 0.0 :
//                         static_cast<double>(cov) / static_cast<double>(samples.size());
//                 union_json << "          \"base" << base_k << "_cap" << cap_k << "\": {"
//                            << "\"covered\":" << cov
//                            << ",\"coverage\":" << coverage
//                            << ",\"final_size\":" << summarize_int_vector_json(final_sizes)
//                            << "}";
//             }
//         }
//         union_json << "\n        }\n";
//         union_json << "      }";
//         if (spi + 1 != sample_per_list_values.size()) {
//             union_json << ",";
//         }
//         union_json << "\n";
//     }
//     union_json << "    }\n";
//     union_json << "  }";
//
//
//     std::ofstream f(out_path);
//     FAISS_THROW_IF_NOT_MSG(f.good(), "failed to open remap candidate diagnostic output");
//     f << "{\n";
//     f << "  \"n_sample\": " << samples.size() << ",\n";
//     f << "  \"max_vectors_requested\": " << max_vectors << ",\n";
//     f << "  \"source_nlist\": " << data.nlist << ",\n";
//     f << "  \"target_nlist\": " << target_nlist << ",\n";
//     f << "  \"max_k\": " << max_k << ",\n";
//     f << "  \"rank_summary\": {\n";
//     f << "    \"found_within_max_k\": " << ranks.size() << ",\n";
//     f << "    \"not_found_within_max_k\": " << not_found << ",\n";
//     f << "    \"mean\": " << mean_rank << ",\n";
//     f << "    \"p50\": " << quantile_rank(0.50) << ",\n";
//     f << "    \"p90\": " << quantile_rank(0.90) << ",\n";
//     f << "    \"p95\": " << quantile_rank(0.95) << ",\n";
//     f << "    \"p99\": " << quantile_rank(0.99) << "\n";
//     f << "  },\n";
//     f << "  \"by_k\": {\n";
//     for (size_t i = 0; i < ks.size(); i++) {
//         const double cov = samples.empty() ? 0.0 :
//                 static_cast<double>(covered[i]) / static_cast<double>(samples.size());
//         f << "    \"" << ks[i] << "\": {\"covered\": " << covered[i]
//           << ", \"missed\": " << (samples.size() - covered[i])
//           << ", \"coverage\": " << cov << "}";
//         if (i + 1 != ks.size()) {
//             f << ",";
//         }
//         f << "\n";
//     }
//     f << "  },\n";
//     f << union_json.str() << "\n";
//     f << "}\n";
//     fprintf(stderr, "remap candidate diagnostic -> %s\n", out_path.c_str());
// }
//

static void merge_stage2_current_lists_kmeans_remap(
        IVFData& data,
        const faiss::MergeOptions& options,
        faiss::MergeRunStats* stats) {
    const size_t target = options.target_nlist;
    FAISS_THROW_IF_NOT(target > 0);
    FAISS_THROW_IF_NOT_MSG(
            data.nlist == target,
            "merge_stage2_current_lists_kmeans_remap: stage1 nlist != target_nlist");

    const std::vector<float> warm_centroids = data.centroids;

    if (stats) {
        stats->remap_snap_to_data_s = 0.0;
    }

    std::vector<float> tgt_centroids;
    if (options.reference_centroids && options.n_reference_centroids == target) {
        tgt_centroids.assign(
                options.reference_centroids,
                options.reference_centroids + target * data.d);
        if (stats) {
            stats->remap_centroid_train_s = 0.0;
        }
    } else {
        auto sample_x = get_stage2_training_vectors(data, options);
        const size_t n_sample = sample_x.size() / data.d;
        std::vector<float> init_centroids = warm_centroids;
#if 0  // Centroid snap ablation is disabled.
        if (options.snap_centroids_to_data) {
            const auto t_snap0 = std::chrono::steady_clock::now();
            snap_centroids_to_nearest_sample_points(
                    warm_centroids,
                    target,
                    data.d,
                    sample_x,
                    n_sample,
                    init_centroids);
            const auto t_snap1 = std::chrono::steady_clock::now();
            if (stats) {
                stats->remap_snap_to_data_s =
                        std::chrono::duration<double>(t_snap1 - t_snap0).count();
            }
        }
#endif
        const auto t_cent0 = std::chrono::steady_clock::now();
        train_kmeans_centroids_with_init(
                sample_x,
                n_sample,
                data.d,
                target,
                init_centroids,
                options.random_state,
                options.sample_kmeans_niter,
                tgt_centroids);
        const auto t_cent1 = std::chrono::steady_clock::now();
        if (stats) {
            stats->remap_centroid_train_s =
                    std::chrono::duration<double>(t_cent1 - t_cent0).count();
        }
    }

// Remap-candidate diagnostics are disabled.
//     if (!options.remap_candidate_diagnostic_path.empty()) {
//         write_remap_candidate_diagnostic(
//                 data,
//                 warm_centroids,
//                 tgt_centroids,
//                 target,
//                 options.remap_candidate_diagnostic_ks,
//                 options.remap_candidate_diagnostic_max_vectors,
//                 options.random_state,
//                 options.remap_candidate_diagnostic_path);
//     }

    const auto t_map0 = std::chrono::steady_clock::now();
    auto src_to_tgt = build_src_to_tgt_neighbor_map(
            warm_centroids,
            target,
            tgt_centroids,
            target,
            data.d,
            options.remap_neighbor_k);
    const auto t_map1 = std::chrono::steady_clock::now();
    if (stats) {
        stats->remap_neighbor_map_s =
                std::chrono::duration<double>(t_map1 - t_map0).count();
    }

    const auto t_re0 = std::chrono::steady_clock::now();
    reassign_all_via_src_to_tgt_neighbors(
            data,
            src_to_tgt,
            tgt_centroids,
            target,
            options.batch_size,
            options.return_final_assign_without_lists,
            options.remap_batch_callback,
            options.remap_batch_callback_user_data,
            stats);
    const auto t_re1 = std::chrono::steady_clock::now();
    if (stats) {
        stats->remap_full_reassign_s =
                std::chrono::duration<double>(t_re1 - t_re0).count();
    }
}

#if 0  // Preserve-source and exact-remap alternatives are disabled.
static void merge_stage2_preserve_source_kmeans_remap(
        IVFData& data,
        const faiss::MergeOptions& options,
        faiss::MergeRunStats* stats,
        const std::vector<float>& source_centroids,
        const std::vector<std::vector<idx_t>>& source_lists,
        size_t source_nlist) {
    const size_t target = options.target_nlist;
    FAISS_THROW_IF_NOT(target > 0);
    FAISS_THROW_IF_NOT_MSG(
            data.nlist == target,
            "merge_stage2_preserve_source_kmeans_remap: stage1 nlist != target_nlist");
    FAISS_THROW_IF_NOT(source_nlist > 0);
    FAISS_THROW_IF_NOT(source_centroids.size() == source_nlist * data.d);
    FAISS_THROW_IF_NOT(source_lists.size() == source_nlist);

    const std::vector<float> warm_centroids = data.centroids;
    auto sample_x = get_stage2_training_vectors(data, options);
    const size_t n_sample = sample_x.size() / data.d;

    std::vector<float> init_centroids = warm_centroids;
    if (stats) {
        stats->remap_snap_to_data_s = 0.0;
    }
#if 0  // Centroid snap ablation is disabled.
    if (options.snap_centroids_to_data) {
        const auto t_snap0 = std::chrono::steady_clock::now();
        snap_centroids_to_nearest_sample_points(
                warm_centroids,
                target,
                data.d,
                sample_x,
                n_sample,
                init_centroids);
        const auto t_snap1 = std::chrono::steady_clock::now();
        if (stats) {
            stats->remap_snap_to_data_s =
                    std::chrono::duration<double>(t_snap1 - t_snap0).count();
        }
    }
#endif
    const auto t_cent0 = std::chrono::steady_clock::now();
    std::vector<float> tgt_centroids;
    train_kmeans_centroids_with_init(
            sample_x,
            n_sample,
            data.d,
            target,
            init_centroids,
            options.random_state,
            options.sample_kmeans_niter,
            tgt_centroids);
    const auto t_cent1 = std::chrono::steady_clock::now();
    if (stats) {
        stats->remap_centroid_train_s =
                std::chrono::duration<double>(t_cent1 - t_cent0).count();
    }

// Remap-candidate diagnostics are disabled.
//     if (!options.remap_candidate_diagnostic_path.empty()) {
//         IVFData diagnostic_data;
//         diagnostic_data.d = data.d;
//         diagnostic_data.nlist = source_nlist;
//         diagnostic_data.ntotal = data.ntotal;
//         diagnostic_data.metric = data.metric;
//         diagnostic_data.centroids = source_centroids;
//         diagnostic_data.lists = source_lists;
//         diagnostic_data.vectors_view = data.vectors.empty() ? data.vectors_view : data.vectors.data();
//         diagnostic_data.index_view = data.index_view;
//         diagnostic_data.id_loc = data.id_loc;
//         write_remap_candidate_diagnostic(
//                 diagnostic_data,
//                 source_centroids,
//                 tgt_centroids,
//                 target,
//                 options.remap_candidate_diagnostic_ks,
//                 options.remap_candidate_diagnostic_max_vectors,
//                 options.random_state,
//                 options.remap_candidate_diagnostic_path);
//     }

    const auto t_map0 = std::chrono::steady_clock::now();
    auto src_to_tgt = build_src_to_tgt_neighbor_map(
            source_centroids,
            source_nlist,
            tgt_centroids,
            target,
            data.d,
            options.remap_neighbor_k);
    const auto t_map1 = std::chrono::steady_clock::now();
    if (stats) {
        stats->remap_neighbor_map_s =
                std::chrono::duration<double>(t_map1 - t_map0).count();
    }

    data.nlist = source_nlist;
    data.centroids = source_centroids;
    data.lists = source_lists;

    const auto t_re0 = std::chrono::steady_clock::now();
    reassign_all_via_src_to_tgt_neighbors(
            data,
            src_to_tgt,
            tgt_centroids,
            target,
            options.batch_size,
            options.return_final_assign_without_lists,
            options.remap_batch_callback,
            options.remap_batch_callback_user_data,
            stats);
    const auto t_re1 = std::chrono::steady_clock::now();
    if (stats) {
        stats->remap_full_reassign_s =
                std::chrono::duration<double>(t_re1 - t_re0).count();
    }
}

static void reassign_all_exact_to_current_centroids(
        IVFData& data,
        int batch_size,
        faiss::MergeRunStats* stats = nullptr) {
    FAISS_THROW_IF_NOT(data.nlist > 0);
    FAISS_THROW_IF_NOT(data.centroids.size() == data.nlist * data.d);
    const auto t0 = std::chrono::steady_clock::now();
    faiss::IndexFlatL2 target_index(static_cast<int>(data.d));
    target_index.add(static_cast<faiss::idx_t>(data.nlist), data.centroids.data());

    std::vector<idx_t> assign(data.ntotal, -1);
    const size_t bs = static_cast<size_t>(std::max(1, batch_size));
    std::vector<float> xb(bs * data.d);
    std::vector<faiss::idx_t> labels(bs);
    std::vector<float> dists(bs);

    size_t offset = 0;
    while (offset < data.ntotal) {
        const size_t cur = std::min(bs, data.ntotal - offset);
        for (size_t i = 0; i < cur; i++) {
            ivfdata_copy_vector(data, static_cast<idx_t>(offset + i), xb.data() + i * data.d);
        }
        target_index.search(
                static_cast<faiss::idx_t>(cur),
                xb.data(),
                1,
                dists.data(),
                labels.data());
        for (size_t i = 0; i < cur; i++) {
            assign[offset + i] = labels[i];
        }
        offset += cur;
    }

    data.lists.assign(data.nlist, {});
    rebuild_lists_from_assign(data.lists, assign);
    const auto t1 = std::chrono::steady_clock::now();
    if (stats) {
        stats->remap_full_reassign_s =
                std::chrono::duration<double>(t1 - t0).count();
    }
}

#endif

static void merge_full_on_ivfdata(
        IVFData& data,
        faiss::MergeOptions options,
        faiss::MergeRunStats* stats) {
    if (options.target_nlist == 0) {
        options.target_nlist = data.nlist;
    }
    FAISS_THROW_IF_NOT(options.target_nlist > 0);
    FAISS_THROW_IF_NOT_MSG(
            options.remap_neighbor_k > 0,
            "remap_neighbor_k must be provided and greater than zero");

    {
        std::vector<idx_t> assign = build_assign_from_lists(data.lists, data.ntotal);
        compress_empty_clusters(data, assign);
        rebuild_lists_from_assign(data.lists, assign);
    }

#if 0  // Historical switchable paths retained only for rollback/reference.
    const size_t source_nlist = data.nlist;
    std::vector<float> source_centroids = data.centroids;
    std::vector<std::vector<idx_t>> source_lists = data.lists;

    if (options.stage1_sample_only) {
        IVFData sample_stage1 = make_stage1_sample_only_data(data, options);
        merge_stage1_adjust_nlist(sample_stage1, options, stats);
        data.nlist = sample_stage1.nlist;
        data.centroids = sample_stage1.centroids;
        data.lists = sample_stage1.lists;
        if (options.use_split_centroids_final_exact_assign) {
            reassign_all_exact_to_current_centroids(data, options.batch_size, stats);
        } else {
            merge_stage2_preserve_source_kmeans_remap(
                    data, options, stats, source_centroids, source_lists, source_nlist);
        }
    } else {
        merge_stage1_adjust_nlist(data, options, stats);
        if (options.use_split_centroids_final_exact_assign) {
            reassign_all_exact_to_current_centroids(data, options.batch_size, stats);
        } else if (options.force_current_lists_remap || source_nlist < options.target_nlist) {
            merge_stage2_current_lists_kmeans_remap(data, options, stats);
        } else {
            merge_stage2_preserve_source_kmeans_remap(
                    data, options, stats, source_centroids, source_lists, source_nlist);
        }
    }
#endif
    merge_stage1_adjust_nlist(data, options, stats);
    merge_stage2_current_lists_kmeans_remap(data, options, stats);
    FAISS_THROW_IF_NOT_MSG(
            data.nlist == options.target_nlist,
            "merge_full_on_ivfdata: nlist != target_nlist after remap");
}

static void materialize_invlists_from_ivfdata(
        faiss::ArrayInvertedLists& invlists,
        const IVFData& data) {
    const size_t code_size = data.d * sizeof(float);
    FAISS_THROW_IF_NOT(invlists.nlist == data.nlist);
    FAISS_THROW_IF_NOT(invlists.code_size == code_size);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (long long cid_ll = 0; cid_ll < static_cast<long long>(data.nlist); cid_ll++) {
        const size_t cid = static_cast<size_t>(cid_ll);
        const auto& lst = data.lists[cid];
        const size_t n = lst.size();
        if (n == 0) {
            continue;
        }

        invlists.resize(cid, n);
        idx_t* id_ptr = invlists.ids[cid].data();
        uint8_t* code_ptr = invlists.codes[cid].data();

        if (!data.vectors.empty()) {
            for (size_t i = 0; i < n; i++) {
                id_ptr[i] = lst[i];
                ivfdata_copy_vector(
                        data,
                        lst[i],
                        reinterpret_cast<float*>(code_ptr + i * code_size));
            }
        } else {
            FAISS_THROW_IF_NOT(data.index_view != nullptr);
            faiss::InvertedLists* src_invlists = data.index_view->invlists;
            FAISS_THROW_IF_NOT(src_invlists != nullptr);
            for (size_t i = 0; i < n; i++) {
                id_ptr[i] = lst[i];
                const VectorLoc& loc = data.id_loc[static_cast<size_t>(lst[i])];
                const uint8_t* src_codes = src_invlists->get_codes(loc.list_no);
                FAISS_THROW_IF_NOT(src_codes != nullptr);
                std::memcpy(
                        code_ptr + i * code_size,
                        src_codes + static_cast<size_t>(loc.off) * code_size,
                        code_size);
            }
        }
    }
}

static std::unique_ptr<faiss::IndexIVFFlat> sync_ivfdata_to_index(IVFData& data) {
    if (data.index_owned && !data.vectors.empty()) {
        FAISS_THROW_MSG("IVFData cannot own both dense vectors and an index view");
    }

    const size_t nlist = data.lists.size();
    FAISS_THROW_IF_NOT_MSG(
            nlist == data.nlist,
            "IVFData nlist mismatch between metadata and lists");
    FAISS_THROW_IF_NOT(data.centroids.size() == nlist * data.d);

    auto quantizer = std::make_unique<faiss::IndexFlatL2>(static_cast<int>(data.d));
    quantizer->add(static_cast<faiss::idx_t>(nlist), data.centroids.data());

    auto invlists = std::make_unique<faiss::ArrayInvertedLists>(
            nlist, data.d * sizeof(float));
    materialize_invlists_from_ivfdata(*invlists, data);

    auto index = std::make_unique<faiss::IndexIVFFlat>(
            quantizer.get(),
            static_cast<int>(data.d),
            static_cast<int>(nlist),
            data.metric,
            false);
    index->is_trained = true;
    index->replace_invlists(invlists.release(), true);
    index->ntotal = static_cast<faiss::idx_t>(data.ntotal);
    index->own_fields = true;
    index->quantizer = quantizer.release();

    data.index_owned.reset();
    data.index_view = nullptr;
    data.id_loc.clear();

    return index;
}

} // namespace

namespace faiss {

// ============================================================================
// Helper functions for IVF merge (used by IVFPQ merge)
// ============================================================================

void finalize_merge_run_stats(MergeRunStats* stats) {
    if (!stats) {
        return;
    }
    stats->merge_total_s = stats->optimize_s + stats->remap_snap_to_data_s +
            stats->remap_centroid_train_s + stats->remap_neighbor_map_s +
            stats->remap_full_reassign_s;
}

void merge_ivf_data(
        IVFDataForMerge& data,
        const MergeOptions& options,
        MergeRunStats* stats) {
    if (ivfflat_merge_mt::use_mt_merge()) {
        return ivfflat_merge_mt::merge_ivf_data(data, options, stats);
    }
    if (data.nlist == 0 || options.target_nlist == 0) {
        return;
    }

    const auto t0 = std::chrono::steady_clock::now();

    IVFData internal_data;
    internal_data.d = data.d;
    internal_data.nlist = data.nlist;
    internal_data.ntotal = data.ntotal;
    internal_data.metric = data.metric;
    internal_data.centroids = std::move(data.centroids);
    internal_data.lists = std::move(data.lists);
    internal_data.vectors = std::move(data.vectors);
    internal_data.vectors_view = data.vectors_view;
    if (!internal_data.vectors.empty()) {
        ivfdata_from_dense_vectors(internal_data);
    } else {
        FAISS_THROW_IF_NOT_MSG(
                internal_data.vectors_view != nullptr,
                "merge_ivf_data requires dense vectors or vectors_view");
    }

    merge_full_on_ivfdata(internal_data, options, stats);
    finalize_merge_run_stats(stats);

    data.d = internal_data.d;
    data.nlist = internal_data.nlist;
    data.ntotal = internal_data.ntotal;
    data.metric = internal_data.metric;
    data.centroids = std::move(internal_data.centroids);
    data.lists = std::move(internal_data.lists);
    data.vectors = std::move(internal_data.vectors);
    data.vectors_view = internal_data.vectors_view;
    data.final_assign = std::move(internal_data.final_assign);

    if (stats) {
        const double dt = std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() - t0)
                                  .count();
        stats->ivf_merge_s += dt;
    }
}

void ivfflat_concat_merge(IndexIVFFlat& dst, IndexIVFFlat& src, idx_t add_id) {
    FAISS_THROW_IF_NOT(&dst != &src);
    FAISS_THROW_IF_NOT(dst.is_trained && src.is_trained);
    FAISS_THROW_IF_NOT(dst.d == src.d);
    FAISS_THROW_IF_NOT(dst.metric_type == src.metric_type);
    FAISS_THROW_IF_NOT(dst.code_size == src.code_size);
    FAISS_THROW_IF_NOT(dst.by_residual == src.by_residual);
    FAISS_THROW_IF_NOT(dst.quantizer && src.quantizer);
    FAISS_THROW_IF_NOT(dst.quantizer->ntotal == dst.nlist);
    FAISS_THROW_IF_NOT(src.quantizer->ntotal == src.nlist);
    FAISS_THROW_IF_NOT_MSG(
            dst.direct_map.no() && src.direct_map.no(),
            "concat merge does not support direct_map");

    const size_t dst_nlist = dst.nlist;
    const size_t src_nlist = src.nlist;
    const size_t new_nlist = dst_nlist + src_nlist;

    std::unique_ptr<Index> new_quantizer(clone_index(dst.quantizer));
    FAISS_THROW_IF_NOT(new_quantizer);
    FAISS_THROW_IF_NOT(new_quantizer->ntotal == dst_nlist);

    std::vector<float> centroid(dst.d);
    for (size_t i = 0; i < src_nlist; i++) {
        src.quantizer->reconstruct(i, centroid.data());
        new_quantizer->add(1, centroid.data());
    }
    FAISS_THROW_IF_NOT(new_quantizer->ntotal == new_nlist);

    std::unique_ptr<InvertedLists> new_lists(
            new ArrayInvertedLists(new_nlist, dst.code_size));

    for (size_t list_no = 0; list_no < dst_nlist; list_no++) {
        const size_t list_size = dst.invlists->list_size(list_no);
        if (list_size == 0) {
            continue;
        }
        InvertedLists::ScopedIds ids(dst.invlists, list_no);
        InvertedLists::ScopedCodes codes(dst.invlists, list_no);
        new_lists->add_entries(list_no, list_size, ids.get(), codes.get());
    }

    for (size_t list_no = 0; list_no < src_nlist; list_no++) {
        const size_t list_size = src.invlists->list_size(list_no);
        if (list_size == 0) {
            continue;
        }
        InvertedLists::ScopedIds ids(src.invlists, list_no);
        InvertedLists::ScopedCodes codes(src.invlists, list_no);
        const size_t new_list_no = dst_nlist + list_no;
        if (add_id == 0) {
            new_lists->add_entries(new_list_no, list_size, ids.get(), codes.get());
        } else {
            std::vector<idx_t> new_ids(list_size);
            for (size_t j = 0; j < list_size; j++) {
                new_ids[j] = ids[j] + add_id;
            }
            new_lists->add_entries(
                    new_list_no, list_size, new_ids.data(), codes.get());
        }
        src.invlists->resize(list_no, 0);
    }

    if (dst.own_fields && dst.quantizer && dst.quantizer != src.quantizer) {
        delete dst.quantizer;
    }
    dst.quantizer = new_quantizer.release();
    dst.own_fields = true;

    dst.nlist = new_nlist;
    dst.replace_invlists(new_lists.release(), true);
    dst.is_trained = true;
    dst.ntotal += src.ntotal;
    src.ntotal = 0;
}

std::unique_ptr<IndexIVFFlat> merge_ivfflat(
        const std::vector<IndexIVFFlat*>& indices,
        const MergeArguments& options) {
    if (ivfflat_merge_mt::use_mt_merge()) {
        return ivfflat_merge_mt::merge_ivfflat(indices, options);
    }
    FAISS_THROW_IF_NOT(!indices.empty());
    const auto t_total0 = std::chrono::steady_clock::now();

    IVFData data;
    const auto t_concat0 = std::chrono::steady_clock::now();
    concat_indices_into_ivfdata(indices, data);
    const auto t_concat1 = std::chrono::steady_clock::now();
    if (options.run_stats) {
        options.run_stats->concat_s =
                std::chrono::duration<double>(t_concat1 - t_concat0).count();
        options.run_stats->build_index_s = 0.0;
        options.run_stats->merge_total_s = 0.0;
    }

    if (options.method == MergeMethod::Merge) {
        faiss::MergeOptions merge_opts = options.merge;
        merge_full_on_ivfdata(data, merge_opts, options.run_stats);

        const auto t_build0 = std::chrono::steady_clock::now();
        auto out = sync_ivfdata_to_index(data);
        const auto t_build1 = std::chrono::steady_clock::now();
        if (options.run_stats) {
            options.run_stats->build_index_s =
                    std::chrono::duration<double>(t_build1 - t_build0).count();
        }
        finalize_merge_run_stats(options.run_stats);
        if (options.run_stats) {
            options.run_stats->total_s =
                    std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t_total0)
                            .count();
        }
        return out;
    }

    // Materialize owned ArrayInvertedLists (global ids in data.lists). The
    // VStack view only borrows shard invlists and is invalid after shards are
    // freed by the caller.
    const auto t_build0 = std::chrono::steady_clock::now();
    auto out = sync_ivfdata_to_index(data);
    const auto t_build1 = std::chrono::steady_clock::now();
    if (options.run_stats) {
        options.run_stats->build_index_s =
                std::chrono::duration<double>(t_build1 - t_build0).count();
    }
    finalize_merge_run_stats(options.run_stats);
    if (options.run_stats) {
        options.run_stats->total_s =
                std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t_total0)
                        .count();
    }
    return out;
}


std::unique_ptr<Index> shard_ivfflat(
        const std::vector<IndexIVFFlat*>& indices) {
    FAISS_THROW_IF_NOT(!indices.empty());

    const size_t d = indices[0]->d;
    FAISS_THROW_IF_NOT(d > 0);

    // Check environment variable to control threading
    // FAISS_SHARD_THREADED=0 or false to disable threading
    bool threaded = true;
    const char* env_threaded = std::getenv("FAISS_SHARD_THREADED");
    if (env_threaded != nullptr) {
        std::string val(env_threaded);
        if (val == "0" || val == "false" || val == "False" || val == "FALSE") {
            threaded = false;
        }
    }

    // Create IndexShards for concat search
    auto shards = std::make_unique<IndexShards>(
            static_cast<int>(d), threaded, true /* successive_ids */);

    for (IndexIVFFlat* shard : indices) {
        FAISS_THROW_IF_NOT(shard->d == static_cast<int>(d));
        Index* clone = clone_index(shard);
        auto* ivfflat_clone = dynamic_cast<IndexIVFFlat*>(clone);
        FAISS_THROW_IF_NOT_MSG(ivfflat_clone, "clone_index did not return IndexIVFFlat");
        shards->add_shard(ivfflat_clone);
    }

    return shards;
}

} // namespace faiss
