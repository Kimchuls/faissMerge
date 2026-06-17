// Multi-threaded snap IVFFlat merge (OpenMP).
#include <faiss/IndexIVFFlatMergeMT.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
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
    /// True once vectors[] holds materialized data (resize alone does not set this).
    bool vectors_dense = false;
    std::unique_ptr<faiss::IndexIVFFlat> index_owned;
    faiss::IndexIVFFlat* index_view = nullptr;
    std::vector<VectorLoc> id_loc;
};

static const float* ivfdata_vector_at(const IVFData& data, idx_t id) {
    if (data.vectors_dense) {
        FAISS_THROW_IF_NOT(static_cast<size_t>(id) < data.ntotal);
        return data.vectors.data() + static_cast<size_t>(id) * data.d;
    }
    FAISS_THROW_IF_NOT(data.index_view != nullptr);
    FAISS_THROW_IF_NOT(data.index_view->invlists != nullptr);
    FAISS_THROW_IF_NOT(static_cast<size_t>(id) < data.id_loc.size());
    const VectorLoc& loc = data.id_loc[static_cast<size_t>(id)];
    const uint8_t* codes = data.index_view->invlists->get_codes(loc.list_no);
    FAISS_THROW_IF_NOT(codes != nullptr);
    return reinterpret_cast<const float*>(
            codes + static_cast<size_t>(loc.off) * data.index_view->code_size);
}

static void ivfdata_copy_vector(const IVFData& data, idx_t id, float* dst) {
    const float* src = ivfdata_vector_at(data, id);
    std::memcpy(dst, src, data.d * sizeof(float));
}

#ifdef _OPENMP
struct ScopedOmpSingleThread {
    int prev = 1;
    ScopedOmpSingleThread() {
        prev = omp_get_max_threads();
        omp_set_num_threads(1);
    }
    ~ScopedOmpSingleThread() {
        omp_set_num_threads(prev);
    }
};
#else
struct ScopedOmpSingleThread {};
#endif

/// One parallel scatter-read pass; later stages read from dense cache.
static void ivfdata_ensure_dense_vectors(IVFData& data) {
    if (data.vectors_dense || data.ntotal == 0) {
        return;
    }
    FAISS_THROW_IF_NOT(data.index_view != nullptr);
    FAISS_THROW_IF_NOT(data.index_view->invlists != nullptr);
    data.vectors.resize(data.ntotal * data.d);
    const size_t code_size = data.d * sizeof(float);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (long long cid_ll = 0; cid_ll < static_cast<long long>(data.nlist); cid_ll++) {
        const size_t cid = static_cast<size_t>(cid_ll);
        const auto& lst = data.lists[cid];
        faiss::InvertedLists* invlists = data.index_view->invlists;
        for (idx_t id : lst) {
            const VectorLoc& loc = data.id_loc[static_cast<size_t>(id)];
            const uint8_t* codes = invlists->get_codes(loc.list_no);
            FAISS_THROW_IF_NOT(codes != nullptr);
            std::memcpy(
                    data.vectors.data() + static_cast<size_t>(id) * data.d,
                    codes + static_cast<size_t>(loc.off) * code_size,
                    code_size);
        }
    }
    data.vectors_dense = true;
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

static void ivfdata_refresh_metadata(IVFData& data) {
    FAISS_THROW_IF_NOT(data.index_view != nullptr);
    faiss::IndexIVFFlat* idx = data.index_view;
    data.d = static_cast<size_t>(idx->d);
    data.nlist = static_cast<size_t>(idx->nlist);
    data.ntotal = static_cast<size_t>(idx->ntotal);
    data.metric = idx->metric_type;
    data.centroids.resize(data.nlist * data.d);
    data.lists.assign(data.nlist, {});
    data.id_loc.assign(data.ntotal, {});

    std::vector<float> centroid(data.d);
    for (size_t list_no = 0; list_no < data.nlist; list_no++) {
        idx->quantizer->reconstruct(
                static_cast<faiss::idx_t>(list_no), centroid.data());
        std::copy(
                centroid.begin(),
                centroid.end(),
                data.centroids.begin() + list_no * data.d);

        const size_t list_size = idx->invlists->list_size(list_no);
        data.lists[list_no].reserve(list_size);
        if (list_size == 0) {
            continue;
        }
        faiss::InvertedLists::ScopedIds ids(idx->invlists, list_no);
        for (size_t i = 0; i < list_size; i++) {
            const idx_t id = ids[i];
            FAISS_THROW_IF_NOT(id >= 0 && static_cast<size_t>(id) < data.ntotal);
            data.lists[list_no].push_back(id);
            data.id_loc[static_cast<size_t>(id)] =
                    VectorLoc{static_cast<uint32_t>(list_no), static_cast<uint32_t>(i)};
        }
    }
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

    auto quantizer = std::make_unique<faiss::IndexFlatL2>(static_cast<int>(d));
    for (const auto* idx : indices) {
        std::vector<float> centroid(d);
        for (size_t list_no = 0; list_no < static_cast<size_t>(idx->nlist); list_no++) {
            idx->quantizer->reconstruct(
                    static_cast<faiss::idx_t>(list_no), centroid.data());
            quantizer->add(1, centroid.data());
        }
    }
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

    out.d = d;
    out.nlist = total_nlist;
    out.ntotal = total_ntotal;
    out.metric = indices[0]->metric_type;
    out.centroids.resize(total_nlist * d);
    out.lists.assign(total_nlist, {});
    out.id_loc.assign(total_ntotal, {});

    std::vector<float> centroid(d);
    size_t global_list_no = 0;
    idx_t id_offset = 0;
    for (const auto* idx : indices) {
        for (size_t local_list = 0; local_list < static_cast<size_t>(idx->nlist);
             local_list++, global_list_no++) {
            idx->quantizer->reconstruct(
                    static_cast<faiss::idx_t>(local_list), centroid.data());
            std::copy(
                    centroid.begin(),
                    centroid.end(),
                    out.centroids.begin() + global_list_no * d);

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
    out.vectors_dense = true;
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
    std::vector<std::vector<idx_t>> new_lists(nlist);
    for (size_t id = 0; id < assign.size(); id++) {
        const idx_t cid = assign[id];
        FAISS_THROW_IF_NOT(cid >= 0 && static_cast<size_t>(cid) < nlist);
        new_lists[static_cast<size_t>(cid)].push_back(static_cast<idx_t>(id));
    }
    lists.swap(new_lists);
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

    const int k = neighbor_k + 1;
    std::vector<int64_t> labels(static_cast<size_t>(nlist) * k);
    std::vector<float> distances(static_cast<size_t>(nlist) * k);
    faiss::knn_L2sqr(
            centroids.data(),
            centroids.data(),
            d,
            nlist,
            nlist,
            static_cast<size_t>(k),
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
            const int64_t id = labels[i * k + j];
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

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (int gid = 0; gid < newC; gid++) {
        double wsum = 0.0;
        float* outc = new_centroids.data() + static_cast<size_t>(gid) * data.d;

        size_t merged_size = 0;
        for (int old : groups[static_cast<size_t>(gid)]) {
            merged_size += data.lists[static_cast<size_t>(old)].size();
        }
        std::vector<idx_t> merged;
        merged.reserve(merged_size);

        for (int old : groups[static_cast<size_t>(gid)]) {
            const size_t sz = data.lists[static_cast<size_t>(old)].size();
            const double w = (sz > 0) ? static_cast<double>(sz) : 1.0;
            const float* c = data.centroids.data() + static_cast<size_t>(old) * data.d;
            for (size_t j = 0; j < data.d; j++) {
                outc[j] += static_cast<float>(w) * c[j];
            }
            wsum += w;

            const auto& lst = data.lists[static_cast<size_t>(old)];
            if (!lst.empty()) {
                merged.insert(merged.end(), lst.begin(), lst.end());
            }
        }

        if (wsum > 0.0) {
            const float inv = 1.0f / static_cast<float>(wsum);
            for (size_t j = 0; j < data.d; j++) {
                outc[j] *= inv;
            }
        }
        new_lists[static_cast<size_t>(gid)] = std::move(merged);
    }

    data.centroids.swap(new_centroids);
    data.lists.swap(new_lists);
    data.nlist = data.lists.size();

    int merged = static_cast<int>(C - data.nlist);
    return {merged > 0, merged};
}

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
        size_t user_batch_size) {
    if (data.nlist >= target_nlist) {
        return {false, 0};
    }
    if (data.nlist == 0 || data.ntotal == 0) {
        return {false, 0};
    }
    FAISS_THROW_IF_NOT_MSG(data.metric == faiss::METRIC_L2, "quota split supports METRIC_L2 only");

    const size_t C = data.nlist;
    const size_t T = target_nlist;
    const size_t N = data.ntotal;

    std::vector<size_t> sizes(C, 0);
    for (size_t i = 0; i < C; i++) {
        sizes[i] = data.lists[i].size();
    }

    std::vector<int> k_per(C, 1);

    const int max_k_per_cluster = 64;
    for (size_t i = 0; i < C; i++) {
        const size_t sz = sizes[i];
        if (sz < 2) {
            k_per[i] = 1;
            continue;
        }
        const double ideal = (static_cast<double>(sz) * static_cast<double>(T)) / static_cast<double>(N);
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
            return sizes[a] > sizes[b];
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
            return sizes[a] < sizes[b];
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
    // if (cur != T) {
    //     // Second pass: same constraints, fill-in-order (largest-first add / smallest-first remove).
    //     // Only runs when first pass did not reach T; does not change sum_k() but may change distribution.
    //     if (cur < T) {
    //         size_t need = T - cur;
    //         std::vector<size_t> order(C);
    //         std::iota(order.begin(), order.end(), 0);
    //         std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    //             return sizes[a] > sizes[b];
    //         });
    //         for (size_t idx = 0; idx < C && need > 0; idx++) {
    //             const size_t i = order[idx];
    //             const int cap = std::min<int>(max_k_per_cluster, static_cast<int>(sizes[i]));
    //             while (need > 0 && k_per[i] < cap) {
    //                 k_per[i]++;
    //                 need--;
    //             }
    //         }
    //     } else {
    //         size_t extra = cur - T;
    //         std::vector<size_t> order(C);
    //         std::iota(order.begin(), order.end(), 0);
    //         std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    //             return sizes[a] < sizes[b];
    //         });
    //         for (size_t idx = 0; idx < C && extra > 0; idx++) {
    //             const size_t i = order[idx];
    //             while (extra > 0 && k_per[i] > 1) {
    //                 k_per[i]--;
    //                 extra--;
    //             }
    //         }
    //     }
    //     cur = sum_k();
    // }

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
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long long i = 0; i < static_cast<long long>(sample_ids.size()); i++) {
            ivfdata_copy_vector(
                    data,
                    sample_ids[static_cast<size_t>(i)],
                    X.data() + static_cast<size_t>(i) * data.d);
        }

        faiss::Clustering clus(static_cast<int>(data.d), k);
        clus.niter = std::max(1, niter);
        clus.nredo = std::max(1, nredo);
        clus.seed = seed + static_cast<int>(cid) * 17;

        {
            ScopedOmpSingleThread guard;
            faiss::IndexFlatL2 assigner(static_cast<int>(data.d));
            clus.train(
                    static_cast<faiss::idx_t>(sample_ids.size()), X.data(), assigner);
        }

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

        std::vector<std::vector<idx_t>> sublists(static_cast<size_t>(k));
        for (auto& sl : sublists) {
            sl.reserve(sz / static_cast<size_t>(k) + 8);
        }

        std::vector<float> sums(static_cast<size_t>(k) * data.d, 0.0f);
        std::vector<size_t> counts(static_cast<size_t>(k), 0);

        std::vector<float> Xb(assign_batch * data.d);
        std::vector<int64_t> labels(assign_batch);
        std::vector<float> dists(assign_batch);
        const bool use_dense = data.vectors_dense;
        const float* dense = use_dense ? data.vectors.data() : nullptr;

        for (size_t s = 0; s < sz; s += assign_batch) {
            const size_t e = std::min(sz, s + assign_batch);
            const size_t bc = e - s;

            for (size_t bi = 0; bi < bc; bi++) {
                const idx_t id = ids[s + bi];
                if (use_dense) {
                    std::memcpy(
                            Xb.data() + bi * data.d,
                            dense + static_cast<size_t>(id) * data.d,
                            data.d * sizeof(float));
                } else {
                    ivfdata_copy_vector(data, id, Xb.data() + bi * data.d);
                }
            }

            {
                ScopedOmpSingleThread guard;
                faiss::knn_L2sqr(
                        Xb.data(),
                        centers.data(),
                        data.d,
                        bc,
                        static_cast<size_t>(k),
                        1,
                        dists.data(),
                        labels.data());
            }

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
        size_t batch_size) {
    if (data.nlist >= target) {
        return;
    }
    int quota_pass = 0;
    while (data.nlist < target && quota_pass < 256) {
        const size_t n_before = data.nlist;
        quota_split_clusters_kmeans(
                data,
                target,
                split_kmeans_niter,
                split_kmeans_nredo,
                random_state + quota_pass * 7919,
                batch_size);
        quota_pass++;
        if (data.nlist == n_before) {
            break;
        }
    }
    int split_pass = 0;
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
}

static std::pair<bool, int> reduce_centroids_to_target_kmeans(
        IVFData& data,
        size_t target_nlist,
        int niter,
        int nredo,
        int seed,
        size_t user_batch_size) {
    if (data.nlist <= target_nlist) {
        return {false, 0};
    }
    if (target_nlist == 0) {
        return {false, 0};
    }
    FAISS_THROW_IF_NOT_MSG(data.metric == faiss::METRIC_L2, "reduce supports METRIC_L2 only");

    const size_t C = data.nlist;
    const size_t T = target_nlist;

    faiss::Clustering clus(static_cast<int>(data.d), static_cast<int>(T));
    clus.niter = std::max(1, niter);
    clus.nredo = std::max(1, nredo);
    clus.seed = seed;

    faiss::IndexFlatL2 assigner(static_cast<int>(data.d));
    clus.train(static_cast<faiss::idx_t>(C), data.centroids.data(), assigner);

    FAISS_THROW_IF_NOT(clus.centroids.size() == T * data.d);

    const size_t batch = std::max<size_t>(256, std::min<size_t>(user_batch_size, 4096));
    std::vector<float> Xb(batch * data.d);
    std::vector<int64_t> labels(batch);
    std::vector<float> dists(batch);

    std::vector<int> map_old_to_new(C, 0);

    for (size_t s = 0; s < C; s += batch) {
        const size_t e = std::min(C, s + batch);
        const size_t bc = e - s;
        std::memcpy(
                Xb.data(),
                data.centroids.data() + s * data.d,
                bc * data.d * sizeof(float));
        faiss::knn_L2sqr(
                Xb.data(),
                clus.centroids.data(),
                data.d,
                bc,
                T,
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

    std::vector<size_t> list_counts(T, 0);
    for (size_t old = 0; old < C; old++) {
        list_counts[static_cast<size_t>(map_old_to_new[old])] +=
                data.lists[old].size();
    }
    std::vector<std::vector<idx_t>> new_lists(T);
    for (size_t t = 0; t < T; t++) {
        new_lists[t].resize(list_counts[t]);
    }
    std::vector<size_t> per_tgt_cursor(T, 0);
    std::vector<size_t> per_old_offset(C, 0);
    for (size_t old = 0; old < C; old++) {
        const size_t t = static_cast<size_t>(map_old_to_new[old]);
        per_old_offset[old] = per_tgt_cursor[t];
        per_tgt_cursor[t] += data.lists[old].size();
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (long long old_ll = 0; old_ll < static_cast<long long>(C); old_ll++) {
        const size_t old = static_cast<size_t>(old_ll);
        const int t = map_old_to_new[old];
        const auto& src = data.lists[old];
        if (src.empty()) {
            continue;
        }
        std::copy(
                src.begin(),
                src.end(),
                new_lists[static_cast<size_t>(t)].data() + per_old_offset[old]);
    }

    std::vector<float> new_centroids = clus.centroids;

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

    data.centroids.swap(new_centroids);
    data.lists.swap(new_lists);
    data.nlist = data.lists.size();

    int reduced = static_cast<int>(C - T);
    return {reduced > 0, reduced};
}

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

    auto t0 = std::chrono::steady_clock::now();

    {
        auto t_merge0 = std::chrono::steady_clock::now();
        dedup_close_centroids_mutual_knn(data, options.merge_threshold, options.neighbor_k);
        auto t_merge1 = std::chrono::steady_clock::now();
        if (stats) {
            stats->merge_close_s = std::chrono::duration<double>(t_merge1 - t_merge0).count();
        }
    }

    {
        auto t_split0 = std::chrono::steady_clock::now();

        if (data.nlist > options.target_nlist) {
            reduce_centroids_to_target_kmeans(
                    data,
                    options.target_nlist,
                    std::max(1, options.split_kmeans_niter),
                    std::max(1, options.split_kmeans_nredo),
                    options.random_state,
                    static_cast<size_t>(options.batch_size));
        } else if (data.nlist < options.target_nlist) {
            ensure_ivfdata_reaches_target_nlist(
                    data,
                    options.target_nlist,
                    std::max(1, options.split_kmeans_niter),
                    std::max(1, options.split_kmeans_nredo),
                    options.random_state,
                    static_cast<size_t>(options.batch_size));
        }

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
    const size_t want = std::max<size_t>(1, static_cast<size_t>(data.ntotal * fraction));
    std::vector<idx_t> all(data.ntotal);
    std::iota(all.begin(), all.end(), static_cast<idx_t>(0));
    std::mt19937 rng(static_cast<uint32_t>(random_state));
    std::shuffle(all.begin(), all.end(), rng);
    if (want >= all.size()) {
        return all;
    }
    all.resize(want);
    return all;
}

/// Parallel Lloyd: assignment via knn_L2sqr, centroid update by point partitions.
static void train_kmeans_centroids_with_init_parallel(
        const std::vector<float>& train_x,
        size_t n_train,
        size_t d,
        size_t nlist,
        const std::vector<float>& init_centroids,
        int niter,
        std::vector<float>& centroids_out) {
    FAISS_THROW_IF_NOT(n_train > 0);
    FAISS_THROW_IF_NOT(init_centroids.size() == nlist * d);
    centroids_out = init_centroids;
    const int iters = std::max(1, niter);

    std::vector<float> dists(n_train);
    std::vector<int64_t> assign(n_train);

#ifdef _OPENMP
    const int nt = omp_get_max_threads();
    std::vector<std::vector<float>> partial_cent(
            static_cast<size_t>(nt),
            std::vector<float>(nlist * d, 0.0f));
    std::vector<std::vector<float>> partial_counts(
            static_cast<size_t>(nt), std::vector<float>(nlist, 0.0f));
#else
    const int nt = 1;
#endif

    std::vector<float> accum(nlist * d);
    std::vector<float> counts(nlist);

    for (int it = 0; it < iters; it++) {
        faiss::knn_L2sqr(
                train_x.data(),
                centroids_out.data(),
                d,
                n_train,
                nlist,
                1,
                dists.data(),
                assign.data());

        std::memset(accum.data(), 0, accum.size() * sizeof(float));
        std::memset(counts.data(), 0, counts.size() * sizeof(float));

#ifdef _OPENMP
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto& local_cent = partial_cent[static_cast<size_t>(tid)];
            auto& local_counts = partial_counts[static_cast<size_t>(tid)];
            std::memset(
                    local_cent.data(),
                    0,
                    local_cent.size() * sizeof(float));
            std::memset(
                    local_counts.data(),
                    0,
                    local_counts.size() * sizeof(float));

#pragma omp for schedule(static)
            for (long long i = 0; i < static_cast<long long>(n_train); i++) {
                const int64_t ci = assign[static_cast<size_t>(i)];
                FAISS_THROW_IF_NOT(
                        ci >= 0 && static_cast<size_t>(ci) < nlist);
                const float* xi = train_x.data() + static_cast<size_t>(i) * d;
                float* c = local_cent.data() + static_cast<size_t>(ci) * d;
                for (size_t j = 0; j < d; j++) {
                    c[j] += xi[j];
                }
                local_counts[static_cast<size_t>(ci)] += 1.0f;
            }
        }

        for (int t = 0; t < nt; t++) {
            const auto& local_cent = partial_cent[static_cast<size_t>(t)];
            const auto& local_counts = partial_counts[static_cast<size_t>(t)];
            for (size_t c = 0; c < nlist; c++) {
                counts[c] += local_counts[c];
                const float* src = local_cent.data() + c * d;
                float* dst = accum.data() + c * d;
                for (size_t j = 0; j < d; j++) {
                    dst[j] += src[j];
                }
            }
        }
#else
        for (size_t i = 0; i < n_train; i++) {
            const int64_t ci = assign[i];
            FAISS_THROW_IF_NOT(ci >= 0 && static_cast<size_t>(ci) < nlist);
            const float* xi = train_x.data() + i * d;
            float* c = accum.data() + static_cast<size_t>(ci) * d;
            for (size_t j = 0; j < d; j++) {
                c[j] += xi[j];
            }
            counts[static_cast<size_t>(ci)] += 1.0f;
        }
#endif

        for (size_t c = 0; c < nlist; c++) {
            if (counts[c] <= 0.0f) {
                continue;
            }
            const float inv = 1.0f / counts[c];
            const float* src = accum.data() + c * d;
            float* dst = centroids_out.data() + c * d;
            for (size_t j = 0; j < d; j++) {
                dst[j] = src[j] * inv;
            }
        }
    }
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

#ifdef _OPENMP
    if (omp_get_max_threads() >
        faiss::ivfflat_merge_mt::high_parallel_stage2_min_threads()) {
        train_kmeans_centroids_with_init_parallel(
                train_x,
                n_train,
                d,
                nlist,
                init_centroids,
                niter,
                centroids_out);
        return;
    }
#endif

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
    const int k = std::min(neighbor_k, static_cast<int>(n_tgt));
    std::vector<int64_t> labels(static_cast<size_t>(n_src) * k);
    std::vector<float> dists(static_cast<size_t>(n_src) * k);
    faiss::knn_L2sqr(
            src_centroids.data(),
            tgt_centroids.data(),
            d,
            n_src,
            n_tgt,
            static_cast<size_t>(k),
            dists.data(),
            labels.data());
    std::vector<std::vector<int>> out(n_src);
    for (size_t i = 0; i < n_src; i++) {
        out[i].reserve(static_cast<size_t>(k));
        for (int j = 0; j < k; j++) {
            const int64_t id = labels[i * k + j];
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

static void reassign_one_cluster(
        IVFData& data,
        size_t src_cid,
        const std::vector<std::vector<int>>& src_to_tgt,
        const std::vector<float>& tgt_centroids,
        size_t n_tgt,
        size_t bs,
        std::vector<idx_t>& assign,
        bool use_dense) {
    const size_t d = data.d;
    const float* dense = use_dense ? data.vectors.data() : nullptr;
    const auto& lst = data.lists[src_cid];
    if (lst.empty()) {
        return;
    }
    const auto& tgt_ids = src_to_tgt[src_cid];
    const size_t ncand = tgt_ids.size();
    std::vector<float> cand_centroids(ncand * d);
    for (size_t i = 0; i < ncand; i++) {
        const int tid = tgt_ids[i];
        FAISS_THROW_IF_NOT(tid >= 0 && static_cast<size_t>(tid) < n_tgt);
        std::copy(
                tgt_centroids.begin() + static_cast<size_t>(tid) * d,
                tgt_centroids.begin() + (static_cast<size_t>(tid) + 1) * d,
                cand_centroids.begin() + i * d);
    }

    faiss::IndexFlatL2 cand_index(static_cast<int>(d));
    cand_index.add(static_cast<faiss::idx_t>(ncand), cand_centroids.data());

    // IndexFlat::search parallelizes when n > 10000; cap batches inside omp regions.
    size_t inner_bs = bs;
#ifdef _OPENMP
    if (omp_in_parallel()) {
        inner_bs = std::min(bs, static_cast<size_t>(8192));
    }
#endif

    std::vector<float> Xc(inner_bs * d);
    std::vector<float> dists(inner_bs);
    std::vector<faiss::idx_t> labels(inner_bs);
    for (size_t s = 0; s < lst.size(); s += inner_bs) {
        const size_t e = std::min(lst.size(), s + inner_bs);
        const size_t bc = e - s;
        for (size_t bi = 0; bi < bc; bi++) {
            const idx_t id = lst[s + bi];
            if (use_dense) {
                std::memcpy(
                        Xc.data() + bi * d,
                        dense + static_cast<size_t>(id) * d,
                        d * sizeof(float));
            } else {
                ivfdata_copy_vector(data, id, Xc.data() + bi * d);
            }
        }
        cand_index.search(
                static_cast<faiss::idx_t>(bc),
                Xc.data(),
                1,
                dists.data(),
                labels.data());
        for (size_t bi = 0; bi < bc; bi++) {
            const idx_t id = lst[s + bi];
            const int best_tgt = tgt_ids[static_cast<size_t>(labels[bi])];
            assign[static_cast<size_t>(id)] = static_cast<idx_t>(best_tgt);
        }
    }
}

/// Serial path (thread=1): IndexFlat per cluster, same as single-thread merge.
static void reassign_all_serial(
        IVFData& data,
        const std::vector<std::vector<int>>& src_to_tgt,
        const std::vector<float>& tgt_centroids,
        size_t n_tgt,
        size_t bs,
        std::vector<idx_t>& assign) {
    for (size_t src_cid = 0; src_cid < data.nlist; src_cid++) {
        const auto& lst = data.lists[src_cid];
        if (lst.empty()) {
            continue;
        }
        const auto& tgt_ids = src_to_tgt[src_cid];
        std::vector<float> cand_centroids(tgt_ids.size() * data.d);
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
                static_cast<faiss::idx_t>(tgt_ids.size()), cand_centroids.data());

        for (size_t s = 0; s < lst.size(); s += bs) {
            const size_t e = std::min(lst.size(), s + bs);
            const size_t bc = e - s;
            std::vector<float> Xc(bc * data.d);
            for (size_t bi = 0; bi < bc; bi++) {
                ivfdata_copy_vector(
                        data, lst[s + bi], Xc.data() + bi * data.d);
            }
            std::vector<faiss::idx_t> labels(bc);
            std::vector<float> dists(bc);
            cand_index.search(
                    static_cast<faiss::idx_t>(bc),
                    Xc.data(),
                    1,
                    dists.data(),
                    labels.data());
            for (size_t bi = 0; bi < bc; bi++) {
                const idx_t id = lst[s + bi];
                const int best_tgt = tgt_ids[static_cast<size_t>(labels[bi])];
                assign[static_cast<size_t>(id)] = static_cast<idx_t>(best_tgt);
            }
        }
    }
}

/// Low thread count: static cluster partition, dense vector cache, knn on local cands.
static void reassign_all_low_thread(
        IVFData& data,
        const std::vector<std::vector<int>>& src_to_tgt,
        const std::vector<float>& tgt_centroids,
        size_t n_tgt,
        size_t bs,
        std::vector<idx_t>& assign) {
    FAISS_THROW_IF_NOT_MSG(
            data.vectors_dense,
            "reassign: dense vector cache required (call ivfdata_ensure_dense_vectors in merge_full)");
    const bool use_dense = true;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long src_cid_ll = 0; src_cid_ll < static_cast<long long>(data.nlist);
         src_cid_ll++) {
        reassign_one_cluster(
                data,
                static_cast<size_t>(src_cid_ll),
                src_to_tgt,
                tgt_centroids,
                n_tgt,
                bs,
                assign,
                use_dense);
    }
}

/// High thread count: dynamic cluster scheduling for load balance.
static void reassign_all_high_thread(
        IVFData& data,
        const std::vector<std::vector<int>>& src_to_tgt,
        const std::vector<float>& tgt_centroids,
        size_t n_tgt,
        size_t bs,
        std::vector<idx_t>& assign) {
    FAISS_THROW_IF_NOT_MSG(
            data.vectors_dense,
            "reassign: dense vector cache required (call ivfdata_ensure_dense_vectors in merge_full)");
    const bool use_dense = true;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (long long src_cid_ll = 0; src_cid_ll < static_cast<long long>(data.nlist);
         src_cid_ll++) {
        reassign_one_cluster(
                data,
                static_cast<size_t>(src_cid_ll),
                src_to_tgt,
                tgt_centroids,
                n_tgt,
                bs,
                assign,
                use_dense);
    }
}

static void reassign_all_via_src_to_tgt_neighbors(
        IVFData& data,
        const std::vector<std::vector<int>>& src_to_tgt,
        const std::vector<float>& tgt_centroids,
        size_t n_tgt,
        int batch_size) {
    FAISS_THROW_IF_NOT(src_to_tgt.size() == data.nlist);
    FAISS_THROW_IF_NOT(tgt_centroids.size() == n_tgt * data.d);
    std::vector<idx_t> assign = build_assign_from_lists(data.lists, data.ntotal);
    const size_t bs = static_cast<size_t>(std::max(1, batch_size));

#ifdef _OPENMP
    const int nt = omp_get_max_threads();
    if (nt <= 1) {
        reassign_all_serial(
                data, src_to_tgt, tgt_centroids, n_tgt, bs, assign);
    } else if (nt <= faiss::ivfflat_merge_mt::reassign_low_thread_max()) {
        // Few threads: parallelize over clusters (static partition).
        reassign_all_low_thread(
                data, src_to_tgt, tgt_centroids, n_tgt, bs, assign);
    } else {
        // Many threads: dynamic cluster scheduling for load balance.
        reassign_all_high_thread(
                data, src_to_tgt, tgt_centroids, n_tgt, bs, assign);
    }
#else
    reassign_all_serial(
            data, src_to_tgt, tgt_centroids, n_tgt, bs, assign);
#endif

    data.nlist = n_tgt;
    data.centroids = tgt_centroids;
    data.lists.assign(n_tgt, {});
    rebuild_lists_from_assign(data.lists, assign);
}

static void snap_centroids_to_nearest_sample_points(
        const std::vector<float>& warm_centroids,
        size_t nlist,
        size_t d,
        const std::vector<float>& sample_x,
        size_t n_sample,
        std::vector<float>& snapped_out) {
    FAISS_THROW_IF_NOT(n_sample > 0);
    FAISS_THROW_IF_NOT(warm_centroids.size() == nlist * d);
    snapped_out.resize(nlist * d);
    std::vector<int64_t> labels(nlist);

#ifdef _OPENMP
    const bool snap_parallel =
            omp_get_max_threads() >
            faiss::ivfflat_merge_mt::high_parallel_stage2_min_threads();
    if (snap_parallel) {
#pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(nlist); i++) {
            const float* q = warm_centroids.data() + static_cast<size_t>(i) * d;
            float best_dis = std::numeric_limits<float>::max();
            size_t best_j = 0;
            for (size_t j = 0; j < n_sample; j++) {
                const float dis = faiss::fvec_L2sqr(
                        q, sample_x.data() + j * d, d);
                if (dis < best_dis) {
                    best_dis = dis;
                    best_j = j;
                }
            }
            labels[static_cast<size_t>(i)] = static_cast<int64_t>(best_j);
        }

#pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(nlist); i++) {
            const int64_t sid = labels[static_cast<size_t>(i)];
            FAISS_THROW_IF_NOT(
                    sid >= 0 && static_cast<size_t>(sid) < n_sample);
            std::copy(
                    sample_x.begin() + static_cast<size_t>(sid) * d,
                    sample_x.begin() + (static_cast<size_t>(sid) + 1) * d,
                    snapped_out.begin() + static_cast<size_t>(i) * d);
        }
        return;
    }
#endif

    std::vector<float> dists(nlist);
    faiss::knn_L2sqr(
            warm_centroids.data(),
            sample_x.data(),
            d,
            nlist,
            n_sample,
            1,
            dists.data(),
            labels.data());
    for (size_t i = 0; i < nlist; i++) {
        const int64_t sid = labels[i];
        FAISS_THROW_IF_NOT(sid >= 0 && static_cast<size_t>(sid) < n_sample);
        std::copy(
                sample_x.begin() + static_cast<size_t>(sid) * d,
                sample_x.begin() + (static_cast<size_t>(sid) + 1) * d,
                snapped_out.begin() + i * d);
    }
}

static std::vector<float> gather_sample_vectors(
        const IVFData& data,
        float sample_fraction,
        int random_state) {
    auto sample_ids_vec = sample_ids_random_fraction(data, sample_fraction, random_state);
    std::vector<float> train_x(sample_ids_vec.size() * data.d);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long i = 0; i < static_cast<long long>(sample_ids_vec.size()); i++) {
        ivfdata_copy_vector(
                data,
                sample_ids_vec[static_cast<size_t>(i)],
                train_x.data() + static_cast<size_t>(i) * data.d);
    }
    return train_x;
}

static void merge_stage2_kmeans_remap_with_init(
        IVFData& data,
        const faiss::MergeOptions& options,
        faiss::MergeRunStats* stats,
        const std::vector<float>& sample_x,
        size_t n_sample,
        const std::vector<float>& init_centroids) {
    const size_t target = options.target_nlist;
    FAISS_THROW_IF_NOT(target > 0);
    FAISS_THROW_IF_NOT(init_centroids.size() == target * data.d);

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

    const auto t_map0 = std::chrono::steady_clock::now();
    auto src_to_tgt = build_src_to_tgt_neighbor_map(
            data.centroids,
            data.nlist,
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
            data, src_to_tgt, tgt_centroids, target, options.batch_size);
    const auto t_re1 = std::chrono::steady_clock::now();
    if (stats) {
        stats->remap_full_reassign_s =
                std::chrono::duration<double>(t_re1 - t_re0).count();
    }
}

static void merge_stage2_snap_to_data_kmeans_remap(
        IVFData& data,
        const faiss::MergeOptions& options,
        faiss::MergeRunStats* stats) {
    const size_t target = options.target_nlist;
    FAISS_THROW_IF_NOT(target > 0);
    FAISS_THROW_IF_NOT_MSG(
            data.nlist == target,
            "merge_stage2_snap_to_data_kmeans_remap: stage1 nlist != target_nlist");

    const std::vector<float> warm_centroids = data.centroids;
    auto sample_x = gather_sample_vectors(
            data, options.sample_fraction, options.random_state);
    const size_t n_sample = sample_x.size() / data.d;

    const auto t_snap0 = std::chrono::steady_clock::now();
    std::vector<float> snapped_centroids;
    snap_centroids_to_nearest_sample_points(
            warm_centroids, target, data.d, sample_x, n_sample, snapped_centroids);
    const auto t_snap1 = std::chrono::steady_clock::now();
    if (stats) {
        stats->remap_snap_to_data_s =
                std::chrono::duration<double>(t_snap1 - t_snap0).count();
    }

    merge_stage2_kmeans_remap_with_init(
            data, options, stats, sample_x, n_sample, snapped_centroids);
}

static void merge_full_on_ivfdata(
        IVFData& data,
        faiss::MergeOptions options,
        faiss::MergeRunStats* stats) {
    if (options.target_nlist == 0) {
        options.target_nlist = data.nlist;
    }
    FAISS_THROW_IF_NOT(options.target_nlist > 0);

    {
        std::vector<idx_t> assign = build_assign_from_lists(data.lists, data.ntotal);
        compress_empty_clusters(data, assign);
        rebuild_lists_from_assign(data.lists, assign);
    }

    ivfdata_ensure_dense_vectors(data);
    merge_stage1_adjust_nlist(data, options, stats);
    merge_stage2_snap_to_data_kmeans_remap(data, options, stats);
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
#pragma omp parallel for schedule(static)
#endif
    for (long long cid_ll = 0; cid_ll < static_cast<long long>(data.nlist); cid_ll++) {
        const size_t cid = static_cast<size_t>(cid_ll);
        invlists.resize(cid, data.lists[cid].size());
    }

    const bool use_dense = data.vectors_dense;
    const float* dense = use_dense ? data.vectors.data() : nullptr;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (long long cid_ll = 0; cid_ll < static_cast<long long>(data.nlist); cid_ll++) {
        const size_t cid = static_cast<size_t>(cid_ll);
        const auto& lst = data.lists[cid];
        const size_t n = lst.size();
        if (n == 0) {
            continue;
        }

        uint8_t* code_ptr = invlists.codes[cid].data();
        if (use_dense) {
            for (size_t i = 0; i < n; i++) {
                std::memcpy(
                        code_ptr + i * code_size,
                        dense + static_cast<size_t>(lst[i]) * data.d,
                        code_size);
            }
        } else {
            FAISS_THROW_IF_NOT(data.index_view != nullptr);
            faiss::InvertedLists* src_invlists = data.index_view->invlists;
            FAISS_THROW_IF_NOT(src_invlists != nullptr);
            for (size_t i = 0; i < n; i++) {
                const VectorLoc& loc = data.id_loc[static_cast<size_t>(lst[i])];
                const uint8_t* src_codes = src_invlists->get_codes(loc.list_no);
                FAISS_THROW_IF_NOT(src_codes != nullptr);
                std::memcpy(
                        code_ptr + i * code_size,
                        src_codes + static_cast<size_t>(loc.off) * code_size,
                        code_size);
            }
        }
        std::memcpy(
                invlists.ids[cid].data(),
                lst.data(),
                n * sizeof(idx_t));
    }
}

static std::unique_ptr<faiss::IndexIVFFlat> sync_ivfdata_to_index(IVFData& data) {
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

    data.vectors.clear();
    data.vectors.shrink_to_fit();
    data.vectors_dense = false;

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
namespace ivfflat_merge_mt {

static void finalize_merge_run_stats_mt(MergeRunStats* stats) {
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
    ivfdata_from_dense_vectors(internal_data);

    merge_full_on_ivfdata(internal_data, options, stats);
    finalize_merge_run_stats_mt(stats);

    data.d = internal_data.d;
    data.nlist = internal_data.nlist;
    data.ntotal = internal_data.ntotal;
    data.metric = internal_data.metric;
    data.centroids = std::move(internal_data.centroids);
    data.lists = std::move(internal_data.lists);
    data.vectors = std::move(internal_data.vectors);

    if (stats) {
        const double dt = std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() - t0)
                                  .count();
        stats->ivf_merge_s += dt;
    }
}

std::unique_ptr<IndexIVFFlat> merge_ivfflat(
        const std::vector<IndexIVFFlat*>& indices,
        const MergeArguments& options) {
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
        finalize_merge_run_stats_mt(options.run_stats);
        if (options.run_stats) {
            options.run_stats->total_s =
                    std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t_total0)
                            .count();
        }
        return out;
    }

    const auto t_build0 = std::chrono::steady_clock::now();
    auto out = sync_ivfdata_to_index(data);
    const auto t_build1 = std::chrono::steady_clock::now();
    if (options.run_stats) {
        options.run_stats->build_index_s =
                std::chrono::duration<double>(t_build1 - t_build0).count();
    }
    finalize_merge_run_stats_mt(options.run_stats);
    if (options.run_stats) {
        options.run_stats->total_s =
                std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t_total0)
                        .count();
    }
    return out;
}

} // namespace ivfflat_merge_mt
} // namespace faiss