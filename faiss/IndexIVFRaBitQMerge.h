#pragma once

#include <memory>
#include <vector>

#include <faiss/Index.h>
#include <faiss/IndexIVFFlatMerge.h>
#include <faiss/IndexIVFRaBitQ.h>

namespace faiss {

std::unique_ptr<Index> merge_ivfrabitq(
        const std::vector<IndexIVFRaBitQ*>& indices,
        const MergeArguments& options);

} // namespace faiss
