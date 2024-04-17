#pragma once
#include <SIR_SBM/common.hpp>
template <typename T>
std::vector<T> vector_merge(const std::vector<std::vector<T>>& vecs)
{
    std::vector<T> result;
    int N = std::accumulate(vecs.begin(), vecs.end(), 0L, [](size_t a, const std::vector<T>& b){return a + b.size();});
    result.reserve(N);
    for(const auto& vec: vecs)
    {
        result.insert(result.end(), vec.begin(), vec.end());
    }
    return result;
}