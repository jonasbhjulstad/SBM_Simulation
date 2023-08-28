#ifndef VECTOR_UTILS_HPP
#define VECTOR_UTILS_HPP
#include <Sycl_Graph/SIR_Types.hpp>

template <typename T>
std::vector<std::vector<T>> vector_split(const std::vector<T>& vec, uint32_t N)
{
    std::vector<std::vector<T>> split_vec(N);
    uint32_t size = vec.size() / N;
    for(int i = 0; i < N; i++)
    {
        split_vec[i].reserve(size);
        std::copy(vec.begin() + i * size, vec.begin() + (i + 1) * size, std::back_inserter(split_vec[i]));
    }
    return split_vec;
}

template <typename T>
std::vector<T> merge_vectors(const std::vector<std::vector<T>> &vectors)
{
    std::vector<T> merged;
    uint32_t size = 0;
    for(int i = 0; i < vectors.size(); i++)
    {
        size += vectors[i].size();
    }
    merged.reserve(size);
    for (auto &v : vectors)
    {
        merged.insert(merged.end(), v.begin(), v.end());
    }
    return merged;
}

#endif
