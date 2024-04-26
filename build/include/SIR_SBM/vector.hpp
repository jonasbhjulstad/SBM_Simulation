// vector.hpp
//

#ifndef LZZ_vector_hpp
#define LZZ_vector_hpp
#include <SIR_SBM/common.hpp>
#include <numeric>
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

template <typename T>
std::vector<uint32_t> subvector_sizes(const std::vector<std::vector<T>>& vecs)
{
    std::vector<uint32_t> result;
    result.reserve(vecs.size());
    for(const auto& vec: vecs)
    {
        result.push_back(vec.size());
    }
    return result;
}


template <typename T>
std::vector<T> make_iota(const T& N)
{
    std::vector<T> result(N);
    std::iota(result.begin(), result.end(), 0);
    return result;
}

template <typename T>
std::vector<T> make_iota(const T& start, const T& end)
{
    std::vector<T> result(end - start);
    std::iota(result.begin(), result.end(), start);
    return result;
}
#define LZZ_INLINE inline
#undef LZZ_INLINE
#endif
