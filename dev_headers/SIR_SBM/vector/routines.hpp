#pragma once
#hdr
#include <SIR_SBM/vector/types.hpp>
#end
namespace SIR_SBM
{

template <typename T>
std::tuple<uint32_t, uint32_t, uint32_t> get_vector_shape(const Vec3D<T> &vec)
{
    return std::make_tuple(vec.size(), vec[0].size(), vec[0][0].size());
}
//for Vec2D
template <typename T>
std::tuple<uint32_t, uint32_t> get_vector_shape(const Vec2D<T> &vec)
{
    return std::make_tuple(vec.size(), vec[0].size());
}

    template <typename T>
std::vector<uint32_t> get_vector_sizes(const Vec2D<T>& vecs)
{
  std::vector<uint32_t> sizes(vecs.size());
  std::transform(vecs.begin(), vecs.end(), sizes.begin(),
                 [](const std::vector<T>& vec) { return vec.size(); });
  return sizes;
}
template <typename T>
std::vector<T> vector_merge(const Vec2D<T> &vecs) {
  std::vector<T> result;
  int N = std::accumulate(
      vecs.begin(), vecs.end(), 0L,
      [](uint32_t a, const std::vector<T> &b) { return a + b.size(); });
  result.reserve(N);
  for (const auto &vec : vecs) {
    result.insert(result.end(), vec.begin(), vec.end());
  }
  return result;
}

template <typename T = uint32_t>
std::vector<T> make_iota(uint32_t N)
{
    std::vector<T> result(N);
    std::iota(result.begin(), result.end(), 0);
    return result;
}

}