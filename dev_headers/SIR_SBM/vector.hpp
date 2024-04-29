#pragma once
#hdr
#include <SIR_SBM/common.hpp>
#include <numeric>
#include <SIR_SBM/vector2D.hpp>
#include <SIR_SBM/vector3D.hpp>
#end
namespace SIR_SBM {


template <typename T>
std::vector<T> get_at(const std::vector<T> &vec,
                      const std::vector<uint32_t> &indices) {
  std::vector<T> result(indices.size());
  std::transform(indices.begin(), indices.end(), result.begin(),
                 [&](auto idx) { return vec[idx]; });
  return result;
}

template <typename T>
void set_at(std::vector<T> &vec, const std::vector<T> &vals,
            const std::vector<uint32_t> &indices) {
  std::transform(indices.begin(), indices.end(), vals.begin(), vec.begin(),
                 [](auto idx, auto val) { return val; });
}


template <typename T> std::vector<T> vector_merge(const Vec2D<T> &vecs) {
  std::vector<T> result;
  int N = std::accumulate(
      vecs.begin(), vecs.end(), 0L,
      [](size_t a, const std::vector<T> &b) { return a + b.size(); });
  result.reserve(N);
  for (const auto &vec : vecs) {
    result.insert(result.end(), vec.begin(), vec.end());
  }
  return result;
}

template <typename T>
std::vector<uint32_t> subvector_sizes(const Vec2D<T> &vecs) {
  std::vector<uint32_t> result;
  result.reserve(vecs.size());
  for (const auto &vec : vecs) {
    result.push_back(vec.size());
  }
  return result;
}

template <typename T> std::vector<T> make_iota(const T &N) {
  std::vector<T> result(N);
  std::iota(result.begin(), result.end(), 0);
  return result;
}
#hdr
typedef std::vector<std::pair<uint32_t, uint32_t>> PairVec;
#end
PairVec make_iota(const uint32_t &n0, const uint32_t n1) {
  PairVec result;
  for (uint32_t i = n0; i < n1; i++) {
    for (uint32_t j = n0; j < n1; j++) {
      result.push_back({i, j});
    }
  }
  return result;
}

std::vector<uint32_t> count_occurrences(const std::vector<uint32_t> &vec,
                                        uint32_t N) {
  std::vector<uint32_t> result(N);
  for (auto val : vec) {
    result[val]++;
  }
  return result;
}

} // namespace SIR_SBM