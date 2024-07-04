#pragma once
#hdr
#include <SIR_SBM/vector/types.hpp>
#include <cstdint>
#include <tuple>
#include <memory>
#end
#hdr
#include <numeric>
#end
#hdr
namespace SIR_SBM {
template <typename T> using Vec2D = std::vector<std::vector<T>>;
template <typename T> using Vec3D = std::vector<std::vector<std::vector<T>>>;
} // namespace SIR_SBM
#end

namespace SIR_SBM {

template <typename T>
Vec3D<T> make_Vec3D(uint32_t N0, uint32_t N1, uint32_t N2) {
  return Vec3D<T>(N0, Vec2D<T>(N1, std::vector<T>(N2, T{})));
}

template <typename T>
std::tuple<uint32_t, uint32_t, uint32_t> get_vector_shape(const Vec3D<T> &vec) {
  return std::make_tuple(vec.size(), vec[0].size(), vec[0][0].size());
}
// for Vec2D
template <typename T>
std::tuple<uint32_t, uint32_t> get_vector_shape(const Vec2D<T> &vec) {
  return std::make_tuple(vec.size(), vec[0].size());
}

template <typename T>
std::vector<uint32_t> get_vector_sizes(const Vec2D<T> &vecs) {
  std::vector<uint32_t> sizes(vecs.size());
  std::transform(vecs.begin(), vecs.end(), sizes.begin(),
                 [](const std::vector<T> &vec) { return vec.size(); });
  return sizes;
}
template <typename T> std::vector<T> vector_merge(const Vec2D<T> &vecs) {
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

template <typename T = uint32_t> std::vector<T> make_iota(uint32_t N) {
  std::vector<T> result(N);
  std::iota(result.begin(), result.end(), 0);
  return result;
}

template <typename T> Vec2D<T> vstack(const Vec3D<T> &data) {
  auto sizes = get_vector_sizes(data);
  auto N = std::accumulate(sizes.begin(), sizes.end(), 0);
  Vec2D<T> result(N);

  int idx = 0;
  for (const auto &vec : data) {
    for (const auto &val : vec) {
      result[idx++] = val;
    }
  }
  return result;
}

template <typename T0, typename T1>
Vec3D<T1> dtype_convert(const Vec3D<T0> &data) {
  Vec3D<T1> result =
      make_Vec3D<T1>(data.size(), data[0].size(), data[0][0].size());
  for (int i = 0; i < data.size(); i++) {
    for (int j = 0; j < data[0].size(); j++) {
      for (int k = 0; k < data[0][0].size(); k++) {
        result[i][j][k] = static_cast<T1>(data[i][j][k]);
      }
    }
  }
  return result;
}

template <typename T>
void vector_add(std::vector<T> &data, const std::vector<T> &other) {
  std::transform(data.begin(), data.end(), other.begin(), data.begin(),
                 std::plus<T>());
}

template <typename T>
std::vector<T> get_column(const Vec2D<T> &data, uint32_t col) {
  std::vector<T> result(data.size());
  for (int i = 0; i < data.size(); i++) {
    result[i] = data[i][col];
  }
  return result;
}
template <typename T>
std::shared_ptr<T> make_shared_array(std::size_t N)
{
  return std::make_shared<T>(N);
}
template <typename T> std::vector<T> get_linear_vector(const Vec3D<T> &data) {
  std::vector<T> result(data.size() * data[0].size() * data[0][0].size());
  for (int i = 0; i < data.size(); i++) {
    for (int j = 0; j < data[0].size(); j++) {
      for (int k = 0; k < data[0][0].size(); k++) {
        result[i * data[0].size() * data[0][0].size() + j * data[0][0].size() +
               k] = data[i][j][k];
      }
    }
  }
  return result;
}

uint32_t get_linear_idx(std::tuple<uint32_t, uint32_t> idx,
                        std::tuple<uint32_t, uint32_t> shape) {
  auto [N0, N1] = shape;
  auto [i, j] = idx;
  return i * N1 + j;
}

uint32_t get_linear_idx(std::tuple<uint32_t, uint32_t, uint32_t> idx,
                        std::tuple<uint32_t, uint32_t, uint32_t> shape) {
  auto [N0, N1, N2] = shape;
  auto [i, j, k] = idx;
  return i * N1 * N2 + j * N2 + k;
}

uint32_t get_row_offset(uint32_t row, std::tuple<uint32_t, uint32_t> shape) {
  auto [N0, N1] = shape;
  return row * N1;
}

uint32_t get_row_offset(uint32_t row,
                        std::tuple<uint32_t, uint32_t, uint32_t> shape) {
  auto [N0, N1, N2] = shape;
  return row * N1 * N2;
}

} // namespace SIR_SBM