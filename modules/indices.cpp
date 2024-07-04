module;

#include <tuple>
#include <cstdint>
export module SIR_SBM.indices;

export uint32_t get_linear_idx(std::tuple<uint32_t, uint32_t> idx,
                        std::tuple<uint32_t, uint32_t> shape) {
  auto [N0, N1] = shape;
  auto [i, j] = idx;
  return i * N1 + j;
}

export uint32_t get_linear_idx(std::tuple<uint32_t, uint32_t, uint32_t> idx,
                        std::tuple<uint32_t, uint32_t, uint32_t> shape) {
  auto [N0, N1, N2] = shape;
  auto [i, j, k] = idx;
  return i * N1 * N2 + j * N2 + k;
}

export uint32_t get_row_offset(uint32_t row, std::tuple<uint32_t, uint32_t> shape) {
  auto [N0, N1] = shape;
  return row * N1;
}

export uint32_t get_row_offset(uint32_t row,
                        std::tuple<uint32_t, uint32_t, uint32_t> shape) {
  auto [N0, N1, N2] = shape;
  return row * N1 * N2;
}
export template <typename T>
std::vector<T> get_column(const Vec2D<T> &data, uint32_t col) {
  std::vector<T> result(data.size());
  for (int i = 0; i < data.size(); i++) {
    result[i] = data[i][col];
  }
  return result;
}