#pragma once
#hdr
#include <SIR_SBM/common.hpp>
#include <numeric>
#end

namespace SIR_SBM {

template <typename T> struct LinearVector2D : public std::vector<T> {
  LinearVector2D(size_t N1, size_t N2)
      : std::vector<T>(N1 * N2), N1(N1), N2(N2) {}
  const T &operator()(size_t i, size_t j) const { return (*this)[i * N2 + j]; }
  LinearVector2D(const T *begin, const T *end) {
    auto N = std::distance(begin, end);
    this->std::vector<T>::resize(N);
    std::copy(begin, end, this->begin());
  }

  const std::vector<T> operator()(size_t i) const {
    return std::vector<T>(this->begin() + i * N2, this->begin() + (i + 1) * N2);
  }

  // T &operator()(size_t i, size_t j) { return (*this)[i * N2 + j]; }

  void resize(size_t N1, size_t N2) {
    this->std::vector<T>::resize(N1 * N2);
    this->N1 = N1;
    this->N2 = N2;
  }

  uint32_t get_linear_idx(size_t i, size_t j) const { return i * N2 + j; }

  LinearVector2D<T> get_rows(const std::vector<uint32_t> &rows) const {
    LinearVector2D<T> result(rows.size(), N2);
    for (size_t i = 0; i < rows.size(); i++) {
      for (size_t j = 0; j < N2; j++) {
        result(i, j) = (*this)(rows[i], j);
      }
    }
    return result;
  }

  size_t N1, N2;
};
template <typename T>
void set_at_row(LinearVector2D<T> &vec, const std::vector<T> &vals,
                const uint32_t &row) {
  std::copy(vals.begin(), vals.end(), vec.begin() + row * vec.N2);
}
template <typename T>
LinearVector2D<T> differentiate(const LinearVector2D<T> &vec) {
  LinearVector2D<T> result(vec.N1 - 1, vec.N2);
  for (size_t i = 0; i < vec.N1 - 1; i++) {
    for (size_t j = 0; j < vec.N2; j++) {
      result(i, j) = vec(i + 1, j) - vec(i, j);
    }
  }
  return result;
}
} // namespace SIR_SBM