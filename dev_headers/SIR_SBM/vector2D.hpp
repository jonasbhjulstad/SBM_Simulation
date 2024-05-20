#pragma once
#hdr
#include <SIR_SBM/common.hpp>
#include <numeric>
#include <fstream>
#end

namespace SIR_SBM {

template <typename T> struct LinearVector2D : public std::vector<T> {
  LinearVector2D(): N0(0), N1(0) {}
  LinearVector2D(size_t N0, size_t N1)
      : std::vector<T>(N0 * N1), N0(N0), N1(N1) {}
  const T &operator()(size_t i, size_t j) const { return (*this)[i * N1 + j]; }
  LinearVector2D(const T *begin, const T *end, size_t N0, size_t N1): N0(N0), N1(N1), std::vector<T>(N0 * N1)
  {
    std::copy(begin, end, this->begin());
  }

  typedef typename std::vector<T>::iterator iterator;

  const std::vector<T> operator()(size_t i) const {
    return std::vector<T>(this->begin() + i * N1, this->begin() + (i + 1) * N1);
  }
  iterator row_begin(size_t i) { return this->begin() + i * N1; }
  iterator row_end(size_t i) { return this->begin() + (i + 1) * N1; }

  void add_to_row(const std::vector<T>& val, size_t row)
  {
    std::transform(this->row_begin(row), this->row_end(row), val.begin(), this->begin(), std::plus<T>());
  }

  void resize(size_t N0, size_t N1) {
    this->std::vector<T>::resize(N0 * N1);
    this->N0 = N0;
    this->N1 = N1;
  }

  uint32_t get_linear_idx(size_t i, size_t j) const { return i * N1 + j; }

  LinearVector2D<T> get_rows(const std::vector<uint32_t> &rows) const {
    LinearVector2D<T> result(rows.size(), N1);
    for (size_t i = 0; i < rows.size(); i++) {
      for (size_t j = 0; j < N1; j++) {
        result(i, j) = (*this)(rows[i], j);
      }
    }
    return result;
  }

  friend std::ofstream& operator<<(std::ofstream &f, const LinearVector2D<T> &vec) {
    for (size_t i = 0; i < vec.N0; i++) {
      for (size_t j = 0; j < vec.N1; j++) {
        f << vec(i, j) << ",";
      }
      f << std::endl;
    }
    return f;
  }

  size_t N0, N1;
};
template <typename T>
void set_at_row(LinearVector2D<T> &vec, const std::vector<T> &vals,
                const uint32_t &row) {
  std::copy(vals.begin(), vals.end(), vec.begin() + row * vec.N1);
}
template <typename T>
LinearVector2D<T> differentiate(const LinearVector2D<T> &vec) {
  LinearVector2D<T> result(vec.N0 - 1, vec.N1);
  for (size_t i = 0; i < vec.N0 - 1; i++) {
    for (size_t j = 0; j < vec.N1; j++) {
      result(i, j) = vec(i + 1, j) - vec(i, j);
    }
  }
  return result;
}
} // namespace SIR_SBM