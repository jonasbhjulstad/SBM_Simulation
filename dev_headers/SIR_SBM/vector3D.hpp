#pragma once
#hdr
#include <SIR_SBM/common.hpp>
#include <SIR_SBM/vector3D.hpp>
#include <numeric>
#end
namespace SIR_SBM {
template <typename T> struct LinearVector3D : public std::vector<T> {
  LinearVector3D(size_t N1, size_t N2, size_t N3)
      : std::vector<T>(N1 * N2 * N3), N1(N1), N2(N2), N3(N3) {}
  const T &operator()(size_t i, size_t j, size_t k) const {
    return (*this)[i * N2 * N3 + j * N3 + k];
  }
  T &operator()(size_t i, size_t j, size_t k) {
    return (*this)[i * N2 * N3 + j * N3 + k];
  }

  void resize(size_t N1, size_t N2, size_t N3) {
    this->std::vector<T>::resize(N1 * N2 * N3);
    this->N1 = N1;
    this->N2 = N2;
    this->N3 = N3;
  }

  uint32_t get_linear_idx(size_t i, size_t j, size_t k) const {
    return i * N2 * N3 + j * N3 + k;
  }

  LinearVector2D<T> get_row(size_t i) const {
    const T* start = this->data() + get_linear_idx(i, 0, 0);
    const T* end = start + N2 * N3;
    return LinearVector2D<T>(start, end);
  }
  typedef std::vector<T>::iterator iterator;
  size_t N1, N2, N3;
};

template <typename T>
void set_at_row(LinearVector3D<T> &vec, const LinearVector2D<T> &vals,
                const uint32_t &row) {
  std::copy(vals.begin(), vals.end(), vec.begin() + row * vec.N2 * vec.N3);
}
} // namespace SIR_SBM