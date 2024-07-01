#pragma once
#hdr
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <SIR_SBM/vector.hpp>
template <typename T> using VecMap = Eigen::Map<const Eigen::Vector<T, -1>>;
#end

namespace SIR_SBM {

template <typename T>
Vec3D<T> make_MatrixVec(int N0, int N1, int N2) {
  return Vec3D<T>(N0, Vec2DView<T>(N1, N2));
}

template <typename Scalar>
Eigen::Vector<Scalar, -1> eigen_vec_convert(const std::vector<Scalar> &vec) {
  return VecMap<Scalar>(vec.data(), vec.size());
}
} // namespace SIR_SBM