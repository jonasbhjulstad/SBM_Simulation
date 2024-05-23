#pragma once
#hdr
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <vector>
template <typename T> using VecMap = Eigen::Map<const Eigen::Vector<T, -1>>;
#end

namespace SIR_SBM {
template <typename Scalar, int rank, typename sizeType>
Eigen::Matrix<Scalar, -1, -1>
Tensor_to_Matrix(const Eigen::Tensor<Scalar, rank> &tensor, const sizeType rows,
                 const sizeType cols) {
  return Eigen::Map<const Eigen::Matrix<Scalar, -1, -1>>(tensor.data(), rows,
                                                         cols);
}

template <typename Scalar>
Eigen::Vector<Scalar, -1> eigen_vec_convert(const std::vector<Scalar> &vec) {
  return VecMap<Scalar>(vec.data(), vec.size());
}
} // namespace SIR_SBM