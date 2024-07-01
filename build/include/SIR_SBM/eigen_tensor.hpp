// eigen.hpp
//

#ifndef LZZ_SIR_SBM_LZZ_eigen_tensor_hpp
#define LZZ_SIR_SBM_LZZ_eigen_tensor_hpp
#line 3 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen.hpp"
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <vector>
template <typename T> using VecMap = Eigen::Map<const Eigen::Vector<T, -1>>;
#line 10 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen.hpp"
namespace SIR_SBM
{
  template <typename Scalar, int rank>
Eigen::Matrix<Scalar, -1, -1>
Tensor_to_Matrix(const Eigen::Tensor<Scalar, rank> &&tensor, size_t rows,
                 size_t cols) {
  return Eigen::Map<const Eigen::Matrix<Scalar, -1, -1>>(tensor.data(), rows,
                                                         cols);
}
}
#define LZZ_INLINE inline
#line 22 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen.hpp"
namespace SIR_SBM
{
#line 24 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen.hpp"
  template <typename Scalar>
#line 25 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen.hpp"
  Eigen::Vector <Scalar, -1> eigen_vec_convert (std::vector <Scalar> const & vec);
}
#line 22 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen.hpp"
namespace SIR_SBM
{
#line 24 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen.hpp"
  template <typename Scalar>
#line 25 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen.hpp"
  Eigen::Vector <Scalar, -1> eigen_vec_convert (std::vector <Scalar> const & vec)
#line 25 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen.hpp"
                                                                            {
  return VecMap<Scalar>(vec.data(), vec.size());
}
}
#undef LZZ_INLINE
#endif
