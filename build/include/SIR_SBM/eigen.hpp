// eigen.hpp
//

#ifndef LZZ_SIR_SBM_LZZ_eigen_hpp
#define LZZ_SIR_SBM_LZZ_eigen_hpp
#line 3 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen.hpp"
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <SIR_SBM/vector.hpp>
template <typename T> using VecMap = Eigen::Map<const Eigen::Vector<T, -1>>;
#define LZZ_INLINE inline
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen.hpp"
namespace SIR_SBM
{
#line 11 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen.hpp"
  template <typename T>
#line 12 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen.hpp"
  Vec3D <T> make_MatrixVec (int N0, int N1, int N2);
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen.hpp"
namespace SIR_SBM
{
#line 16 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen.hpp"
  template <typename Scalar>
#line 17 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen.hpp"
  Eigen::Vector <Scalar, -1> eigen_vec_convert (std::vector <Scalar> const & vec);
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen.hpp"
namespace SIR_SBM
{
#line 11 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen.hpp"
  template <typename T>
#line 12 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen.hpp"
  Vec3D <T> make_MatrixVec (int N0, int N1, int N2)
#line 12 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen.hpp"
                                                {
  return Vec3D<T>(N0, Vec2DView<T>(N1, N2));
}
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen.hpp"
namespace SIR_SBM
{
#line 16 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen.hpp"
  template <typename Scalar>
#line 17 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen.hpp"
  Eigen::Vector <Scalar, -1> eigen_vec_convert (std::vector <Scalar> const & vec)
#line 17 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen.hpp"
                                                                            {
  return VecMap<Scalar>(vec.data(), vec.size());
}
}
#undef LZZ_INLINE
#endif
