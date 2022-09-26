#ifndef FROLS_EIGEN_TYPEDEFS_HPP
#define FROLS_EIGEN_TYPEDEFS_HPP
#include <cmath>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <limits>
namespace FROLS {

typedef Eigen::Matrix<double, -1, -1> Mat;
typedef Eigen::Vector<double, -1> Vec;
typedef Eigen::Vector<int, -1> iVec;
typedef const Eigen::Ref<const Vec> crVec;
typedef const Eigen::Ref<const Mat> crMat;


struct Feature {
  double f_ERR = -std::numeric_limits<double>::infinity(); // objective/Error Reduction Ratio
  double g;       // Feature (Orthogonalized Linear-in-the-parameters form)
  size_t index;   // Index of the feature in the original feature set
  double theta = 0;
};
} // namespace FROLS
#endif