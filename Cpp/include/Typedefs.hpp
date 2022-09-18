#ifndef FROLS_EIGEN_TYPEDEFS_HPP
#define FROLS_EIGEN_TYPEDEFS_HPP
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cstdint>

typedef Eigen::Matrix<double, -1, -1> Mat;
typedef Eigen::Vector<double, -1> Vec;
typedef Eigen::Vector<int, -1> iVec;
typedef const Eigen::Ref<const Vec> crVec;
typedef const Eigen::Ref<const Mat> crMat;
typedef const Eigen::Ref<const iVec> ciVecRef;

namespace FROLS {

struct Feature {
  double f_ERR = -1; // objective/Error Reduction Ratio
  double g;       // Feature (Orthogonalized Linear-in-the-parameters form)
  size_t index;   // Index of the feature in the original feature set
  double theta = 0;
};

namespace Features {
typedef Feature (*FeatureFunction)(const Mat &, const Vec &, const iVec &, const double*);
}
} // namespace FROLS
#endif