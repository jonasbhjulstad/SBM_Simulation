#ifndef EIGEN_FROLS_TYPEDEFS_HPP
#define EIGEN_FROLS_TYPEDEFS_HPP
#include <Eigen/Dense>
#include <fstream>
#include <string>
#include <vector>
typedef Eigen::Matrix<double, -1, -1> Mat;
typedef Eigen::Vector<double, -1> Vec;
typedef Eigen::Vector<int, -1> iVec;

namespace FROLS {

struct Feature {
  double ERR = 0; // Error Reduction Ratio
  double g;       // Feature (Orthogonalized Linear-in-the-parameters form)
  size_t index;   // Index of the feature in the original feature set
  double coeff = 0;
};
struct Regression_Model {
  std::vector<std::vector<Feature>>
      features; // Best regression-features in ERR-decreasing order
  size_t N_control_inputs;
  size_t N_states;
  Regression_Model(size_t N_states, size_t N_control_inputs)
      : N_control_inputs(N_control_inputs), N_states(N_states) {}
};

} // namespace FROLS
#endif