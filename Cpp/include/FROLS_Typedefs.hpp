#ifndef EIGEN_FROLS_TYPEDEFS_HPP
#define EIGEN_FROLS_TYPEDEFS_HPP
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
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
  void summary() const {
    std::cout << "ERR: " << ERR << "\t"
              << "g: " << g << "\t"
              << "coeff: " << coeff << "\n";
  }
};

} // namespace FROLS
#endif