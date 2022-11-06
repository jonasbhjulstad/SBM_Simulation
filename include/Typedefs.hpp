#ifndef SYCL_GRAPH_EIGEN_TYPEDEFS_HPP
#define SYCL_GRAPH_EIGEN_TYPEDEFS_HPP
#include <cmath>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <limits>
#include <map>
#include <stdint.h>
namespace SYCL::Graph {

typedef Eigen::Matrix<float, -1, -1> Mat;
typedef Eigen::Vector<float, -1> Vec;
typedef Eigen::Vector<int, -1> iVec;
typedef const Eigen::Ref<const Vec> crVec;
typedef const Eigen::Ref<const Mat> crMat;
} // namespace SYCL::Graph
#endif