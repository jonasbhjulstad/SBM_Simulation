#include <FROLS_Algorithm.hpp>
#include <FROLS_Features.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
std::vector<double> single_polynomial_transform_wrapper(Eigen::Ref<const Mat> X,
                                                        size_t d_max,
                                                        size_t target_idx) {
  Vec Result = FROLS::Features::Polynomial::single_feature_transform(
      X, d_max, target_idx);
  return std::vector<double>(Result.data(), Result.data() + Result.rows());
};

PYBIND11_MODULE(pyFROLS, m) {
  using namespace FROLS;
  using namespace FROLS::Regression;
  namespace py = pybind11;
  m.doc() = "OpenMP-enabled Forward Orthogonal Least Squares Regression";
  m.def("feature_select", &Features::feature_select,
        "Selects the best feature to add to the feature set");
  m.def("used_feature_orthogonalize", &used_feature_orthogonalize,
        "Orthogonalizes x with respect to previously selected features in Q");
  m.def("single_response_regression", &single_response_regression,
        "Computes feature coefficients for feature batch X with respect to a "
        "single response variable y");
  m.def("multiple_response_regression", &multiple_response_regression,
        "Computes feature coefficients for feature batch X with respect to "
        "multiple response variables Y");
  m.def("feature_names", &FROLS::Features::Polynomial::feature_names,
        "Returns a vector of feature names constructed from polynomial "
        "expansions");
  py::class_<Feature>(m, "Feature")
      .def_readwrite("index", &Feature::index)
      .def_readwrite("ERR", &Feature::ERR)
      .def_readwrite("g", &Feature::g)
      .def_readwrite("coeff", &Feature::coeff);
  m.def("polynomial_feature_transform",
        &FROLS::Features::Polynomial::feature_transform,"Polynomial feature transform");
  m.def("polynomial_single_feature_transform",
        &single_polynomial_transform_wrapper,
        "Single polynomial feature transform");
  m.def("polynomial_feature_name", &FROLS::Features::Polynomial::feature_name,
        "Names corresponding to polynomial features at a given index");
  m.def("polynomial_feature_names", &FROLS::Features::Polynomial::feature_names,
        "Names corresponding to polynomial features");
}
