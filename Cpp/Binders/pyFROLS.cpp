#include "FROLS_Polynomial.hpp"
#include <FROLS_Algorithm.hpp>
#include <FROLS_Features.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

PYBIND11_MODULE(pyFROLS, m) {
  using namespace FROLS;
  using namespace FROLS::Regression;
  namespace py = pybind11;
  m.def("multiple_response_regression",
        &FROLS::Regression::multiple_response_regression);
  m.doc() = "OpenMP-enabled Forward Orthogonal Least Squares Regression";
  m.def("feature_select", &Features::feature_select,
        "Selects the best feature to add to the feature set");
  m.def("used_feature_orthogonalize", &used_feature_orthogonalize,
        "Orthogonalizes x with respect to previously selected features in Q");
  m.def("polynomial_feature_name", &FROLS::Features::Polynomial::feature_name,
        "Names corresponding to polynomial features at a given index");
  m.def("single_response_regression",
        &FROLS::Regression::single_response_regression,
        "Computes feature coefficients for feature batch X with respect to a "
        "single response variable y");
  py::class_<Feature>(m, "Feature")
      .def(py::init<>())
      .def_readwrite("index", &Feature::index)
      .def_readwrite("ERR", &Feature::ERR)
      .def_readwrite("g", &Feature::g)
      .def_readwrite("coeff", &Feature::coeff);
  m.def("polynomial_feature_transform",
        &FROLS::Features::Polynomial::feature_transform,
        "Polynomial feature transform");
//   m.def("polynomial_single_feature_transform",
//         static_cast<Vec (*)(Eigen::Ref<const Mat>, size_t,
//                                             size_t)>(
//             &FROLS::Features::Polynomial::single_feature_transform),
//         "Single polynomial feature transform");
// m.def("polynomial_single_feature_transform",
// static_cast<double (*)(Eigen::Ref<const Vec>,size_t ,size_t)>(&FROLS::Features::Polynomial::single_feature_transform,
// "Single polynomial feature transform"
// );
//   m.def("polynomial_feature_names", &FROLS::Features::Polynomial::feature_names,
      //   "Names corresponding to polynomial features");
  m.def("polynomial_model_print",&FROLS::Features::Polynomial::model_print,
        "Prints the model");

py::class_<Features::Polynomial::Polynomial_Model>(m, "Polynomial_Model")
.def(py::init<size_t, size_t>())
.def("get_features", &Features::Polynomial::Polynomial_Model::get_features)
.def("print", &Features::Polynomial::Polynomial_Model::print)
.def("get_equations", &Features::Polynomial::Polynomial_Model::get_equations)
.def("transform", &Features::Polynomial::Polynomial_Model::transform)
.def("multiple_response_regression", &Features::Polynomial::Polynomial_Model::multiple_response_regression)
.def("step", &Features::Polynomial::Polynomial_Model::step)
.def("simulate", &Features::Polynomial::Polynomial_Model::simulate)
.def("feature_summary", &Features::Polynomial::Polynomial_Model::feature_summary)
.def("write_csv", &Features::Polynomial::Polynomial_Model::write_csv);
// m.def("multiple_response_regression")
}
