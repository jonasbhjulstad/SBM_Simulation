#include <FROLS.hpp>
#include <FROLS_Features.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
PYBIND11_MODULE(FROLS, m) {
  using namespace FROLS;
  using namespace FROLS::Regression;
  namespace py = pybind11;
  m.doc() = "OpenMP-enabled Forward Orthogonal Least Squares Regression";
  m.def("feature_select", &feature_select,
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
"Returns a vector of feature names constructed from polynomial expansions");
  py::class_<Regression_Data>(m, "MC_SIR_Params")
      .def_readwrite("best_features", &Regression_Data::best_features)
      .def_readwrite("coefficients", &Regression_Data::coefficients);
  py::class_<Feature>(m, "Feature")
      .def_readwrite("index", &Feature::index)
      .def_readwrite("ERR", &Feature::ERR)
      .def_readwrite("g", &Feature::g);
}