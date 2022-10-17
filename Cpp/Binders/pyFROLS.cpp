
#include <Polynomial_Discrete.hpp>
#include <Regressor.hpp>
#include <Quantile_Regressor.hpp>
#include <ERR_Regressor.hpp>
#include <Typedefs.hpp>
#include <FROLS_Eigen.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

PYBIND11_MODULE(pyFROLS, m) {
    using namespace FROLS;
    using namespace FROLS::Regression;
    using namespace FROLS::Features;
    namespace py = pybind11;
    py::module_ m_features =
            m.def_submodule("Features", "Feature selection models");
    py::module_ m_regression = m.def_submodule("Regression", "Regression models");

    py::class_<Feature>(m, "Feature")
            .def(py::init<>())
            .def_readwrite("index", &Feature::index)
            .def_readwrite("ERR", &Feature::f_ERR)
            .def_readwrite("g", &Feature::g)
            .def_readwrite("coeff", &Feature::theta);

    py::class_<Feature_Model>(m, "Feature_Model")
            .def("step", &Feature_Model::step)
            .def("simulate", &Feature_Model::simulate);

    py::class_<Polynomial_Model, Feature_Model>(m, "Polynomial_Model")
            .def(py::init<uint16_t, uint16_t, uint16_t, uint16_t>())
            .def("transform", static_cast<Mat (Polynomial_Model::*)(crMat &)>(
                    &Polynomial_Model::transform))
            .def("get_features", &Polynomial_Model::get_features)
            .def("write_csv", &Polynomial_Model::write_csv)
            .def("feature_summary", &Polynomial_Model::feature_summary)
            .def("feature_name", &Polynomial_Model::feature_name)
            .def("feature_names", &Polynomial_Model::feature_names)
            .def("model_equation", &Polynomial_Model::model_equation)
            .def("model_equations", &Polynomial_Model::model_equations);

    py::class_<Regressor>(m, "Regressor")
            //   .def(py::init<double>())
            .def("transform_fit", &Regressor::transform_fit)
            .def("fit", &Regressor::fit);

    py::class_<Quantile_Regressor, Regressor>(m, "Quantile_Regressor")
            .def(py::init<double, double, double, const std::string>())
            .def("transform_fit", &Quantile_Regressor::transform_fit)
            .def("fit", &Quantile_Regressor::fit);
    py::class_<ERR_Regressor, Regressor>(m, "ERR_Regressor")
            .def(py::init<double, double>())
            .def("transform_fit", &Quantile_Regressor::transform_fit)
            .def("fit", &Quantile_Regressor::fit);
}