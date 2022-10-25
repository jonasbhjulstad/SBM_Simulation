
#include <Regressor.hpp>
#include <Quantile_Regressor.hpp>
#include <Polynomial_Discrete.hpp>
#include <Bernoulli_SIR_MC.hpp>
#include <ERR_Regressor.hpp>
#include <Typedefs.hpp>
#include <FROLS_Eigen.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>


PYBIND11_MODULE(pyFROLS, m)
{
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
            .def("simulate", &Feature_Model::simulate)
            .def("read_csv", &Feature_Model::read_csv);

        py::class_<Polynomial_Model, Feature_Model>(m, "Polynomial_Model")
            .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t>())

            // .def("get_features", &Polynomial_Model::get_features)
            .def("write_csv", &Polynomial_Model::write_csv)
            .def("feature_summary", &Polynomial_Model::feature_summary)
            .def("feature_name", &Polynomial_Model::feature_name)
            .def("feature_names", &Polynomial_Model::feature_names)
            .def("equation", &Polynomial_Model::model_equation);
            // .def("equations", &Polynomial_Model::model_equations);

        // py::class_<Regressor>(m, "Regressor")
        //     //   .def(py::init<float>())
        //     .def("transform_fit", static_cast<void (Regressor::*)(const std::vector<std::string>&, const std::vector<std::string>&, const std::vector<std::string>&, Features::Feature_Model&)>(&Regressor::transform_fit))
        //     .def("fit", &Regressor::fit);

        // py::class_<Quantile_Regressor, Regressor>(m, "Quantile_Regressor")
        //     .def(py::init<Quantile_Param>());
        // py::class_<ERR_Regressor, Regressor>(m, "ERR_Regressor")
        //     .def(py::init<Regressor_Param>());

        m.def("Bernoulli_SIR_MC_Simulations", &FROLS::Bernoulli_SIR_MC_Simulations<50>);
}