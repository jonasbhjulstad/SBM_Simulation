
#include <Bernoulli_SIR_MC_Dynamic.hpp>
#include <ERR_Regressor.hpp>
#include <FROLS_Eigen.hpp>
#include <Polynomial_Discrete.hpp>
#include <Quantile_Regressor.hpp>
#include <Regressor.hpp>
#include <Typedefs.hpp>
#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/operators.h>

PYBIND11_MODULE(pyFROLS, m) {
  using namespace FROLS;
  using namespace FROLS::Regression;
  using namespace FROLS::Features;
  using namespace Network_Models;
  namespace py = pybind11;
  py::module_ m_features =
      m.def_submodule("Features", "Feature selection models");
  py::module_ m_regression = m.def_submodule("Regression", "Regression models");

  py::class_<Feature>(m, "Feature")
      .def(py::init<>())
      .def_readwrite("ERR", &Feature::f_ERR)
      .def_readwrite("g", &Feature::g)
      .def_readwrite("index", &Feature::index)
      .def_readwrite("coeff", &Feature::theta)
      .def(py::pickle(
          [](const Feature &f) { // dump
            return py::make_tuple(f.f_ERR, f.g, f.index, f.theta);
          },
          [](py::tuple t) { // load
            return Feature{t[0].cast<float>(), t[1].cast<float>(),
                           t[2].cast<uint32_t>(), t[3].cast<float>(),
                           FEATURE_REGRESSION};
          }));

  py::class_<Feature_Model>(m, "Feature_Model")
      .def("step", &Feature_Model::step)
      .def("simulate", &Feature_Model::simulate)
      .def("read_csv", &Feature_Model::read_csv)
      .def("write_latex", &Feature_Model::write_latex);

  py::class_<Polynomial_Model, Feature_Model>(m, "Polynomial_Model")
      .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t>())
      // .def("get_features", &Polynomial_Model::get_features)
      .def("write_csv", &Polynomial_Model::write_csv)
      .def("feature_summary", &Polynomial_Model::feature_summary)
      .def("feature_name", &Polynomial_Model::feature_name)
      .def("feature_names", &Polynomial_Model::feature_names)
      .def("equation", &Polynomial_Model::model_equation)
      .def("get_N_features_max", &Polynomial_Model::get_N_features_max);

  py::class_<Regression_Data>(m, "Regression_Data")
      .def_readwrite("X", &Regression_Data::X)
      .def_readwrite("Y", &Regression_Data::Y)
      .def_readwrite("U", &Regression_Data::U);

  py::class_<Regressor_Param>(m, "Regressor_Param")
      .def(py::init<>())
      .def_readwrite("tol", &Regressor_Param::tol)
      .def_readwrite("theta_tol", &Regressor_Param::theta_tol)
      .def_readwrite("N_terms_max", &Regressor_Param::N_terms_max)
      .def_readwrite("f_ERR_tol", &Regressor_Param::f_ERR_tol);
  py::class_<Quantile_Param, Regressor_Param>(m, "Quantile_Param")
      .def(py::init<>())
      .def_readwrite("tau", &Quantile_Param::tau)
      .def_readwrite("N_rows", &Quantile_Param::N_rows)
      .def_readwrite("N_threads", &Quantile_Param::N_threads)
      .def_readonly("solver_type", &Quantile_Param::solver_type);
  py::class_<Regressor>(m, "Regressor")
    //   .def("fit", &Regressor::fit)
      .def("transform_fit", py::overload_cast<const std::vector<Mat>&, const std::vector<Mat>&, const std::vector<Mat>&,
                                              Features::Feature_Model &>(
                                &Regressor::transform_fit))
      .def(
          "transform_fit",
          py::overload_cast<const Regression_Data &, Features::Feature_Model &>(
              &Regressor::transform_fit))
      .def("theta_solve", &Regressor::theta_solve);

  py::class_<ERR_Regressor, Regressor>(m, "ERR_Regressor")
      .def(py::init<const Regressor_Param &>());

  py::class_<Quantile_Regressor, Regressor>(m, "Quantile_Regressor")
      .def(py::init<const Quantile_Param &>())
      .def_readonly("tau", &Quantile_Regressor::tau);

  using Bernoulli_Network =
      Network_Models::Vector_SIR_Bernoulli_Network<random::default_rng, float>;
  typedef VectorNetwork<SIR_Param<>, Bernoulli_Network> SIR_Network;
  py::class_<SIR_Network>(m, "SIR_Network")
      .def("simulate", &SIR_Network::simulate)
      .def("advance", &SIR_Network::advance)
      .def("reset", &SIR_Network::reset)
      .def("initialize", &SIR_Network::initialize)
      .def("population_count", &SIR_Network::population_count);

  py::class_<random::default_rng>(m, "default_rng").def(py::init<>());

  py::class_<Bernoulli_Network, SIR_Network>(m, "SIR_Bernoulli_Network")
      .def(py::init<SIR_VectorGraph &, float, float, random::default_rng>())
      .def_readonly("p_I0", &Bernoulli_Network::p_I0)
      .def_readonly("p_R0", &Bernoulli_Network::p_R0)
      .def_readonly("t", &Bernoulli_Network::t);
  // .def("advance", &Bernoulli_Network::advance)
  // .def("reset", &Bernoulli_Network::reset)
  // .def("initialize", &Bernoulli_Network::initialize)
  // .def("population_count", &Bernoulli_Network::population_count);

  py::class_<SIR_Param<float>>(m, "SIR_Param")
      .def(py::init<>())
      .def_readwrite("p_I", &SIR_Param<float>::p_I)
      .def_readwrite("p_R", &SIR_Param<float>::p_R)
      .def_readwrite("Nt_min", &SIR_Param<float>::Nt_min)
      .def_readwrite("N_I_min", &SIR_Param<float>::N_I_min);

  py::class_<MC_SIR_VectorData>(m, "MC_SIR_Data")
      .def_readwrite("trajectory", &MC_SIR_VectorData::traj)
      .def_readwrite("probabilities", &MC_SIR_VectorData::p_I);
  m.def("generate_interaction_probabilities",
        &generate_interaction_probabilities<random::default_rng, float>);

  m.def("fixed_interaction_probabilities",
        &fixed_interaction_probabilities<float>);

  py::class_<SIR_VectorGraph::Vertex_t>(m, "SIR_Vertex")
      .def_readwrite("id", &SIR_VectorGraph::Vertex_t::id)
      .def_readwrite("data", &SIR_VectorGraph::Vertex_t::data)
      .def_readonly("mx", &SIR_VectorGraph::Vertex_t::mx)
      .def(py::pickle(
          [](const SIR_VectorGraph::Vertex_t &v) { // dump
            std::map<SIR_State, std::string> SIR_State_es_map = {
                {SIR_State::SIR_S, "S"},
                {SIR_State::SIR_I, "I"},
                {SIR_State::SIR_R, "R"},
            };
            return py::make_tuple(v.id, SIR_State_es_map[v.data]);
          },
          [](py::tuple t) { // load
                      std::map<std::string, SIR_State> SIR_State_se_map = {
                {"S", SIR_State::SIR_S},
                {"I", SIR_State::SIR_I},
                {"R", SIR_State::SIR_R},
            };
            return SIR_VectorGraph::Vertex_t{
                t[0].cast<uint32_t>(),
                SIR_State_se_map[t[1].cast<std::string>()],
                std::make_shared<std::mutex>()};
          }));

  py::class_<SIR_VectorGraph::Edge_t>(m, "SIR_Edge")
      .def_readwrite("data", &SIR_VectorGraph::Edge_t::data)
      .def_readwrite("to", &SIR_VectorGraph::Edge_t::to)
      .def_readwrite("from", &SIR_VectorGraph::Edge_t::from)
      .def_readonly("mx", &SIR_VectorGraph::Edge_t::mx)
      .def(py::pickle(
          [](const SIR_VectorGraph::Edge_t &v) { // dump
            return py::make_tuple(v.to, v.from);
          },
          [](py::tuple t) { // load
            return SIR_VectorGraph::Edge_t{SIR_Edge{}, t[0].cast<uint32_t>(),
                                           t[1].cast<uint32_t>(),
                                           std::make_shared<std::mutex>()};
          }));

  py::class_<SIR_VectorGraph>(m, "SIR_Graph")
      .def("get_vertex_prop", &SIR_VectorGraph::get_vertex_prop)
      .def("get_vertex", &SIR_VectorGraph::get_vertex)
      .def("assign_vertex", &SIR_VectorGraph::assign_vertex)
      .def("add_vertex", &SIR_VectorGraph::add_vertex)
      .def("add_edge", &SIR_VectorGraph::add_edge)
      .def("node_prop", &SIR_VectorGraph::node_prop)
      .def("assign", &SIR_VectorGraph::assign)
      .def("remove_vertex", &SIR_VectorGraph::remove_vertex)
      .def("remove_edge", &SIR_VectorGraph::remove_edge)
      .def("is_in_edge", &SIR_VectorGraph::is_in_edge)
      .def("get_neighbor", &SIR_VectorGraph::get_neighbor)
      .def("neighbors", &SIR_VectorGraph::neighbors)
      .def("assign", &SIR_VectorGraph::operator=)
      .def(py::pickle(
          [](const SIR_VectorGraph &G) { // dump
            auto new_edges = G._edges;
            for (auto &e : new_edges) {
              e.mx = std::make_shared<std::mutex>();
            }
            auto new_vertices = G._vertices;
            for (auto &n : new_vertices) {
              n.mx = std::make_shared<std::mutex>();
            }
            return py::make_tuple(new_vertices, new_edges, G.N_vertices,
                                  G.N_edges);
          },
          [](py::tuple t) { // load
            return SIR_VectorGraph{
                t[0].cast<std::vector<SIR_VectorGraph::Vertex_t>>(),
                t[1].cast<std::vector<SIR_VectorGraph::Edge_t>>()};
          }));

  m.def("generate_SIR_ER_graph", &generate_SIR_ER_graph);
  m.def("generate_Bernoulli_SIR_Network",
        py::overload_cast<Network_Models::SIR_VectorGraph &, float, uint32_t,
                          float>(&generate_Bernoulli_SIR_Network));
  // m.def("generate_Bernoulli_SIR_Network", py::overload_cast<uint32_t, float,
  // float, uint32_t, float>(&generate_Bernoulli_SIR_Network));
  m.def("traj_to_file", &traj_to_file<float>);
  py::class_<MC_SIR_Params<float>>(m, "MC_SIR_Params")
      .def(py::init<>())
      .def_readwrite("N_pop", &MC_SIR_Params<float>::N_pop)
      .def_readwrite("p_ER", &MC_SIR_Params<float>::p_ER)
      .def_readwrite("p_I0", &MC_SIR_Params<float>::p_I0)
      .def_readwrite("p_R0", &MC_SIR_Params<float>::p_R0)
      .def_readwrite("R0_max", &MC_SIR_Params<float>::R0_max)
      .def_readwrite("R0_min", &MC_SIR_Params<float>::R0_min)
      .def_readwrite("alpha", &MC_SIR_Params<float>::alpha)
      .def_readwrite("p_I", &MC_SIR_Params<float>::p_I)
      .def_readwrite("N_sim", &MC_SIR_Params<float>::N_sim)
      .def_readwrite("Nt_min", &MC_SIR_Params<float>::Nt_min)
      .def_readwrite("p_R", &MC_SIR_Params<float>::p_R)
      .def_readwrite("seed", &MC_SIR_Params<float>::seed)
      .def_readwrite("N_I_min", &MC_SIR_Params<float>::N_I_min)
      .def_readwrite("iter_offset", &MC_SIR_Params<float>::iter_offset)
      .def_readwrite("csv_termination_tol",
                     &MC_SIR_Params<float>::csv_termination_tol);

  m.def("MC_SIR_simulation",
        py::overload_cast<Network_Models::SIR_VectorGraph &,
                          const MC_SIR_Params<> &, uint32_t,
                          const std::vector<float> &>(&MC_SIR_simulation));
  m.def("MC_SIR_simulation",
        py::overload_cast<Network_Models::Vector_SIR_Bernoulli_Network<
                              random::default_rng, float> &,
                          const MC_SIR_Params<> &, uint32_t,
                          const std::vector<float> &>(&MC_SIR_simulation));
  m.def(
      "MC_SIR_simulations",
      py::overload_cast<Network_Models::SIR_VectorGraph &,
                        const MC_SIR_Params<> &, const std::vector<uint32_t> &,
                        uint32_t>(&MC_SIR_simulations));
  m.def(
      "MC_SIR_simulations",
      py::overload_cast<Network_Models::Vector_SIR_Bernoulli_Network<
                            random::default_rng, float> &,
                        const MC_SIR_Params<> &, const std::vector<uint32_t> &,
                        const std::vector<float> &, uint32_t>(
          &MC_SIR_simulations));
  m.def("MC_SIR_simulation",
        py::overload_cast<Network_Models::Vector_SIR_Bernoulli_Network<
                              random::default_rng, float> &,
                          const MC_SIR_Params<> &, uint32_t,
                          const std::vector<float> &>(&MC_SIR_simulation));
  m.def("MC_SIR_simulations",
        py::overload_cast<uint32_t, float, float, const std::vector<uint32_t> &,
                          std::vector<float>, uint32_t, uint32_t>(
            &MC_SIR_simulations));
  m.def("MC_SIR_simulations",
        py::overload_cast<uint32_t, float, float, const std::vector<uint32_t> &,
                          uint32_t, uint32_t>(&MC_SIR_simulations));

  m.def(
      "MC_SIR_simulations_to_regression",
      py::overload_cast<Network_Models::SIR_VectorGraph &,
                        const MC_SIR_Params<> &, const std::vector<uint32_t> &,
                        uint32_t>(&MC_SIR_simulations_to_regression));
  m.def(
      "MC_SIR_simulations_to_regression",
      py::overload_cast<const MC_SIR_Params<> &, const std::vector<uint32_t> &,
                        uint32_t>(&MC_SIR_simulations_to_regression));

  m.def("MC_SIR_simulations_to_regression",
        py::overload_cast<Network_Models::SIR_VectorGraph &,
                          const MC_SIR_Params<> &,
                          const std::vector<Network_Models::SIR_Param<>> &,
                          const std::vector<uint32_t> &, uint32_t>(
            &MC_SIR_simulations_to_regression));
}
