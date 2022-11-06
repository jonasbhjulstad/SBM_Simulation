#include <pybind11/pybind11.h>
#include "Bernoulli_SIR_MC.hpp"
#include "Bernoulli_SIS_MC.hpp"
PYBIND11_MODULE(Network_Models, m) {
  using namespace SYCL::Graph;
  namespace py = pybind11;
  m.doc() = "OpenMP-accelerated Bernoulli-SIR Monte-Carlo simulations";
py::class_<MC_SIR_Params>(m, "MC_SIR_Params")
.def_readwrite("N_pop", &MC_SIR_Params::N_pop)
.def_readwrite("p_ER", &MC_SIR_Params::p_ER)
.def_readwrite("N_sim", &MC_SIR_Params::p_I0)
.def_readwrite("p_I_min", &MC_SIR_Params::p_I_min)
.def_readwrite("p_I_max", &MC_SIR_Params::p_I_max)
.def_readwrite("N_sim", &MC_SIR_Params::N_sim)
.def_readwrite("Nt", &MC_SIR_Params::Nt_max)
.def_readwrite("p_R", &MC_SIR_Params::p_R);

py::class_<MC_SIS_Params>(m, "MC_SIS_Params")
.def_readwrite("N_pop", &MC_SIS_Params::N_pop)
.def_readwrite("p_ER", &MC_SIS_Params::p_ER)
.def_readwrite("N_sim", &MC_SIS_Params::p_I0)
.def_readwrite("p_I_min", &MC_SIS_Params::p_I_min)
.def_readwrite("p_I_max", &MC_SIS_Params::p_I_max)
.def_readwrite("N_sim", &MC_SIS_Params::N_sim)
.def_readwrite("Nt", &MC_SIS_Params::Nt_max)
.def_readwrite("p_R", &MC_SIS_Params::p_S);
}