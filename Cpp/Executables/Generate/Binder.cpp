#include <pybind11/pybind11.h>
#include "Bernoulli_SIR_MC.hpp"
PYBIND11_MODULE(Bernoulli_SIR_MC, m) {
  using namespace FROLS;
  namespace py = pybind11;
  m.doc() = "OpenMP-accelerated Bernoulli-SIR Monte-Carlo simulations";
py::class_<MC_SIR_Params>(m, "MC_SIR_Params")
.def_readwrite("N_pop", &MC_SIR_Params::N_pop)
.def_readwrite("p_ER", &MC_SIR_Params::p_ER)
.def_readwrite("N_sim", &MC_SIR_Params::p_I0)
.def_readwrite("p_I_min", &MC_SIR_Params::p_I_min)
.def_readwrite("p_I_max", &MC_SIR_Params::p_I_max)
.def_readwrite("N_sim", &MC_SIR_Params::N_sim_tot)
.def_readwrite("Nt", &MC_SIR_Params::Nt)
.def_readwrite("p_R", &MC_SIR_Params::p_R);
}