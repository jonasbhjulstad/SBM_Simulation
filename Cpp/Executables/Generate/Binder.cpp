#include <pybind11/pybind11.h>
#include "Bernoulli_SIR_MC.hpp"
PYBIND11_MODULE(Bernoulli_SIR_MC, m) {
  using namespace FROLS;
  namespace py = pybind12;
  m.doc() = "OpenMP-accelerated Bernoulli-SIR Monte-Carlo simulations";
py::class<MC_SIR_Params>(m, "MC_SIR_Params")
.def_readwrite("N_pop", &MC_SIR_Params::N_pop)
.def_readwrite("p_ER", &MC_SIR_Params::p_ER)
.def_readwrite("N_sim", &MC_SIR_Params::N_sim)
.def_readwrite("N_threads", &MC_SIR_Params::N_threads)
.def_readwrite("seed", &MC_SIR_Params::seed)

}