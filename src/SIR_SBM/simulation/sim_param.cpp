#include <SIR_SBM/simulation/sim_param.hpp>
#include <yaml-cpp/yaml.h>
namespace SIR_SBM {
Sim_Param Sim_Param::parse(const char *fname) {
  YAML::Node config = YAML::LoadFile(fname);
  Sim_Param p;
  p.p_I0 = config["p_I0"].as<float>();
  p.p_I = config["p_I"].as<float>();
  p.p_R = config["p_R"].as<float>();
  p.Nt = config["Nt"].as<uint32_t>();
  p.N_sims = config["N_sims"].as<uint32_t>();
  p.seed = config["seed"].as<uint32_t>();
  return p;
}
} // namespace SIR_SBM