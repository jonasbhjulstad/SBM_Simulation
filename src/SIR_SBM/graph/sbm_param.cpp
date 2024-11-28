#include <SIR_SBM/graph/sbm_param.hpp>
#include <yaml-cpp/yaml.h>
namespace SIR_SBM {
SBM_Param SBM_Param::parse(const std::filesystem::path &fname) {
  YAML::Node config = YAML::LoadFile(fname);
  SBM_Param p;
  p.N_pop = config["SBM"]["N_pop"].as<int>();
  p.N_communities = config["SBM"]["N_communities"].as<int>();
  p.seed = config["seed"].as<int>();
  p.p_in = config["SBM"]["p_in"].as<float>();
  p.p_out = config["SBM"]["p_out"].as<float>();
  return p;
}
} // namespace SIR_SBM