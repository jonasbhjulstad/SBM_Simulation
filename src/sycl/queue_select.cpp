#include <SIR_SBM/sycl/queue_select.hpp>
#include <yaml-cpp/yaml.h>
namespace SIR_SBM {
sycl::queue parse_queue(const char *fname) {
  YAML::Node config = YAML::LoadFile(fname);
  auto queue_type = config["queue_type"].as<std::string>();
  if (queue_type == "gpu") {
    return sycl::gpu_selector();
  } else if (queue_type == "cpu") {
    return sycl::cpu_selector();
  } else {
    throw std::runtime_error("Unknown queue type: " + queue_type);
  }
}
} // namespace SIR_SBM