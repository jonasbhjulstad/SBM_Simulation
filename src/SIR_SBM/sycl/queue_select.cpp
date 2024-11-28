#include <SIR_SBM/sycl/queue_select.hpp>
#include <stdexcept>
#include <yaml-cpp/yaml.h>
namespace SIR_SBM {
sycl::queue parse_queue(const std::filesystem::path &fname) {
  YAML::Node config = YAML::LoadFile(fname);
  auto queue_type = config["queue_type"].as<std::string>();
  if (queue_type == "gpu") {
    return sycl::queue(sycl::gpu_selector_v);
  } else if (queue_type == "cpu") {
    return sycl::queue(sycl::cpu_selector_v);
  } else {
    throw std::runtime_error("Unknown queue type: " + queue_type);
  }
}
} // namespace SIR_SBM