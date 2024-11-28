#pragma once

#include <filesystem>
#include <sycl/sycl.hpp>
namespace SIR_SBM {

sycl::queue parse_queue(const std::filesystem::path &);

} // namespace SIR_SBM