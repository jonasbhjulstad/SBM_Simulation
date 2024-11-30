#pragma once

#include <filesystem>
#include <sycl/sycl.hpp>
namespace SBM {

sycl::queue parse_queue(const std::filesystem::path &);

} // namespace SBM