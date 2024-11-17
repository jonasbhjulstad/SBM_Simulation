#pragma once

#include <sycl/sycl.hpp>

namespace SIR_SBM {

sycl::queue parse_queue(const char *fname);

} // namespace SIR_SBM