#ifndef FROLS_SYCL_HPP
#define FROLS_SYCL_HPP
#include <CL/sycl.hpp>
namespace FROLS::sycl
{

#ifdef FROLS_USE_INTEL_SYCL
using namespace cl::sycl;
#endif

#ifdef FROLS_USE_HIPSYCL
using namespace hipsycl::sycl;
#endif

}

#endif