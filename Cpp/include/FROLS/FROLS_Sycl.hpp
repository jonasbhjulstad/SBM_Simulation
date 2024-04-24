#ifndef FROLS_SYCL_HPP
#define FROLS_SYCL_HPP
#include <CL/sycl.hpp>
namespace FROLS
{

#ifdef FROLS_USE_INTEL_SYCL
using namespace cl;
#endif

#ifdef FROLS_USE_HIPSYCL
using namespace hipsycl;
#endif

}

#endif