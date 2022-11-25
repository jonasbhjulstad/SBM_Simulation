#ifndef SYCL_GRAPH_SYCL_HPP
#define SYCL_GRAPH_SYCL_HPP
#include <CL/sycl.hpp>
namespace Sycl_Graph
{

#ifdef SYCL_GRAPH_USE_INTEL_SYCL
using namespace cl;
#endif

#ifdef SYCL_GRAPH_USE_HIPSYCL
using namespace hipsycl;
#endif

}

#endif