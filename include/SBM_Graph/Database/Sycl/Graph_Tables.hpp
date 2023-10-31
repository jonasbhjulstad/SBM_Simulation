#ifndef SBM_DATABASE_SYCL_BUFFER_INTERFACE_HPP
#define SBM_DATABASE_SYCL_BUFFER_INTERFACE_HPP

#include <CL/sycl.hpp>
#include <orm/db.hpp>
#include <Sycl_Buffer_Routines/Buffer_Routines.hpp>
#include <SBM_Graph/Database/Graph_Tables.hpp>
namespace SBM_Graph{
namespace Sycl {

std::shared_ptr<sycl::buffer<std::pair<uint32_t, uint32_t>, 1>> read_edgelist(sycl::queue& q, uint32_t p_out, uint32_t graph, sycl::event& res_event);

std::shared_ptr<sycl::buffer<uint32_t, 1>> read_ecm(sycl::queue& q, uint32_t p_out, uint32_t graph, sycl::event& res_event);

std::shared_ptr<sycl::buffer<uint32_t, 1>> read_vcm(sycl::queue& q, uint32_t p_out, uint32_t graph, sycl::event& res_event);

}

}  // namespace SBM_Graph

#endif
