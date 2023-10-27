#ifndef SBM_DATABASE_SYCL_BUFFER_INTERFACE_HPP
#define SBM_DATABASE_SYCL_BUFFER_INTERFACE_HPP

#include <CL/sycl.hpp>
#include <orm/db.hpp>
#include <Sycl_Buffer_Routines/Buffer_Routines.hpp>
#include <SBM_Graph/Database/Graph_Tables.hpp>
namespace SBM_Graph{
namespace Sycl {

std::shared_ptr<sycl::buffer<std::pair<uint32_t, uint32_t>, 1>> read_edgelist(sycl::queue& q, uint32_t p_out, uint32_t graph, sycl::event& res_event)
{
    // std::shared_ptr<sycl::buffer<T, N>> make_shared_device_buffer(sycl::queue &q, const std::vector<T> &vec, sycl::range<N> range, sycl::event &res_event)
    auto edge_list = SBM_Graph::read_edgelist(p_out, graph);
    auto buf = Buffer_Routines::make_shared_device_buffer<std::pair<uint32_t, uint32_t>, 1>(q, edge_list, sycl::range<1>(edge_list.size()), res_event);
    return buf;
}

std::shared_ptr<sycl::buffer<uint32_t, 1>> read_ecm(sycl::queue& q, uint32_t p_out, uint32_t graph, sycl::event& res_event)
{
    // std::shared_ptr<sycl::buffer<T, N>> make_shared_device_buffer(sycl::queue &q, const std::vector<T> &vec, sycl::range<N> range, sycl::event &res_event)
    auto ecm = SBM_Graph::read_ecm(p_out, graph);
    auto buf = Buffer_Routines::make_shared_device_buffer<uint32_t,  1>(q, ecm, sycl::range<1>(ecm.size()), res_event);
    return buf;
}

std::shared_ptr<sycl::buffer<uint32_t, 1>> read_vcm(sycl::queue& q, uint32_t p_out, uint32_t graph, sycl::event& res_event)
{
    // std::shared_ptr<sycl::buffer<T, N>> make_shared_device_buffer(sycl::queue &q, const std::vector<T> &vec, sycl::range<N> range, sycl::event &res_event)
    auto vcm = SBM_Graph::read_vcm(p_out, graph);
    auto buf = Buffer_Routines::make_shared_device_buffer<uint32_t,  1>(q, vcm, sycl::range<1>(vcm.size()), res_event);
    return buf;
}

}

}  // namespace SBM_Graph

#endif
