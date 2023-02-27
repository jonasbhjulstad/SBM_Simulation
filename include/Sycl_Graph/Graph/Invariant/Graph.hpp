#ifndef SYCL_GRAPH_GRAPH_HPP
#define SYCL_GRAPH_GRAPH_HPP
#include <CL/sycl.hpp>

#include <array>
#include <concepts>
namespace Sycl_Graph::Graph::Invariant
{


  template <Vertex_Buffer_type vertex_buffer, 
            Edge_Buffer_type edge_buffer>
  struct Graph: public Sycl_Graph::Graph::Base<Vertex_Buffer_type, Edge_Buffer_type>
  {
    template <Vertex_Buffer_type ... VBs, Edge_Buffer_type ... EBs>
    Graph_Base(VBs &&... vertex_buffers, EBs &&... edge_buffers)
        : vertex_buf(vertex_buffers ...), edge_buf(edge_buffers ...)
    {
    }
    typedef decltype(vertex_buffer) Vertex_Buffer_t;
    typedef decltype(edge_buffer) Edge_Buffer_t;

    typedef Vertex_Buffer_t::uI_t uI_t;
    typedef Vertex_Buffer_t::Vertex_t Vertex_t;
    typedef Vertex_Buffer_t::Data_t Vertex_Data_t;
    typedef Edge_Buffer_t::Edge_t Edge_t;
    typedef Edge_Buffer_t::Data_t Edge_Data_t;

    uI_t& Graph_ID = this->Graph_ID;

    static constexpr auto invalid_id = std::numeric_limits<uI_t>::max();

  };

  template <typename T>
  concept Graph_type = 
  Sycl_Graph::Invariant::Vertex_Buffer_type<Vertex_Buffer_t> &&
  Sycl_Graph::Invariant::Edge_Buffer_type<Edge_Buffer_t> &&
  requires(T a)
  {
  };

} // namespace Sycl_Graph
#endif