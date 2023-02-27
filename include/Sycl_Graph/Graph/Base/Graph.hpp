#ifndef SYCL_GRAPH_GRAPH_HPP
#define SYCL_GRAPH_GRAPH_HPP
#include <concepts>
#include <Sycl_Graph/Graph/Edge_Buffer_Base.hpp>
#include <Sycl_Graph/Graph/Vertex_Buffer_Base.hpp>
namespace Sycl_Graph::Graph::Base
{


  template <Vertex_Buffer_type _Vertex_Buffer_t, 
            Edge_Buffer_type _Edge_Buffer_t>
  struct Graph_Base
  {
    Graph_Base(const _Vertex_Buffer_t&& vertex_buffer, const _Edge_Buffer_t && edge_buffer)
        : vertex_buf(vertex_buffer), edge_buf(edge_buffer)
    {
    }
    typedef _Vertex_Buffer_t Vertex_Buffer_t;
    typedef _Edge_Buffer_t Edge_Buffer_t;

    typedef typename Vertex_Buffer_t::uI_t uI_t;
    typedef typename Vertex_Buffer_t::Vertex_t Vertex_t;
    typedef typename Vertex_Buffer_t::Data_t Vertex_Data_t;
    typedef typename Edge_Buffer_t::Edge_t Edge_t;
    typedef typename Edge_Buffer_t::Data_t Edge_Data_t;

    typedef Graph_Base<Vertex_Buffer_t, Edge_Buffer_t> Graph_t;

    Vertex_Buffer_t vertex_buf;
    Edge_Buffer_t edge_buf;

    uI_t Graph_ID = 0;

    static constexpr auto invalid_id = std::numeric_limits<uI_t>::max();
    uI_t N_vertices() const { return vertex_buf.N_vertices(); }
    uI_t N_edges() const { return edge_buf.N_edges(); }

    void resize(uI_t NV_new, uI_t NE_new)
    {
      vertex_buf.resize(NV_new);
      edge_buf.resize(NE_new);
    }

    auto& operator+(const Graph_t &other) const
    {
      
      auto new_graph = Graph_t(vertex_buf + other.vertex_buf, edge_buf + other.edge_buf);
      return new_graph;
    }

    Graph_t &operator=(Graph_t &other)
    {
      vertex_buf = other.vertex_buf;
      edge_buf = other.edge_buf;
      return *this;
    }


    void add_vertex(const auto&& ... args)
    {
      vertex_buf.add(std::forward<decltype(args)>(args) ...);
    }

    void add_edge(const auto&& ... args)
    {
      edge_buf.add(std::forward<decltype(args)>(args) ...);
    }

    void remove_vertex(const auto&& ... args)
    {
      vertex_buf.remove(std::forward<decltype(args)>(args) ...);
    }

    void remove_edge(const auto&& ... args)
    {
      edge_buf.remove(std::forward<decltype(args)>(args) ...);
    }
    template <typename T>
    auto get_edges(const std::vector<uI_t>&& ids)
    {
      return edge_buf.template get_edges<T>(std::forward<decltype(ids)>(ids));
    }
    template <typename T>
    auto get_edges()
    {
      return edge_buf.template get_edges<T>();
    }
    auto get_edges()
    {
      return edge_buf.get_edges();
    }

    // // file I/O
    // void write_edgelist(std::string filename, std::string delimiter = ",",
    //                     bool edges_only = true)
    // {
    //   auto edges = edge_buf.get_edges();
    //   std::ofstream file(filename);
    //   file << "to" << delimiter << "from";
    //   if (!edges_only)
    //   {
    //     file << delimiter << "data";
    //   }
    //   file << "\n";

    //   write_edgelist(file, delimiter, edges_only);
    //   file.close();
    // }

    // void write_edgelist(std::ofstream &file, std::string delimiter = ",",
    //                     bool edges_only = true)
    // {
    //   auto edges = edge_buf.get_edges();
    //   for (auto e : edges)
    //   {
    //     file << delimiter << e.to << delimiter << e.from;
    //     if (!edges_only)
    //     {
    //       file << delimiter << e.data;
    //     }
    //     file << "\n";
    //   }
    // }

    // void write_vertexlist(std::string filename, std::string delimiter = ",")
    // {
    //   auto vertices = vertex_buf.get_vertices();
    //   std::ofstream file(filename);
    //   file << delimiter << "id" << delimiter << "data"
    //        << "\n";
    //   write_vertexlist(file, delimiter);
    //   file.close();
    // }

    // void write_vertexlist(std::ofstream &file, std::string delimiter = ",")
    // {
    //   auto vertices = vertex_buf.get_vertices();
    //   for (auto v : vertices)
    //   {
    //     file << delimiter << v.id << delimiter << v.data << "\n";
    //   }
    // }
  };




} // namespace Sycl_Graph
#endif