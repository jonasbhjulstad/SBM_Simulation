#ifndef SYCL_GRAPH_GRAPH_HPP
#define SYCL_GRAPH_GRAPH_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/Graph/Graph_Types.hpp>
#include <Sycl_Graph/Graph/Vertex_Buffer.hpp>
#include <Sycl_Graph/Graph/Edge_Buffer.hpp>
#include <array>
#include <concepts>
namespace Sycl_Graph
{


  template <Invariant_Vertex_Buffer_type vertex_buffer, 
            Invariant_Edge_Buffer_type edge_buffer>
  struct Invariant_Graph_Base
  {
    template <Vertex_Buffer_type ... VBs, Edge_Buffer_type ... EBs>
    Invariant_Graph_Base(VBs &&... vertex_buffers, EBs &&... edge_buffers)
        : vertex_buf(vertex_buffers ...), edge_buf(edge_buffers ...)
    {
    }
    typedef Invariant_Edge_Buffer IEB;
    typedef Invariant_Vertex_Buffer IVB;

    typedef IVB::uI_t uI_t;
    typedef IVB::Vertex_t Vertex_t;
    typedef IVB::Data_t Vertex_Data_t;
    typedef IEB::Edge_t Edge_t;
    typedef IEB::Data_t Edge_Data_t;

    IVB vertex_buf;
    IEB edge_buf;

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
      return edge_buf.get_edges<T>(std::forward<decltype(ids)>(ids));
    }
    template <typename T>
    auto get_edges()
    {
      return edge_buf.get_edges<T>();
    }
    auto get_edges()
    {
      return edge_buf.get_edges();
    }

    // file I/O
    void write_edgelist(std::string filename, std::string delimiter = ",",
                        bool edges_only = true)
    {
      auto edges = edge_buf.get_edges();
      std::ofstream file(filename);
      file << "to" << delimiter << "from";
      if (!edges_only)
      {
        file << delimiter << "data";
      }
      file << "\n";

      write_edgelist(file, delimiter, edges_only);
      file.close();
    }

    void write_edgelist(std::ofstream &file, std::string delimiter = ",",
                        bool edges_only = true)
    {
      auto edges = edge_buf.get_edges();
      for (auto e : edges)
      {
        file << delimiter << e.to << delimiter << e.from;
        if (!edges_only)
        {
          file << delimiter << e.data;
        }
        file << "\n";
      }
    }

    void write_vertexlist(std::string filename, std::string delimiter = ",")
    {
      auto vertices = vertex_buf.get_vertices();
      std::ofstream file(filename);
      file << delimiter << "id" << delimiter << "data"
           << "\n";
      write_vertexlist(file, delimiter);
      file.close();
    }

    void write_vertexlist(std::ofstream &file, std::string delimiter = ",")
    {
      auto vertices = vertex_buf.get_vertices();
      for (auto v : vertices)
      {
        file << delimiter << v.id << delimiter << v.data << "\n";
      }
    }
  };




} // namespace Sycl_Graph
#endif