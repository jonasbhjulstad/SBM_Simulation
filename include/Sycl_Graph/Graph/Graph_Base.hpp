#ifndef SYCL_GRAPH_GRAPH_HPP
#define SYCL_GRAPH_GRAPH_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/Graph/Graph_Types.hpp>
// #include <Sycl_Graph/Graph/Dynamic/Graph.hpp>
#include <array>
#include <concepts>
namespace Sycl_Graph
{
  template <typename V, typename E, typename DV, typename DE,
            std::unsigned_integral _uI_t = uint32_t,
            std::floating_point _dType = float>
  struct Graph_Base
  {
    Graph_Base(Vertex_Buffer_Base<V, DV, _uI_t> &vertex_buf, Edge_Buffer_Base<E, DE, _uI_t> &edge_buf) : vertex_buf(vertex_buf), edge_buf(edge_buf) {}
    Vertex_Buffer_Base<V, DV, _uI_t> &vertex_buf;
    Edge_Buffer_Base<E, DE, _uI_t> &edge_buf;
    typedef Vertex<V, _uI_t> Vertex_t;
    typedef Edge<E, _uI_t> Edge_t;
    typedef V Vertex_Prop_t;
    typedef E Edge_Prop_t;
    typedef _uI_t uI_t;
    typedef _dType dType;
    uI_t Graph_ID = 0;
    typedef Graph_Base<V, E, DV, DE, uI_t, dType> Graph_t;

    static constexpr auto invalid_id = std::numeric_limits<uI_t>::max();
    uI_t N_vertices() const { return vertex_buf.N_vertices(); }
    uI_t N_edges() const { return edge_buf.N_edges(); }

    void resize(uI_t NV_new, uI_t NE_new)
    {
      vertex_buf.resize(NV_new);
      edge_buf.resize(NE_new);
    }


    template <typename... Args>
    void add_vertex(Args &&...args)
    {
      vertex_buf.add(std::forward<Args>(args)...);
    }

    template <typename... Args>
    void add_edge(Args &&...args)
    {
      edge_buf.add(std::forward<Args>(args)...);
    }

    template <typename... Args>
    void remove_vertex(Args &&...args)
    {
      vertex_buf.remove(std::forward<Args>(args)...);
    }

    template <typename... Args>
    void remove_edge(Args &&...args)
    {
      edge_buf.remove(std::forward<Args>(args)...);
    }

    template <typename... Args>
    void assign_vertex(Args &&...args)
    {
      vertex_buf.assign(std::forward<Args>(args)...);
    }

    template <typename... Args>
    void assign_edge(Args &&...args)
    {
      edge_buf.assign(std::forward<Args>(args)...);
    }

    template <typename... Args>
    V get_vertex(Args &&...args)
    {
      return vertex_buf.get_data(std::forward<Args>(args)...);
    }

    template <typename ... Args>
    const std::vector<uI_t> get_vertex_ids(Args && ... args)
    {
      return vertex_buf.get_valid_ids(std::forward<Args>(args)...);
    }

    template <typename... Args>
    E get_edge(Args &&...args)
    {
      return edge_buf.get_data(std::forward<Args>(args)...);
    }

    template <typename... Args>
    std::vector<V> get_vertex_data(Args &&...args)
    {
      return vertex_buf.get_data(std::forward<Args>(args)...);
    }

    template <typename... Args>
    std::vector<E> get_edge_data(Args &&...args)
    {
      return edge_buf.get_data(std::forward<Args>(args)...);
    }

    // file I/O
    void write_edgelist(std::string filename, std::string delimiter = ",",
                        bool edges_only = true)
    {
      auto edges = edge_buf.get_edges();
      std::ofstream file(filename);
      file << "Graph_ID" << delimiter << "to" << delimiter << "from";
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
        file << Graph_ID << delimiter << e.to << delimiter << e.from;
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
      file << "Graph_ID" << delimiter << "id" << delimiter << "data"
           << "\n";
      write_vertexlist(file, delimiter);
      file.close();
    }

    void write_vertexlist(std::ofstream &file, std::string delimiter = ",")
    {
      auto vertices = vertex_buf.get_vertices();
      for (auto v : vertices)
      {
        file << Graph_ID << delimiter << v.id << delimiter << v.data << "\n";
      }
    }
  };

  // namespace _detail {
  // template <typename...> struct Typelist {};

  // }

  // template <typename uI_t, typename V_LABEL_TYPES, typename E_LABEL_TYPES,
  //           typename DV_LABEL_TYPES, typename DE_LABEL_TYPES>
  // struct Labeled_Graph;

  // template <std::unsigned_integral uI_t, typename V_LABEL_TYPES, typename E_LABEL_TYPES,
  //           typename DV_LABEL_TYPES, typename DE_LABEL_TYPES>
  // struct Labeled_Graph;

  // template <std::unsigned_integral uI_t, typename... Vs, typename... Es, typename... DVs,
  //           typename... DEs>
  // struct Labeled_Graph<uI_t, _detail::Typelist<Vs...>, _detail::Typelist<Es...>,
  //                      _detail::Typelist<DVs...>, _detail::Typelist<DEs...>> {
  //   // get number of types of Vs
  //   static constexpr auto N_LABELS_V = sizeof...(Vs);
  //   // get number of types of Es
  //   static constexpr auto N_LABELS_E = sizeof...(Es);

  //   static constexpr auto V_LABELS =
  //       std::array<const char *, N_LABELS_V>{typeid(Vs).name()...};
  //   static constexpr auto E_LABELS =
  //       std::array<const char *, N_LABELS_E>{typeid(Es).name()...};

  //   // create N_LABELS_V Vertex_Buffers
  //   std::array<Vertex_Buffer_Base<Vs..., uI_t, DVs...>, N_LABELS_V> vertex_bufs;
  //   // create N_LABELS_E Edge_Buffers
  //   std::array<Edge_Buffer_Base<Es..., uI_t, DEs...>, N_LABELS_E> edge_bufs;

  //   using Graph_t = Labeled_Graph<uI_t, _detail::Typelist<Vs...>,
  //                                 _detail::Typelist<Es...>,
  //                                 _detail::Typelist<DVs...>,
  //                                 _detail::Typelist<DEs...>>;

  //   using Vertex_t = boost::variant<Vertex<Vs, uI_t>...>;
  //   using Edge_t = boost::variant<Edge<Es, uI_t>...>;

  //   Labeled_Graph(const std::vector<Vertex<Vs, uI_t>> &...vertices,
  //                 const std::vector<Edge<Es, uI_t>> &...edges)
  //       : vertex_bufs{vertices...}, edge_bufs{edges...} {}

  // };

} // namespace Sycl_Graph
#endif