#ifndef SYCL_GRAPH_GRAPH_FIXED_HPP
#define SYCL_GRAPH_GRAPH_FIXED_HPP
namespace Sycl_Graph::Fixed
{
// namespace Fixed
// {
//   template <std::unsigned_integral uI_t, uI_t NV, uI_t NE, typename V, template <typename, uI_t> typename Array_t>
//   using Scatter_Fn = Array_t<V, NE> (*)(const Array_t<V, NV> &);

//   template <typename V, typename E, std::unsigned_integral uI_t, uI_t NV, uI_t NE,
//             template <typename, uI_t> typename Array_t>
//   struct HostGraphContainer
//       : public GraphContainerBase<GraphContainer<V, E, uI_t, NV, NE, Array_t>,
//                                   uI_t, V, E>
//   {
//     using Base = GraphContainerBase<GraphContainer<V, E, uI_t, NV, NE, Array_t>,
//                                     uI_t, V, E>;
//     Array_t<Vertex<V, uI_t>, NV> vertices;
//     Array_t<Edge<E, uI_t>, NE> edges;
//     uI_t &N_vertices = Base::N_vertices;
//     uI_t &N_edges = Base::N_edges;
//     auto begin() { return std::begin(vertices); }
//     auto end() { return std::begin(vertices) + NV; }

//     void assign(uI_t id, const V &v_data)
//     {
//       auto v = std::find_if(std::begin(vertices), std::end(vertices),
//                             [id](const auto &v)
//                             { return v.id == id; });
//       assert(v != std::end(vertices) && "Index not found");
//       v->data = v_data;
//     }
//   };

//   template <typename V, typename E, std::unsigned_integral uI_t, uI_t NV, uI_t NE,
//             std::unsigned_integral uA_t,
//             template <typename, uA_t> typename Array_t>
//   struct Graph : public GraphContainer<V, E, uI_t, NV, NE, Array_t>
//   {
//     static constexpr uI_t NV_MAX = NV;
//     static constexpr uI_t NE_MAX = NE;

//     using Vertex_t = Vertex<V, uI_t>;
//     using Edge_t = Edge<E, uI_t>;
//     using Vertex_Prop_t = V;
//     using Edge_Prop_t = E;
//     using Container_t = GraphContainer<V, E, uI_t, NV, NE, Array_t>;
//     Container_t C;
//     uI_t N_vertices = 0;
//     uI_t N_edges = 0;

//     const Vertex_Prop_t &operator[](uI_t id) const { return get_vertex_prop(id); }

//     const Vertex_Prop_t &get_vertex_prop(uI_t id) const
//     {
//       return get_vertex(id)->data;
//     }

//     const Array_t<Vertex_t *, NV + 1> neighbors(uI_t idx) const
//     {
//       Array_t<Vertex_t *, NV + 1> neighbors = {};
//       std::for_each(C._edges.begin(), C._edges.end(),
//                     [&, N = 0](const auto e) mutable
//                     {
//                       if (is_in_edge(e, idx))
//                       {
//                         const auto p_nv = this->get_neighbor(e, idx);
//                         if (p_nv != nullptr)
//                         {
//                           neighbors[N] = p_nv;
//                           N++;
//                         }
//                       }
//                     });
//       return neighbors;
//     }
//   };
// } // namespace Fixed
}

#endif // SYCL_GRAPH_GRAPH_FIXED_HPP