#ifndef GRAPH_SYCL_GRAPH_DYNAMIC_IMPL_HPP
#define GRAPH_SYCL_GRAPH_DYNAMIC_IMPL_HPP
#include "Graph_Types.hpp"
namespace Sycl_Graph::Dynamic
{
    // template <typename V, typename E, std::unsigned_integral uI_t>
    using Sycl_Graph::GraphContainerBase;
    // template <typename D, std::unsigned_integral ID_t>
    using Sycl_Graph::Vertex;
    // template <typename D, std::unsigned_integral ID_t>
    using Sycl_Graph::Edge;

    template <typename V, typename E, std::unsigned_integral uI_t,
              template <typename> typename Array_t>
    struct GraphContainer : public GraphContainerBase<GraphContainer<V, E, uI_t, Array_t>, V, E, uI_t>
    {
        GraphContainer(uI_t NV_max, uI_t NE_max)
            : NV_max(NV_max), NE_max(NE_max), vertices(NV_max), edges(NE_max) {}
        using Base = GraphContainerBase<GraphContainer<V, E, uI_t, Array_t>, V, E, uI_t>;

        using Vertex_t = typename Base::Vertex_t;
        using Edge_t = typename Base::Edge_t;

        Array_t<Vertex_t> vertices;
        Array_t<Edge_t> edges;
        uI_t &N_vertices = Base::N_vertices;
        uI_t &N_edges = Base::N_edges;
        auto begin() { return std::begin(vertices); }
        auto end() { return std::begin(vertices) + N_vertices; }
        uI_t NV_max, NE_max;

        uI_t get_max_vertices()
        {
            return NV_max;
        }

        uI_t get_max_edges()
        {
            return NE_max;
        }

        std::vector<V> vertex_prop(const std::vector<uI_t> &ids)
        {
            std::vector<V> res;
            std::transform(ids.begin(), ids.end(), res.begin(), [&](auto id)
                           {
                auto it = std::find_if(vertices.begin(), vertices.end(),
                [id](const auto &v) { return v.id == id; });
                return it->data; });
            return res;
        }

        std::vector<typename std::vector<V>::iterator> find(const std::vector<uI_t> &ids) const
        {
            std::vector<typename std::vector<V>::iterator> res;
            std::transform(ids.begin(), ids.end(), res.begin(), [&](auto id)
                           {
                auto it = std::find_if(vertices.begin(), vertices.end(),
                [id](const auto &v) { return v.id == id; });
                return it; });
            return res;
        }

        bool add(const std::vector<Vertex_t>& edges)
        {
            if (N_vertices + edges.size() > NV_max)
            {
                return false;
            }
            std::copy(edges.begin(), edges.end(), vertices.begin() + N_vertices);
            return true;
        }


        bool add(const std::vector<uI_t> &id, const std::vector<V> &v_data)
        {
            if (N_vertices == NV_max)
                return false;
            for (int i = 0; i < id.size(); i++)
            {
                vertices[N_vertices] = Vertex_t{id[i], v_data[i]};
                N_vertices++;
            }
            return true;
        }

        bool add(const std::vector<uI_t> &from, const std::vector<uI_t> &to, const std::vector<E> &e_data = std::vector<E>())
        {

            if (N_edges == NE_max)
                return false;
            for (int i = 0; i < from.size(); i++)
            {
                edges[N_edges] = Edge_t{from[i], to[i], e_data[i]};
                N_edges++;
            }
            return true;
        }

        void assign(const std::vector<uI_t> &id, const std::vector<V> &v_data)
        {
            // find vertex with id
            auto it = std::find_if(vertices.begin(), vertices.end(),
                                   [id](const auto &v)
                                   { return v.id == id; });
            if (it != vertices.end())
            {
                it->data = v_data;
            }
        }

        void remove(const std::vector<uI_t> &id)
        {
            // find vertex with id
            auto it = std::find_if(vertices.begin(), vertices.end(),
                                   [id](const auto &v)
                                   { return v.id == id; });
            if (it != vertices.end())
            {
                it->id = Vertex_t::invalid_id;
                N_vertices--;
            }
        }

        void sort_vertices()
        {
            std::sort(vertices.begin(), vertices.end(),
                      [](const auto &a, const auto &b)
                      { return a.id < b.id; });
        }

        void remove(const std::vector<uI_t> &to, const std::vector<uI_t> &from)
        {
            for (int i = 0; i < to.size(); i++)
            {
                // find edge with id
                auto it = std::find_if(edges.begin(), edges.end(),
                                       [&](const auto &e)
                                       { return (e.to == to[i]) && (e.from == from[i]); });
                if (it != edges.end())
                {
                    it->id = Edge_t::invalid_id;
                    N_edges--;
                }
            }
        }
    };

    template <typename V, typename E, std::unsigned_integral uI_t,
              template <typename> typename Array_t>
    struct Graph
    {
        Graph() = default;
        Graph(uI_t NV_max, uI_t NE_max)
            : NV_max(NV_max), NE_max(NE_max), C(NV_max, NE_max) {}
        Graph(const std::vector<Vertex<V, uI_t>> &vertices,
              const std::vector<Edge<E, uI_t>> &edges) : C(vertices, edges) {}

        using Container_t = GraphContainer<V, E, uI_t, Array_t>;
        using Vertex_t = typename Container_t::Vertex_t;
        using Edge_t = Edge<E, uI_t>;
        using Vertex_Prop_t = V;
        using Edge_Prop_t = E;
        Container_t C;
        uI_t &N_vertices = C.N_vertices;
        uI_t &N_edges = C.N_edges;
        uI_t NV_max, NE_max;

        const V &operator[](uI_t id) const { return get_vertex_prop(id); }

        const Array_t<const Vertex_t *> neighbors(uI_t idx) const
        {
            Array_t<const Vertex_t *> neighbors(N_vertices);
            std::for_each(C.edges.begin(), C.edges.end(), [&, N = 0](const auto e) mutable
                          {
      if (is_in_edge(e, idx)) {
        const auto p_nv = get_neighbor(e, idx);
        if (p_nv != nullptr) {
          neighbors[N] = p_nv;
          N++;
        }
      } });
            return neighbors;
        }

        // Forward declaration of container methods
        template <typename... Args>
        auto add(Args &...args)
        {
            return C.add(args...);
        }

        template <typename... Args>
        auto assign(Args &...args)
        {
            return C.assign(args...);
        }

        template <typename... Args>
        auto remove(Args &...args)
        {
            return C.remove(args...);
        }
        void sort_vertices()
        {
            C.sort_vertices();
        }
        auto begin()
        {
            return C.vertices.begin();
        }
        auto end()
        {
            return C.vertices.end();
        }

        auto get_neighbor(const Edge_t &e, uI_t idx) const
        {
            if (e.to == idx)
            {
                return &C.vertices[e.from];
            }
            else if (e.from == idx)
            {
                return &C.vertices[e.to];
            }
            return (uI_t)Vertex_t::invalid_id;
        }
    };
} // namespace Dynamic

#endif // GRAPH_SYCL_GRAPH_DYNAMIC_IMPL_HPP