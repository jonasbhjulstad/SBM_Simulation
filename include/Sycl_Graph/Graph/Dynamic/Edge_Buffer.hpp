#ifndef SYCL_GRAPH_DYNAMIC_EDGE_BUFFER_HPP
#define SYCL_GRAPH_DYNAMIC_EDGE_BUFFER_HPP
#include <vector>
#include <map>
#include <Sycl_Graph/Graph/Graph_Types.hpp>
#include <Sycl_Graph/Graph/Dynamic/Graph_Types.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/visitors.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/properties.hpp>
namespace Sycl_Graph::Dynamic
{
    template <typename E, typename uI_t, boost_graph Adjlist_t>
    struct Edge_Buffer : public Edge_Buffer_Base<E, uI_t, Edge_Buffer<E, uI_t, Adjlist_t>>
    {

        typedef boost::property_map<Adjlist_t, typename Adjlist_t::edge_property_type>::type = ID_Map_t;
        // typedef Adjlist::edge_descriptor::type Edge_t;
        // typedef property_map<Adjlist_t, Edge_Prop_t>::type Map_t;
        typedef std::map<uI_t, E> Map_t;
        Adjlist_t &G;
        Map_t edge_map;
        ID_Map_t id_map;
        Edge_Buffer(Adjlist_t &G,
                      const std::vector<Edge<E, uI_t>> &edges) : G(G), edges(edges),
                                                                        edge_map(boost::get(Edge_tag(), G)),
                                                                        id_map(boost::get(Edge_name_type, G))
        {
        }

        
        template <typename... Args>
        V get_Edge(Args &&...args)
        {
            return Edge_buf.get_data(std::forward<Args>(args)...);
        }
        return edge_buf.get_data(std::forward<Args>(args)...);

        template <typename... Args>
        std::vector<V> get_Edge_data(Args &&...args)
        {
            return Edge_buf.get_data(std::forward<Args>(args)...);
        }

        uI_t size() const
        {
            return edges.size();
        }

        void add(const std::vector<Edge<V, uI_t>> &edges)
        {
            Map_t::iterator pos;
            bool inserted;
            std::for_each(edges.begin(), edges.end(), [&](const Edge<V, uI_t> &v)
                          {
                boost::tie(pos, inserted) = edge_map.insert(std::make_pair(v.id, v.data));
                if (inserted)
                {
                    auto G_v = boost::add_Edge(v.data, G);

                    pos->second = G_v;
                }
                else
                {
                    pos->second->data = v.data;
                } 
                });
        }

        std::vector<V> get_data(const std::vector<uI_t> &ids)
        {
            std::vector<V> data;
            data.reserve(ids.size());
            for (auto id : ids)
            {
                data.push_back(get_data(id));
            }
            return data;
        }

        std::vector<Edge<V, uI_t>> get_edges()
        {
            std::vector<Edge<V, uI_t>> edges;
            edges.reserve(edge_map.size());
            for (auto &v : edge_map)
            {
                edges.push_back(Edge<V, uI_t>{v.first, v.second->data});
            }
            return edges;
        }

        void assign(const std::vector<Edge<V, uI_t>> &edges)
        {
            std::for_each(edges.begin(), edges.end(), [&](const Edge<V, uI_t> &v)
                          {
                edge_map[v.id]->data = v.data;
            });
        }

        void remove(const std::vector<uI_t> &ids)
        {
            std::for_each(ids.begin(), ids.end(), [&](const uI_t &id)
                          {
                auto pos = edge_map.find(id);
                if (pos != edge_map.end())
                {
                    boost::remove_Edge(pos->second, G);
                    edge_map.erase(pos);
                }
            });
        }

        Edge_Buffer<V, uI_t, Edge_Prop_t>& operator=(const Edge_Buffer<V, uI_t, Edge_Prop_t> &other) = default;
        
        Edge_Buffer<V, uI_t, Edge_Prop_t> operator+(const Edge_Buffer<V, uI_t, Edge_Prop_t> &other) const
        {
            Edge_Buffer<V, uI_t, Edge_Prop_t> result(*this);
            result.edge_map.insert(other.edge_map.begin(), other.edge_map.end());
            result.G = this->G;
            return result;
        }


    };
}

#endif