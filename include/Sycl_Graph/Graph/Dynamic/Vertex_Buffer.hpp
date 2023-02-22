#ifndef SYCL_GRAPH_META_VERTEX_BUFFER_HPP
#define SYCL_GRAPH_META_VERTEX_BUFFER_HPP
#include <vector>
#include <map>
#include <Sycl_Graph/Graph/Graph_Types.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/visitors.hpp>
#include <boost/graph/breadth_first_search.hpp>
namespace Sycl_Graph::Dynamic
{
    template <typename V, typename E, typename uI_t, boost::directed_tag graph_direction = boost::bidirectional>
    struct Vertex_Buffer : public Vertex_Buffer_Base<V, Vertex_Buffer<V, E, uI_t, graph_direction>, uI_t>
    {
        using namespace boost;
        struct Vertex_tag
        {
            typedef vertex_property_tag kind;
        };

        typedef property<Vertex_tag, V> Vertex_Prop_t;
        typedef adjacency_list<vecS, vecS, graph_direction, Vertex_Prop_t, Edge_Prop_t> Adjlist_t;
        typedef property<vertex_name_type, uI_t> Vertex_Name_t;
        typedef property_map<Adjlist_t, vertex_name_type>::type = ID_Map_t;
        typedef graph_traits<Adjlist_t>::vertex_descriptor Vertex_t;
        // typedef property_map<Adjlist_t, Vertex_Prop_t>::type Map_t;
        typedef std::map<uI_t, Vertex_t> Map_t;

        std::vector<Vertex<V *, uI_t>> vertices;
        Adjlist_t &G;
        Map_t vertex_map;
        ID_Map_t id_map;
        Vertex_Buffer(Adjlist_t &G,
                      const std::vector<Vertex<V *, uI_t>> &vertices) : G(G), vertices(vertices),
                                                                        vertex_map(get(Vertex_tag(), G)),
                                                                        id_map(get(vertex_name_type, G))
        {
        }

        
        template <typename... Args>
        V get_vertex(Args &&...args)
        {
            return vertex_buf.get_data(std::forward<Args>(args)...);
        }
        return edge_buf.get_data(std::forward<Args>(args)...);

        template <typename... Args>
        std::vector<V> get_vertex_data(Args &&...args)
        {
            return vertex_buf.get_data(std::forward<Args>(args)...);
        }

        uI_t size() const
        {
            return vertices.size();
        }

        void add(const std::vector<Vertex<V, uI_t>> &vertices)
        {
            Map_t::iterator pos;
            bool inserted;
            std::for_each(vertices.begin(), vertices.end(), [&](const Vertex<V, uI_t> &v)
                          {
                boost::tie(pos, inserted) = vertex_map.insert(std::make_pair(v.id, v.data));
                if (inserted)
                {
                    auto G_v = boost::add_vertex(v.data, G);

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

        std::vector<Vertex<V, uI_t>> get_vertices()
        {
            std::vector<Vertex<V, uI_t>> vertices;
            vertices.reserve(vertex_map.size());
            for (auto &v : vertex_map)
            {
                vertices.push_back(Vertex<V, uI_t>{v.first, v.second->data});
            }
            return vertices;
        }

        void assign(const std::vector<Vertex<V, uI_t>> &vertices)
        {
            std::for_each(vertices.begin(), vertices.end(), [&](const Vertex<V, uI_t> &v)
                          {
                vertex_map[v.id]->data = v.data;
            });
        }

        void remove(const std::vector<uI_t> &ids)
        {
            std::for_each(ids.begin(), ids.end(), [&](const uI_t &id)
                          {
                auto pos = vertex_map.find(id);
                if (pos != vertex_map.end())
                {
                    boost::remove_vertex(pos->second, G);
                    vertex_map.erase(pos);
                }
            });
        }

        Vertex_Buffer<V, uI_t, Edge_Prop_t>& operator=(const Vertex_Buffer<V, uI_t, Edge_Prop_t> &other) = default;
        
        Vertex_Buffer<V, uI_t, Edge_Prop_t> operator+(const Vertex_Buffer<V, uI_t, Edge_Prop_t> &other) const
        {
            Vertex_Buffer<V, uI_t, Edge_Prop_t> result(*this);
            result.vertex_map.insert(other.vertex_map.begin(), other.vertex_map.end());
            result.G = this->G;
            return result;
        }


    };
}

#endif