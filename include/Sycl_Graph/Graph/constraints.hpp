#ifndef SYCL_GRAPH_GRAPH_CONSTRAINTS_HPP
#define SYCL_GRAPH_GRAPH_CONSTRAINTS_HPP
#include <concepts>
#include <vector>
namespace Sycl_Graph
{
    template <typename T>
    concept Vertex_type = requires(T) {
                              typename T::ID_t;
                              typename T::Data_t;
                          };

    template <typename T>
    concept Edge_type = requires(T) {
                            typename T::ID_t;
                            typename T::Data_t;
                        };

    template <typename T>
    concept Graph_type = true;

    template <typename T>
    concept Vertex_Buffer_type = requires(T buf) {
                                     typename T::uI_t;
                                     typename T::Vertex_t;
                                     buf.size();
                                     buf.add(std::vector<typename T::uI_t>{}, std::vector<typename T::Vertex_t>{});
                                     buf.add(std::vector<typename T::uI_t>{});
                                     buf.get_vertices();
                                     buf.remove(0);
                                 };

    template <typename T>
    concept Edge_Buffer_type = requires(T buf) {
                                   typename T::uI_t;
                                   typename T::Edge_t;
                                   buf.size();
                                   buf.add(std::vector<typename T::uI_t>{}, std::vector<typename T::uI_t>{}, std::vector<typename T::Edge_t>{});
                                   buf.add(std::vector<typename T::uI_t>{}, std::vector<typename T::uI_t>{});
                                   buf.get_edges();
                                   buf.remove(0);
                               };
}

#endif