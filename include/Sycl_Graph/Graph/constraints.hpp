#ifndef SYCL_GRAPH_GRAPH_CONSTRAINTS_HPP
#define SYCL_GRAPH_GRAPH_CONSTRAINTS_HPP
#include <concepts>

namespace Sycl_Graph
{
    concept Vertex_type = requires
    {
        typename Vertex::ID_t;
        typename Vertex::Data_t;
    };

    concept Edge_type = requires
    {
        typename Edge::ID_t;
        typename Edge::Data_t;
    };

    template <typename T>
    concept Vertex_Buffer_type = requires
    {
        typename T::Vertex_t;
        typename T::ID_t;
        typename T::size();
        typename T::add();
        typename T::get_vertices();
        typename T::remove();
    };

    template <typename T>
    concept Edge_Buffer_type = requires
    {
        typename T::Edge_t;
        typename T::ID_t;
        typename T::size();
        typename T::add();
        typename T::get_edges();
        typename T::remove();
    };

    template <typename T>
    concept Graph_type = requires
    {
        typename T::Vertex_t;
        typename T::Edge_t;
        typename T::ID_t;
        typename T::size();
        typename T::add();
        typename T::get_vertices();
        typename T::get_edges();
        typename T::remove();
    };

}


#endif