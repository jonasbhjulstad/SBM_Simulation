#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_PROPERTY_EXTRACTOR_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_PROPERTY_EXTRACTOR_HPP
#include <tuple>
#include <Sycl_Graph/Algorithms/Properties/Property_Extractor.hpp>
#include <metal.hpp>
namespace Sycl_Graph::Sycl
{
    template <typename T>
    concept Property_Extractor_type = Sycl_Graph::Base::Property_Extractor_type<T>;



    template <Sycl_Graph::Invariant::Graph_type Graph_t, Property_Extractor_type ... Es>
    std::tuple<typename Es::Accumulation_Property_t...> extract_properties(Graph_t& graph)
    {
        
    }
}