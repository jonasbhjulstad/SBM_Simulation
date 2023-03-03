#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_DEGREE_PROPERTIES_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_DEGREE_PROPERTIES_HPP
#include <Sycl_Graph/Algorithms/Properties/Property_Extractor.hpp>
#include <Sycl_Graph/Buffer/Invariant/Buffer.hpp>
#include <CL/sycl.hpp>
namespace Sycl_Graph::Sycl
{
    enum Degree_Property
    {
        In_Degree,
        Out_Degree
    };

    template <Sycl_Graph::Base::Vertex_Buffer_type _Vertex_Buffer_t, Sycl_Graph::Base::Edge_Buffer_type _Edge_Buffer_t, Degree_Property _Degree_Property>
    struct Degree_Extractor
    {
        static constexpr Degree_Property property = _Degree_Property;
        typedef _Vertex_Buffer_t Vertex_Buffer_t;
        typedef _Edge_Buffer_t Edge_Buffer_t;
        typedef typename Edge_Buffer_t::Edge_t Edge_t;
        typedef typename Vertex_Buffer_type::uI_t uI_t;
        typedef typename Vertex_Buffer_type::ID_t ID_t;
        typedef typename Edge_Buffer_t::Edge_t::From_t From_t;
        typedef typename Edge_Buffer_t::Edge_t::To_t To_t;

        typedef ID_t Property_t;
        typedef sycl::accessor<Property_t, 1, sycl::access::mode::read> Return_Properties_t;
        typedef std::pair<ID_t, uI_t> Accumulation_Property_t;
        typedef sycl::accessor<Accumulation_Property_t, 1, sycl::access::mode::write> Accumulate_Access_t; 

        Property_t apply(const Edge_Target_t& edge_target, const From_t& from, const To_t& to)
        {
            if constexpr (property == In_Degree)
            {
                return edge_target.to;
            }
            else if constexpr (property == Out_Degree)
            {
                return edge_target.from;
            }
        }

        void accumulate(const Return_Properties_t& return_properties, Accumulate_Access_t& accumulated_property)
        {
            for (int i = 0; i < return_properties.size(); i++)
            {
                for(int j = 0; j < accumulated_property.size(); j++)
                {
                    if (return_properties[i] == accumulated_property[j].first)
                    {
                        accumulated_property[j].second++;
                    }
                }
            }
        }
    };
}