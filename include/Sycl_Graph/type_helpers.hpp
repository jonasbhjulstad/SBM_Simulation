#ifndef SYCL_GRAPH_TYPE_HELPERS_HPP
#define SYCL_GRAPH_TYPE_HELPERS_HPP
#include <type_traits>
#include <tuple>
#include <array>

namespace Sycl_Graph
{
    template <class T, class Tuple>
    struct Tuple_Index;

    template <class T, class... Types>
    struct Tuple_Index<T, std::tuple<T, Types...>> {
        static const std::size_t value = 0;
    };

    template <class T, class U, class... Types>
    struct Tuple_Index<T, std::tuple<U, Types...>> {
        static const std::size_t value = 1 + Tuple_Index<T, std::tuple<Types...>>::value;
    };

    template <typename T, typename... Types>
    struct index_of_type;

    template <typename T, typename... Rest>
    struct index_of_type<T, T, Rest...> : std::integral_constant<std::size_t, 0> {};

    template <typename T, typename First, typename... Rest>
    struct index_of_type<T, First, Rest...>
        : std::integral_constant<std::size_t, 1 + index_of_type<T, Rest...>::value> {};

}
#endif