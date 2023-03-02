#ifndef SYCL_GRAPH_TYPE_HELPERS_HPP
#define SYCL_GRAPH_TYPE_HELPERS_HPP
#include <type_traits>
#include <tuple>
#include <array>

namespace Sycl_Graph::Type_Helpers
{
    template <class T, class Tuple>
    struct Tuple_Index;

    template <class T, class... Types>
    struct Tuple_Index<T, std::tuple<T, Types...>>
    {
        static const std::size_t value = 0;
    };

    template <class T, class U, class... Types>
    struct Tuple_Index<T, std::tuple<U, Types...>>
    {
        static const std::size_t value = 1 + Tuple_Index<T, std::tuple<Types...>>::value;
    };

    // Get the index of a type in a parameter pack
    template <typename T, typename... Types>
    struct index_of_type;

    template <typename T, typename... Rest>
    struct index_of_type<T, T, Rest...> : std::integral_constant<std::size_t, 0>
    {
    };

    template <typename T, typename First, typename... Rest>
    struct index_of_type<T, First, Rest...>
        : std::integral_constant<std::size_t, 1 + index_of_type<T, Rest...>::value>
    {
    };

    template <typename... Ts, typename... Types>
    constexpr auto get_by_types(const std::tuple<Types...> &tuple)
    {
        return std::make_tuple(std::get<index_of_type<Ts, Types...>::value>(tuple)...);
    }

    template <typename...> struct list;

    template <typename, typename> struct list_append_impl;

    template <typename... Ts, typename... Us>
    struct list_append_impl<list<Ts...>, list<Us...>> {
        using type = list<Ts..., Us...>;
    };

    template <template <typename> class, typename...>
    struct filter_impl;

    template <template <typename> class Predicate>
    struct filter_impl<Predicate> {
        using type = list<>;
    };

    template <template <typename> class Predicate, typename T, typename... Rest>
    struct filter_impl<Predicate, T, Rest...> {
        using type = typename list_append_impl<
                        std::conditional_t<
                            Predicate<T>::value,
                            list<T>,
                            list<>
                        >,
                        typename filter_impl<Predicate, Rest...>::type
                    >::type;
    };

    template <template <typename> class Predicate, typename... Ts>
    using filter = typename filter_impl<Predicate, Ts...>::type;


    template <typename T, typename... Ts>
    struct is_in_pack
    {
        static constexpr bool value = std::disjunction_v<std::is_same<T, Ts>...>;
    };



    


    // //Get the Nth base type of a class

    // template <typename T, std::size_t N>
    // struct get_base;

    // template <typename T, std::size_t N>
    // struct get_base
    // {
    //     using type = typename get_base<typename T::Base_t, N-1>::type;
    // };

    // template <typename T>
    // struct get_base<typename T::Base_t, 0>
    // {
    //     using type = typename T::Base_t;
    // };

    // template <typename T, std::size_t N>
    // using get_base_t = typename get_base<T, N>::type;

    // Get the uppermost base type of a class
    //  template <typename T> requires std::is_same_v<T, typename T::Base_t>
    //  struct get_uppermost_base {
    //      using type = T;
    //  };

    // template <typename T> requires (!std::is_same_v<T, typename T::Base_t>)
    // struct get_uppermost_base<void>
    // {
    //     using type = typename get_uppermost_base<typename T::Base_t>::type;
    // };

    // template <typename T>
    // using get_uppermost_base_t = typename get_uppermost_base<T>::type;

}
#endif