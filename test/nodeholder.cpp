#include <concepts>
#include <array>
#include <type_traits>
#include <vector>
#include <tuple>
#include <stddef.h>
#include <iostream>
#include <Sycl_Graph/type_helpers.hpp>
template<typename... Ts>
struct VecStruct {
    std::tuple<std::vector<Ts>...> vecs;

    template<typename T>
    requires (std::is_same_v<T, Ts> || ...)
    void add(T data)
    {
        std::get<std::vector<T>>(vecs).push_back(data);
    }

    template<typename T>  
    static constexpr auto type_index()
    {
        return Sycl_Graph::Tuple_Index<T, std::tuple<Ts...>>::value;
    }
};

int main()
{
    VecStruct<int, float, double> vecStruct;
    vecStruct.add((int)1);
    vecStruct.add((float)1.0f);
    vecStruct.add((double)1.0);
    std::cout << vecStruct.type_index<double>() << std::endl;

}