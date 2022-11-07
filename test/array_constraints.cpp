#include <Data_Containers.hpp>
#include <array>
#include <vector>
using namespace containers;
template <template <typename ...> typename Array_t, typename ... Data_t>
requires DynamicArray<Array_t<Data_t ...>>
void resizable_fn(Array_t<Data_t ...> a)
{
    a.resize(10);
}

template <template <typename, size_t> typename Array_t, typename Data_t, std::size_t N>
requires FixedArray<Array_t<Data_t, N>>
void fixed_fn(Array_t<Data_t, N> a)
{
    auto elem = a[0];
}

int main()
{
    std::vector<int> v(1);
    resizable_fn(v);
    std::array<int, 10> a;
    fixed_fn(a);
}