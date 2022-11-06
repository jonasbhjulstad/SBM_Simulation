#include <Data_Containers.hpp>
#include <array>
#include <vector>
using namespace containers;
template <template <typename ...> typename Array_t, typename D>
requires DynamicArray<Array_t<D>>
void resizable_fn(Array_t<D> a)
{
    a.resize(10);
}

template <template <typename ...> typename Array_t, typename D>
requires FixedArray<Array_t<D>>
void fixed_fn(Array_t<D> a)
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