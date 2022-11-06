#include <Data_Containers.hpp>
#include <array>
#include <vector>
using namespace containers;
template <ResizableArray A>
void resizable_fn(A a)
{
    a.resize(10);
}

template <FixedArray A>
void fixed_fn(A a)
{
    a[0];
}

int main()
{
    std::vector<int> v;
    resizable_fn(v);
    std::array<int, 10> a;
    fixed_fn(a);
}