#include <algorithm>
#include <any>
#include <vector>

template <std::unsigned_integral uI_t>
struct CartesianProductSampler
{
    const std::pair<std::vector<uI_t>, std::vector<uI_t>> ranges;
    std::pair<uI_t, uI_t> operator[]
}

template <std::unsigned_integral uI_t>
struct EdgeList_t
{
    std::any<std::vector<uI_t>, >
}
