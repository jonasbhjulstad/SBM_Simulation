#include <Sycl_Graph/Utils/math.hpp>
#include <numeric>
std::vector<float> make_linspace(float start, float end, float step)
{
    std::vector<float> vec;
    for(float i = start; i < end; i += step)
    {
        vec.push_back(i);
    }
    return vec;
}

std::vector<uint32_t> make_iota(uint32_t N)
{
    std::vector<uint32_t> result(N);
    std::iota(result.begin(), result.end(), 0);
    return result;
};

std::vector<uint32_t> make_iota(uint32_t start, uint32_t end)
{
    std::vector<uint32_t> result(end - start);
    std::iota(result.begin(), result.end(), start);
    return result;
}
