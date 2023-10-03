#include <Sycl_Graph/Utils/math.hpp>
std::vector<float> make_linspace(float start, float end, float step)
{
    std::vector<float> vec;
    for(auto i = start; i < end; i += step)
    {
        vec.push_back(i);
    }
    return vec;
}
