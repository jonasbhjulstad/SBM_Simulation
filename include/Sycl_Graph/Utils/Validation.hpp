#ifndef SYCL_GRAPH_UTILS_VALIDATION__HPP
#define SYCL_GRAPH_UTILS_VALIDATION__HPP
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
void if_false_throw(bool condition, std::string msg)
{
    if (!condition)
    {
        throw std::runtime_error(msg);
    }
}

template <typename T>
void validate_elements_throw(const std::vector<T>& data, auto f, const std::string& msg)
{
    auto it = std::find_if(data.begin(), data.end(), f);
    if (it != data.end())
    {
        auto idx = std::distance(data.begin(), it);
        throw std::runtime_error(msg + "\nInvalid element found at index " + std::to_string(idx));
    }
}

#endif
