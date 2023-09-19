#ifndef SYCL_GRAPH_UTILS_VALIDATION_HPP
#define SYCL_GRAPH_UTILS_VALIDATION_HPP
#include <string>
#include <stdexcept>
void if_false_throw(bool condition, std::string msg)
{
    if (!condition)
    {
        throw std::runtime_error(msg);
    }
};

#endif
