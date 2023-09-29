#include <Sycl_Graph/Utils/Validation.hpp>

void if_false_throw(bool condition, std::string msg)
{
    if (!condition)
    {
        throw std::runtime_error(msg);
    }
}
