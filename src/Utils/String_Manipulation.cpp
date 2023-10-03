#include <Sycl_Graph/Utils/String_Manipulation.hpp>

std::string float_to_decimal_string(float f, int precision)
{
    std::string result = std::to_string(f);
    auto decimal_pos = result.find(".");
    if (decimal_pos == std::string::npos)
    {
        return result;
    }
    else
    {
        return result.substr(0, decimal_pos + precision + 1);
    }
}
