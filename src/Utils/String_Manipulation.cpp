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

std::string merge_strings(const std::vector<std::string>& strings, std::string delim)
{
    std::string result;
    for (int i = 0; i < strings.size(); i++)
    {
        result += strings[i];
        result += (i != (strings.size()-1)) ? delim : "";
    }
    return result;
}

std::string vector_to_string(const std::vector<float>& vec, int precision)
{
    std::string result = "'{";
    for (const auto& v : vec)
    {
        result += float_to_decimal_string(v, precision) + ",";
    }
    result.pop_back();
    result += "}'";
    return result;
}

std::string vector_to_string(const std::vector<uint32_t>& vec)
{
    std::string result = "'{";
    for (const auto& v : vec)
    {
        result += std::to_string(v) + ",";
    }
    result.pop_back();
    result += "}'";
    return result;
}
