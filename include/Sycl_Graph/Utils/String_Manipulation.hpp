#ifndef STRING_MANIPULATION_HPP
#define STRING_MANIPULATION_HPP
#include <string>
#include <vector>
#include <sstream>
#include <cstdint>
std::string float_to_decimal_string(float f, int precision = 2);


std::string vector_to_string(const std::vector<float>& vec, int precision = 4);
std::string vector_to_string(const std::vector<uint32_t>& vec);

template <typename ... Ts>
std::string tuple_to_string(const std::tuple<Ts...>& data)
{
    std::stringstream ss;
    std::apply([&ss](auto&&... args) { ((ss << args << ", "), ...); }, data);
    // remove last comma
    auto str = ss.str();
    str.pop_back();
    str.pop_back();
    return str;
}

template <typename... Ts>
std::vector<std::string> tuple_to_str_vec(const std::tuple<Ts...> &data)
{
    std::vector<std::string> vec;
    std::apply([&vec](auto &&...args)
               { ((vec.push_back(std::to_string(args))), ...); },
               data);
    return vec;
}
std::string merge_strings(const std::vector<std::string>& strings, std::string delim = "");

template <typename ... Ts>
std::string quoted_tuple_to_string(const std::tuple<Ts...>& data)
{
    std::stringstream ss;
    std::apply([&ss](auto&&... args) { ((ss << "\"" << args << "\", "), ...); }, data);
    // remove last comma
    auto str = ss.str();
    str.pop_back();
    str.pop_back();
    return str;
}



#endif
