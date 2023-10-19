#include <SBM_Simulation/Graph/Graph_Types.hpp>
#include <algorithm>
std::fstream& operator<<(std::fstream &os, const Edge_t &e)
{
os << e.to_array_string();
return os;
}
std::ofstream& operator<<(std::ofstream &os, const Edge_t &e)
{
os << e.to_array_string();
return os;
}

std::ostream& operator<<(std::ostream &os, const Edge_t &e)
{
os << e.to_array_string();
return os;
}

std::stringstream& operator<<(std::stringstream &os, const Edge_t &e)
{
os << e.to_array_string();
return os;
}

std::string Edge_t::to_array_string() const
{
    return "'{" + std::to_string(from) + "," + std::to_string(to) + "," + std::to_string(weight) + "}'";
}

std::vector<uint32_t> Edge_t::get_weights(const std::vector<Edge_t> &edges)
{
    std::vector<uint32_t> result(edges.size());
    std::transform(edges.begin(), edges.end(), result.begin(), [](auto edge)
                   { return edge.weight; });
    return result;
}
std::vector<uint32_t> Edge_t::get_from(const std::vector<Edge_t> &edges)
{
    std::vector<uint32_t> result(edges.size());
    std::transform(edges.begin(), edges.end(), result.begin(), [](auto edge)
                   { return edge.from; });
    return result;
}
std::vector<uint32_t> Edge_t::get_to(const std::vector<Edge_t> &edges)
{
    std::vector<uint32_t> result(edges.size());
    std::transform(edges.begin(), edges.end(), result.begin(), [](auto edge)
                   { return edge.to; });
    return result;
}

// std::vector<uint32_t> Edge_t::get_from(const std::vector<Edge_t> &edges)
// {
//     std::vector<uint32_t> result(edges.size());
//     std::transform(edges.begin(), edges.end(), result.begin(), [](auto edge)
//                    { return edge.from; });
//     return result;
// }
// std::vector<uint32_t> Edge_t::get_to(const std::vector<Edge_t> &edges)
// {
//     std::vector<uint32_t> result(edges.size());
//     std::transform(edges.begin(), edges.end(), result.begin(), [](auto edge)
//                    { return edge.to; });
//     return result;
// }
