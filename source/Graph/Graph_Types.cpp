#include <SBM_Simulation/Graph/Graph_Types.hpp>
#include <algorithm>
std::fstream& operator<<(std::fstream &os, const Edge_t &e)
{
os << e.from << "," << e.to << "," << e.weight;
return os;
}
std::ofstream& operator<<(std::ofstream &os, const Edge_t &e)
{
os << e.from << "," << e.to << "," << e.weight;
return os;
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

std::vector<uint32_t> Edge_t::get_from(const std::vector<std::pair<uint32_t, uint32_t>> &edges)
{
    std::vector<uint32_t> result(edges.size());
    std::transform(edges.begin(), edges.end(), result.begin(), [](auto edge)
                   { return edge.first; });
    return result;
}
std::vector<uint32_t> Edge_t::get_to(const std::vector<std::pair<uint32_t, uint32_t>> &edges)
{
    std::vector<uint32_t> result(edges.size());
    std::transform(edges.begin(), edges.end(), result.begin(), [](auto edge)
                   { return edge.second; });
    return result;
}
