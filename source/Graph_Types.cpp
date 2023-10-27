#include <SBM_Graph/Graph_Types.hpp>
#include <algorithm>
std::vector<uint32_t> Edge_t::get_from(const std::vector<Edge_t> &edges)
{
    std::vector<uint32_t> result(edges.size());
    std::transform(edges.begin(), edges.end(), result.begin(), [](const Edge_t &e)
                   { return e.from; });
    return result;
}
std::vector<uint32_t> Edge_t::get_to(const std::vector<Edge_t> &edges)
{
    std::vector<uint32_t> result(edges.size());
    std::transform(edges.begin(), edges.end(), result.begin(), [](const Edge_t &e)
                   { return e.to; });
    return result;
}

std::vector<uint32_t> Weighted_Edge_t::get_from(const std::vector<Weighted_Edge_t> &edges)
{
    std::vector<uint32_t> result(edges.size());
    std::transform(edges.begin(), edges.end(), result.begin(), [](const Weighted_Edge_t &e)
                   { return e.from; });
    return result;
}
std::vector<uint32_t> Weighted_Edge_t::get_to(const std::vector<Weighted_Edge_t> &edges)
{
    std::vector<uint32_t> result(edges.size());
    std::transform(edges.begin(), edges.end(), result.begin(), [](const Weighted_Edge_t &e)
                   { return e.to; });
    return result;
}
std::vector<uint32_t> Weighted_Edge_t::get_weights(const std::vector<Weighted_Edge_t> &edges)
{
    std::vector<uint32_t> result(edges.size());
    std::transform(edges.begin(), edges.end(), result.begin(), [](const Weighted_Edge_t &e)
                   { return e.weight; });
    return result;
}
