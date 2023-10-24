#include <SBM_Simulation/Graph/Graph_Types.hpp>
#include <algorithm>
std::fstream &operator<<(std::fstream &os, const Edge_t &e)
{
    os << e.to_array_string();
    return os;
}
std::ofstream &operator<<(std::ofstream &os, const Edge_t &e)
{
    os << e.to_array_string();
    return os;
}

std::ostream &operator<<(std::ostream &os, const Edge_t &e)
{
    os << e.to_array_string();
    return os;
}

std::stringstream &operator<<(std::stringstream &os, const Edge_t &e)
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

typename Edge_t::Serialized_Data_t  Edge_t::serialize() const
{
    return std::make_tuple(from, to, weight);
}


namespace soci
{
    void type_conversion<Edge_t>::from_base(const std::string &s, soci::indicator ind, Edge_t &edge)
    {
        if (ind == i_null)
        {
            throw soci_error("Null value not allowed for this type");
        }
        std::string s1 = s.substr(0, s.find(","));
        std::string s2 = s.substr(s.find(",") + 1, s.size() - 1);
        edge = Edge_t{(uint32_t)std::stoi(s1), (uint32_t)std::stoi(s2), 0};
    }
    void type_conversion<Edge_t>::to_base(const Edge_t &edge, std::string &s, soci::indicator &ind)
    {
        s = std::to_string(edge.from) + "," + std::to_string(edge.to);
        ind = i_ok;
    }
}
