#include <SBM_Simulation/Epidemiological/SIR_Types.hpp>
namespace soci
{
    void type_conversion<State_t>::from_base(const std::string &s, soci::indicator ind, State_t &state)
    {
        if (ind == i_null)
        {
            throw soci_error("Null value not allowed for this type");
        }
        state = State_t::from_string(s);
    }

    void type_conversion<State_t>::to_base(const State_t &state, std::string &s, soci::indicator &ind)
    {
        s = State_t::to_string(state);
        ind = i_ok;
    }

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
} // namespace soci
