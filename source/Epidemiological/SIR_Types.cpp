#include <SBM_Simulation/Epidemiological/SIR_Types.hpp>
bool State_t::is_valid(uint32_t N_pop) const
{
    return ((*this)[0] + (*this)[1] + (*this)[2]) <= N_pop;
}
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

} // namespace soci
