#include <SBM_Simulation/Epidemiological/SIR_Types.hpp>
bool State_t::is_valid(uint32_t N_pop) const
{
    return ((*this)[0] + (*this)[1] + (*this)[2]) <= N_pop;
}
