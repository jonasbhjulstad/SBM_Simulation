#include <Sycl_Graph/Simulation/Sim_Data.hpp>
Sim_Data::Sim_Data(uint32_t Nt, uint32_t N_sims, uint32_t N_communities, uint32_t N_connections):
events_to_timeseries(Nt, std::vector<std::vector<uint32_t>>(N_sims, std::vector<uint32_t>(N_connections, 0))),
events_from_timeseries(Nt, std::vector<std::vector<uint32_t>>(N_sims, std::vector<uint32_t>(N_connections, 0))),
state_timeseries(Nt+1, std::vector<std::vector<State_t>>(N_sims, std::vector<State_t>(N_communities, {0,0,0}))),
connection_infections(Nt, std::vector<std::vector<uint32_t>>(N_sims, std::vector<uint32_t>(N_connections, 0)))
{
}
