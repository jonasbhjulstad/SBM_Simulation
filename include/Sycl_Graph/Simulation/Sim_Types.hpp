#ifndef SIM_TYPES_HPP
#define SIM_TYPES_HPP
#include <Sycl_Graph/Epidemiological/SIR_Types.hpp>

struct Sim_Param
{
    // construct with all params
    Sim_Param(uint32_t N_pop, const std::vector<uint32_t>& N_communities, float p_in, float p_out, uint32_t N_sims, uint32_t Nt, uint32_t Nt_alloc, uint32_t seed, float p_I_min, float p_I_max)
        : N_pop(N_pop), N_communities(N_communities), N_graphs(N_communities.size()), p_in(p_in), p_out(p_out), N_sims(N_sims), Nt(Nt), Nt_alloc(Nt_alloc), seed(seed), p_I_min(p_I_min), p_I_max(p_I_max)
    {
    }
    uint32_t N_pop = 100;
    std::vector<uint32_t> N_communities;
    float p_in = 1.0f;
    float p_out = 0.5f;
    uint32_t N_graphs = 2;
    uint32_t N_sims = 2;
    uint32_t Nt = 56;
    uint32_t Nt_alloc = 20;
    uint32_t seed = 234;
    float p_I_min = 0.1f;
    float p_I_max = 0.2f;
    uint32_t p_out_idx = 0;
    std::size_t N_sims_tot() const { return N_graphs * N_sims; }
    std::size_t N_communities_max() const;
    std::vector<uint32_t> N_connections() const;
    std::size_t N_connections_tot() const;
    std::size_t N_connections_max() const;
};

#endif
