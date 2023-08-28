
#include <Sycl_Graph/Dynamics.hpp>
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
#include <Sycl_Graph/Utils/Buffer_Validation.hpp>
#include <Sycl_Graph/Simulation/Sim_Types.hpp>
#include <Sycl_Graph/Utils/Vector_Remap.hpp>

std::vector<sycl::event> recover(sycl::queue &q,
                                 const Sim_Param &p,
                                 Sim_Buffers &b,
                                 uint32_t t,
                                 std::vector<sycl::event> &dep_event)
{
    float p_R = p.p_R;
    uint32_t N_vertices = p.N_pop * p.N_communities;
    uint32_t N_sims = p.N_sims;
    auto t_alloc = t % p.Nt_alloc;
    auto cpy_event = q.submit([&](sycl::handler &h)
                              {
    h.depends_on(dep_event);
    h.parallel_for(p.N_sims, [=](sycl::item<1> it)
    {
        auto prev_offset = get_linear_offset(N_sims, p.Nt_alloc, N_vertices, it[0], t_alloc, 0);
        auto next_offset = get_linear_offset(N_sims, p.Nt_alloc, N_vertices, it[0], t_alloc + 1, 0);
        for(int v_idx = 0; v_idx < N_vertices; v_idx++)
        {
            b.vertex_state[next_offset + v_idx] = b.vertex_state[prev_offset + v_idx];
        }
    }); });

    auto event = q.submit([&](sycl::handler &h)
                          {
                              h.depends_on(cpy_event);
                              h.parallel_for(p.N_sims, [=](sycl::item<1> it)
                                             {
        Static_RNG::bernoulli_distribution<float> bernoulli_R(p_R);
        auto next_offset = get_linear_offset(N_sims, p.Nt_alloc, N_vertices, it[0], t_alloc + 1, 0);
        auto prev_offset = get_linear_offset(N_sims, p.Nt_alloc, N_vertices, it[0], t_alloc, 0);
        auto& rng = b.rngs[it];
        for(int v_idx = 0; v_idx < N_vertices; v_idx++)
        {
            auto state_prev = b.vertex_state[prev_offset + v_idx];
            if (state_prev == SIR_INDIVIDUAL_I) {
                if (bernoulli_R(rng)) {
                b.vertex_state[next_offset + v_idx] = SIR_INDIVIDUAL_R;
                }
            }
            } }); });
return {event};
}

sycl::event initialize_vertices(sycl::queue &q, const Sim_Param &p,
                                Sim_Buffers &b)
{
    uint32_t N_vertices = p.N_pop * p.N_communities;
    float p_I0 = p.p_I0;
    float p_R0 = p.p_R0;

    auto init_event = q.submit([&](sycl::handler &h)
                               { h.parallel_for(p.N_sims, [=](sycl::item<1> it)
                                                {

            auto sim_offset = it[0]*N_vertices;
            for(int vertex_idx = sim_offset; vertex_idx < sim_offset + N_vertices; vertex_idx++)
            {
            auto& rng = b.rngs[it];
            Static_RNG::bernoulli_distribution<float> bernoulli_I(p_I0);
            Static_RNG::bernoulli_distribution<float> bernoulli_R(p_R0);
            if (bernoulli_I(rng)) {
                b.vertex_state[it] = SIR_INDIVIDUAL_I;
            } else if (bernoulli_R(rng)) {
                b.vertex_state[it] = SIR_INDIVIDUAL_R;
            } else {
                b.vertex_state[it] = SIR_INDIVIDUAL_S;
            }} }); });

    return init_event;
}

SYCL_EXTERNAL auto floor_div(auto a, auto b) { return static_cast<uint32_t>(std::floor(static_cast<double>(a) / static_cast<double>(b))); }

std::vector<sycl::event> infect(sycl::queue &q,
                                const Sim_Param &p,
                                Sim_Buffers &b,
                                uint32_t t,
                                std::vector<sycl::event> &dep_event)
{

    uint32_t N_connections = b.N_connections;
    uint32_t N_vertices = p.N_communities * p.N_pop;
    uint32_t N_sims = p.N_sims;
    uint32_t t_alloc = t % p.Nt_alloc;
    uint32_t Nt = p.Nt;

    auto inf_event = q.submit([&](sycl::handler &h)
                              {
                                  h.depends_on(dep_event);
                                  h.parallel_for(p.N_sims, [=](sycl::item<1> it)
                                                            {
                                        uint32_t N_inf = 0;
                                        auto graph_idx = floor_div(it[0], N_sims);
                                        auto v_prev_offset = get_linear_offset(N_sims, p.Nt_alloc, N_vertices, it[0], t_alloc, 0);
                                        auto v_next_offset = get_linear_offset(N_sims, p.Nt_alloc, N_vertices, it[0], t_alloc + 1, 0);
                                        auto e_prev_offset = get_linear_offset(N_sims, p.Nt_alloc, N_connections, it[0], t_alloc, 0);
                                        auto e_next_offset = get_linear_offset(N_sims, p.Nt_alloc, N_connections, it[0], t_alloc + 1, 0);
                                        auto p_I_offset = get_linear_offset(N_sims, p.Nt, N_connections, it[0], t, 0);
                                          for (uint32_t edge_idx = b.edge_offsets[graph_idx]; edge_idx < b.edge_offsets[graph_idx+1]; edge_idx++)
                                          {
                                            auto connection_id = b.ecm[edge_idx];
                                            auto v_from_id = b.edge_from[edge_idx];
                                            auto v_to_id = b.edge_to[edge_idx];
                                            const auto v_prev_from = b.vertex_state[v_prev_offset + v_from_id];
                                            const auto v_prev_to = b.vertex_state[v_prev_offset + v_to_id];

                                            auto& v_next_from = b.vertex_state[v_next_offset + v_from_id];
                                            auto& v_next_to = b.vertex_state[v_next_offset + v_to_id];

                                              if ((v_prev_from == SIR_INDIVIDUAL_S) && (v_prev_to == SIR_INDIVIDUAL_I))
                                              {
                                                  float p_I = b.p_Is[p_I_offset + connection_id];
                                                  Static_RNG::bernoulli_distribution<float> bernoulli_I(p_I);
                                                  auto &rng = b.rngs[it];
                                                  bernoulli_I.p = p_I;
                                                  if (bernoulli_I(rng))
                                                  {
                                                    N_inf++;

                                                      v_next_from = SIR_INDIVIDUAL_I;
                                                      b.events_to[e_next_offset + connection_id]++;
                                                  }
                                              }
                                              else if ((v_prev_from == SIR_INDIVIDUAL_I) && (v_prev_to == SIR_INDIVIDUAL_S))
                                              {
                                                  float p_I = b.p_Is[p_I_offset + connection_id];
                                                  Static_RNG::bernoulli_distribution<float> bernoulli_I(p_I);
                                                  auto &rng = b.rngs[it]; // was lid
                                                  bernoulli_I.p = p_I;
                                                  if (bernoulli_I(rng))
                                                  {

                                                    N_inf++;
                                                    v_next_to = SIR_INDIVIDUAL_I;
                                                      b.events_from[e_next_offset + connection_id]++;
                                                  }
                                              }

                                          } }); });

    std::cout << "Timestep " << t << " done\n";
    return {inf_event};
}
