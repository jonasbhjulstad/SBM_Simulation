
#include <Sycl_Graph/Dynamics.hpp>
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
#include <Sycl_Graph/Utils/Buffer_Validation.hpp>

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
        auto prev_offset = p.N_sims*N_vertices*t_alloc + it[0]*N_vertices;
        auto next_offset = p.N_sims*N_vertices*(t_alloc + 1) + it[0]*N_vertices;
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
        auto sim_offset = it[0]*N_vertices;
        auto next_offset = p.N_sims*N_vertices*(t_alloc + 1) + sim_offset;
        auto prev_offset = p.N_sims*N_vertices*(t_alloc) + sim_offset;
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

std::vector<sycl::event> infect(sycl::queue &q,
                                const Sim_Param &p,
                                Sim_Buffers &b,
                                uint32_t t,
                                std::vector<sycl::event> &dep_event)
{

    uint32_t N_connections = b.N_connections;
    uint32_t N_edges = b.N_edges;
    uint32_t N_vertices = p.N_communities * p.N_pop;
    uint32_t N_sims = p.N_sims;
    uint32_t t_alloc = t % p.Nt_alloc;
    uint32_t Nt = p.Nt;

    auto inf_event = q.submit([&](sycl::handler &h)
                              {
                                  h.depends_on(dep_event);
                                  h.parallel_for(p.N_sims, [&](sycl::item<1> it)
                                                            {
                                        uint32_t N_inf = 0;
                                        auto sim_offset = it[0]*N_vertices;
                                        auto t_offset = t*p.N_sims*N_vertices;
                                          for (uint32_t edge_idx = 0; edge_idx < N_edges; edge_idx++)
                                          {
                                            auto connection_id = b.ecm[edge_idx];
                                            auto v_from_id = b.edge_from[edge_idx];
                                            auto v_to_id = b.edge_to[edge_idx];
                                            const auto v_prev_from = b.vertex_state[t_offset + sim_offset + v_from_id];
                                            const auto v_prev_to = b.vertex_state[t_offset + sim_offset + v_to_id];

                                            auto& v_next_from = b.vertex_state[t_offset + p.N_sims*N_vertices + sim_offset + v_from_id];
                                            auto& v_next_to = b.vertex_state[t_offset + p.N_sims*N_vertices + sim_offset + v_to_id];

                                              if ((v_prev_from == SIR_INDIVIDUAL_S) && (v_prev_to == SIR_INDIVIDUAL_I))
                                              {
                                                  float p_I = b.p_Is[N_connections*p.N_sims*t + N_connections*it[0] + connection_id];
                                                  Static_RNG::bernoulli_distribution<float> bernoulli_I(p_I);
                                                  auto &rng = b.rngs[it];
                                                  bernoulli_I.p = p_I;
                                                  if (bernoulli_I(rng))
                                                  {
                                                    N_inf++;

                                                      v_next_from = SIR_INDIVIDUAL_I;
                                                      b.events_to[N_connections*p.N_sims*t + N_connections*it[0] + connection_id]++;
                                                  }
                                              }
                                              else if ((v_prev_from == SIR_INDIVIDUAL_I) && (v_prev_to == SIR_INDIVIDUAL_S))
                                              {
                                                  float p_I = b.p_Is[N_connections*p.N_sims*t + N_connections*it[0] + connection_id];
                                                  Static_RNG::bernoulli_distribution<float> bernoulli_I(p_I);
                                                  auto &rng = b.rngs[it]; // was lid
                                                  bernoulli_I.p = p_I;
                                                  if (bernoulli_I(rng))
                                                  {

                                                    N_inf++;
                                                    v_next_to = SIR_INDIVIDUAL_I;
                                                      b.events_from[N_connections*p.N_sims*t + N_connections*it[0] + connection_id]++;
                                                  }
                                              }

                                          } }); });

    std::cout << "Timestep " << t << " done\n";
    return {inf_event};
}
