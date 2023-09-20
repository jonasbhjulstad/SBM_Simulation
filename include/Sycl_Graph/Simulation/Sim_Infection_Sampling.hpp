#ifndef SIR_INFECTION_SAMPLING_HPP
#define SIR_INFECTION_SAMPLING_HPP
#include <Sycl_Graph/Simulation/Sim_Types.hpp>
#include <Sycl_Graph/Utils/Dataframe.hpp>
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
#include <execution>
// std::vector<uint32_t> sample_connection_infections(Inf_Sample_Data_t &z);
auto make_iota(auto N)
{
    std::vector<uint32_t> result(N);
    std::iota(result.begin(), result.end(), 0);
    return result;
}
// std::vector<uint32_t> sample_timestep_infections(const std::vector<int> &delta_Is, const std::vector<uint32_t> &from_events, const std::vector<uint32_t> &to_events, const std::vector<uint32_t> &ccm, const std::vector<uint32_t> &ccm_weights, uint32_t N_connections, uint32_t seed);
Dataframe_t<int, 2> get_delta_Is(const Dataframe_t<State_t, 2> &community_state);
std::tuple<std::vector<uint32_t>,std::vector<uint32_t>> get_related_connections(size_t c_idx, const std::vector<Edge_t> &ccm);
std::vector<uint32_t> get_related_events(size_t c_idx, const std::vector<Edge_t> &ccm, const std::vector<uint32_t> &events);


std::vector<uint32_t> sample_timestep(const std::vector<uint32_t> &events, const std::vector<int> &delta_I, const std::vector<Edge_t> &ccm);

Dataframe_t<uint32_t, 2> sample_infections(const Dataframe_t<State_t, 2> &community_state, const Dataframe_t<uint32_t, 2> &events, const Dataframe_t<Edge_t, 2> &ccm, uint32_t seed)
{
    auto N_connections = ccm[0].size();
    auto delta_Is = get_delta_Is(community_state);
    auto Nt = delta_Is.size();
    Dataframe_t<uint32_t, 2> sampled_infections(Nt, N_connections * 2);
    auto t_vec = make_iota(Nt);
    std::transform(std::execution::par_unseq, t_vec.begin(), t_vec.end(), sampled_infections.data.begin(), [&events, &delta_Is, &ccm](auto t)
                   { return sample_timestep(events[t], delta_Is[t], ccm[0]); });

    return sampled_infections;
}

void event_inf_summary(const Dataframe_t<State_t, 4> &community_state, const Dataframe_t<uint32_t, 4> &events, const auto &ccms)
{
    auto [Ng, N_sims, Nt, dummy] = community_state.get_ranges();
    for (int g_idx = 0; g_idx < Ng; g_idx++)
    {
        std::cout << "Graph " << g_idx << "\n";
        for (int sim_idx = 0; sim_idx < N_sims; sim_idx++)
        {
            std::cout << "Simulation " << sim_idx << "\n";
            auto dIs = get_delta_Is(community_state[g_idx][sim_idx]);
            for (int t = 0; t < Nt - 1; t++)
            {
                std::cout << "Timestep " << t << "\n";
                auto N_communities = dIs[t].size();
                for (int c_idx = 0; c_idx < N_communities; c_idx++)
                {
                    auto [r_idx, r_w] = get_related_connections(c_idx, ccms[g_idx][sim_idx][t]);
                    auto r_events = get_related_events(c_idx, ccms[g_idx][sim_idx][t], events[g_idx][sim_idx][t]);
                    std::cout << "Community " << c_idx << "\n";
                    std::cout << "Delta I: " << dIs[t][c_idx] << "\n";
                    std::cout << "Related events: ";
                    for (auto &&e : r_events)
                    {
                        std::cout << e << ", ";
                    }
                    std::cout << "Related idx: ";
                    for (auto &&e : r_idx)
                    {
                        std::cout << e << ", ";
                    }

                    std::cout << "\n";
                    if (std::accumulate(r_events.begin(), r_events.end(), 0) < dIs[t][c_idx])
                    {
                        throw std::runtime_error("Error: too few events related to community in timeseries, (g_idx, sim_idx, t, c_idx): (" + std::to_string(g_idx) + "," + std::to_string(sim_idx) + "," + std::to_string(t) + "," + std::to_string(c_idx) + ")");
                    }
                }
            }
        }
    }
}


template <std::size_t N>
std::enable_if_t<(N > 2), Dataframe_t<uint32_t, N>> sample_infections(const Dataframe_t<State_t, N> &community_state, const Dataframe_t<uint32_t, N> &events, const Dataframe_t<Edge_t, N> &ccm, uint32_t seed);

template <std::size_t N>
std::enable_if_t<(N > 2), Dataframe_t<uint32_t, N>> sample_infections(const Dataframe_t<State_t, N> &community_state, const Dataframe_t<uint32_t, N> &events, const Dataframe_t<Edge_t, N> &ccm, uint32_t seed)
{
    Dataframe_t<uint32_t, N> result(events.get_ranges());
    auto df_pack = dataframe_tie(community_state, events, ccm);
    auto seeds = generate_seeds(df_pack.size(), seed);
    std::transform(df_pack.begin(), df_pack.end(), seeds.begin(), result.data.begin(), [](const auto& pack, auto seed)
    {
        auto [community_state, events, ccm] = pack;
        return sample_infections(community_state, events, ccm, seed);
    });
    return result;
}

#endif
