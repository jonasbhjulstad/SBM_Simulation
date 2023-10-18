
#include <SBM_Simulation/Simulation/Sim_Infection_Sampling.hpp>
#include <SBM_Simulation/Utils/Buffer_Utils.hpp>
#include <SBM_Simulation/Utils/math.hpp>
#include <execution>
#include <iostream>
#include <random>
#define INF_MAX_SAMPLE_LIMIT 10000


void event_inf_summary(const Dataframe::Dataframe_t<State_t, 4> &community_state, const Dataframe::Dataframe_t<uint32_t, 4> &events, const auto &ccms)
{
    auto [Ng, N_sims, Nt, dummy] = community_state.get_ranges();
    for (int g_idx = 0; g_idx < Ng; g_idx++)
    {
        std::cout << "Graph " << g_idx << "\n";
        for (int sim_idx = 0; sim_idx < N_sims; sim_idx++)
        {
            const auto& ccm = ccms[g_idx];
            std::cout << "Simulation " << sim_idx << "\n";
            auto dIs = get_delta_Is(community_state[g_idx][sim_idx]);
            for (int t = 0; t < Nt - 1; t++)
            {
                std::cout << "Timestep " << t << "\n";
                auto N_communities = dIs[t].size();
                for (int c_idx = 0; c_idx < N_communities; c_idx++)
                {
                    auto [r_idx, r_w] = get_related_connections(c_idx, ccm);
                    std::vector<uint32_t> r_events(r_idx.size(), 0);
                    for(int i = 0; i < r_idx.size(); i++)
                    {
                        r_events[i] = events[g_idx][sim_idx][t][r_idx[i]];
                    }
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
void event_inf_validation(const Dataframe::Dataframe_t<State_t, 4> &community_state, const Dataframe::Dataframe_t<uint32_t, 4> &events, const auto &ccms)
{
    auto [Ng, N_sims, Nt, dummy] = community_state.get_ranges();
    for (int g_idx = 0; g_idx < Ng; g_idx++)
    {
        for (int sim_idx = 0; sim_idx < N_sims; sim_idx++)
        {
            const auto& ccm = ccms[g_idx];
            auto dIs = get_delta_Is(community_state[g_idx][sim_idx]);
            for (int t = 0; t < Nt - 1; t++)
            {
                auto N_communities = dIs[t].size();
                for (int c_idx = 0; c_idx < N_communities; c_idx++)
                {
                    auto [r_idx, r_w] = get_related_connections(c_idx, ccm);
                    std::vector<uint32_t> r_events(r_idx.size(), 0);
                    for(int i = 0; i < r_idx.size(); i++)
                    {
                        r_events[i] = events[g_idx][sim_idx][t][r_idx[i]];
                    }

                    if (std::accumulate(r_events.begin(), r_events.end(), 0) < dIs[t][c_idx])
                    {
                        throw std::runtime_error("Error: too few events related to community in timeseries, (g_idx, sim_idx, t, c_idx): (" + std::to_string(g_idx) + "," + std::to_string(sim_idx) + "," + std::to_string(t) + "," + std::to_string(c_idx) + ")");
                    }
                }
            }
        }
    }
}

std::vector<uint32_t> constrained_weight_sample(size_t N_samples, const std::vector<uint32_t> weights, const std::vector<uint32_t> &max_values)
{
    if (N_samples == 0)
    {
        return std::vector<uint32_t>(weights.size(), 0);
    }
    assert(!std::all_of(max_values.begin(), max_values.end(), [](auto x)
                        { return x == 0; }) &&
           "All max values are zero, but infections to be sampled are > 0");
    std::vector<uint32_t> sample_counts(weights.size(), 0);
    std::discrete_distribution<uint32_t> dist(weights.begin(), weights.end());
    std::mt19937 rng(std::random_device{}());
    uint32_t N_sampled = 0;
    for (int i = 0; i < INF_MAX_SAMPLE_LIMIT; i++)
    {
        auto idx = dist(rng);
        if (sample_counts[idx] < max_values[idx])
        {
            sample_counts[idx]++;
            N_sampled++;
        }
        if (N_sampled >= N_samples)
        {
            return sample_counts;
        }
    }
    std::cout << "Warning: Maximum number of samples exceeded\n";
    return sample_counts;
}
template <typename T>
void validate_elements_throw(const std::vector<T>& data, auto f, const std::string& msg)
{
    auto it = std::find_if(data.begin(), data.end(), f);
    if (it != data.end())
    {
        auto idx = std::distance(data.begin(), it);
        throw std::runtime_error(msg + "\nInvalid element found at index " + std::to_string(idx));
    }
}

Dataframe::Dataframe_t<int, 2> get_delta_Is(const Dataframe::Dataframe_t<State_t, 2> &community_state)
{
    auto [Nt, N_communities] = community_state.get_ranges();

    auto I_trajectories = community_state.apply([](const State_t& state)
    {
        return state[1];
    });
    auto R_trajectories = community_state.apply([](const State_t& state)
    {
        return state[2];
    });

    Dataframe::Dataframe_t<int, 2> delta_I(Nt-1, N_communities);
    Dataframe::Dataframe_t<int, 2> delta_R(Nt-1, N_communities);
    for(int t = 0; t < Nt-1; t++)
    {
        for(int col_idx = 0; col_idx < N_communities; col_idx++)
        {
            delta_R[t][col_idx] = R_trajectories[t+1][col_idx] - R_trajectories[t][col_idx];
            delta_I[t][col_idx] = I_trajectories[t+1][col_idx] - I_trajectories[t][col_idx] + delta_R[t][col_idx];
        }
    }

    for(int t = 0; t < Nt-1; t++)
    {
    validate_elements_throw(delta_I[t], [](auto x){return x < 0;}, "Negative delta_I");
    validate_elements_throw(delta_R[t], [](auto x){return x < 0;}, "Negative delta_R");
    }
    return delta_I;
}

std::tuple<std::vector<uint32_t>,std::vector<uint32_t>> get_related_connections(size_t c_idx, const std::vector<Edge_t> &ccm)
{
    auto ccm_weights = Edge_t::get_weights(ccm);
    std::vector<uint32_t> connection_indices;
    std::vector<uint32_t> connection_weights;
    for (int i = 0; i < ccm.size(); i++)
    {
        if (ccm[i].from == c_idx)
        {
            connection_indices.push_back(i);
            connection_weights.push_back(ccm_weights[i]);
        }
    }
    return std::make_tuple(connection_indices, connection_weights);
}
std::vector<uint32_t> get_related_events(size_t c_idx, const std::vector<Edge_t> &ccm, const std::vector<uint32_t> &events)
{
    auto [r_sql, rw] = get_related_connections(c_idx, ccm);
    std::vector<uint32_t> r_con_events(r_con.size(), 0);
    for (int i = 0; i < r_con_events.size(); i++)
    {
        r_con_events[i] = events[r_con[i]];
    }
    return r_con_events;
}
auto get_community_connections(size_t N_communities, const auto &ccm)
{
    std::vector<uint32_t> community_indices(N_communities);
    std::iota(community_indices.begin(), community_indices.end(), 0);
    std::vector<std::vector<uint32_t>> indices(community_indices.size());
    std::vector<std::vector<uint32_t>> weights(community_indices.size());

    for (int i = 0; i < community_indices.size(); i++)
    {
        auto [rc, rw] = get_related_connections(community_indices[i], ccm);
        indices[i] = rc;
        weights[i] = rw;
    }
    return std::make_tuple(indices, weights);
}

auto zip_merge(const std::vector<uint32_t> &v0, const std::vector<uint32_t> &v1)
{
    std::vector<uint32_t> result(v0.size() + v1.size());
    for (int i = 0; i < v0.size(); i++)
    {
        result[2 * i] = v0[i];
        result[2 * i + 1] = v1[i];
    }
    return result;
}

void verbose_community_infection_debug(auto c_idx, const auto &related_connections)
{
    std::cout << "Sampling for community " << c_idx << "\n";
    std::cout << "Related connections: ";
    for (auto &&rcon : related_connections)
    {
        std::cout << rcon << ", ";
    }
    std::cout << "\n";
}



std::vector<uint32_t> sample_community(const auto &related_connections, const auto &related_weights, const auto &events, auto N_samples)
{
    auto N_connections = events.size();
    std::vector<uint32_t> result(N_connections, 0);
    if (!N_samples)
        return result;
    std::vector<uint32_t> r_con_events(related_connections.size(), 0);
    for (int i = 0; i < related_connections.size(); i++)
    {
        r_con_events[i] = events[related_connections[i]];
    }
    auto sample_counts = constrained_weight_sample(N_samples, related_weights, r_con_events);
    for (int sample_idx = 0; sample_idx < sample_counts.size(); sample_idx++)
    {
        result[related_connections[sample_idx]] = sample_counts[sample_idx];
    }
    return result;
}

std::vector<uint32_t> sample_timestep(const std::vector<uint32_t> &events, const std::vector<int> &delta_I, const std::vector<Edge_t> &ccm)
{
    auto N_communities = delta_I.size();
    auto N_connections = ccm.size();
    auto merge_sample_result = [&](const std::vector<std::vector<uint32_t>> &sample_result)
    {
        std::vector<uint32_t> merged_result(N_connections, 0);
        for (int i = 0; i < sample_result.size(); i++)
        {
            for (int j = 0; j < sample_result[i].size(); j++)
            {
                merged_result[j] += sample_result[i][j];
            }
        }
        return merged_result;
    };

    std::vector<std::vector<uint32_t>> result(N_communities, std::vector<uint32_t>(N_connections, 0));
    auto community_idx = make_iota(N_communities);
    std::transform(std::execution::par_unseq, community_idx.begin(), community_idx.end(), result.begin(), [&](auto c_idx)
                   {
                    auto [r_sql, r_weight] = get_related_connections(c_idx, ccm);
                    auto dI = delta_I[c_idx];
                    return sample_community(r_sql, r_weight, events, dI); });
    auto merged_result = merge_sample_result(result);
    uint32_t merged_infs = std::accumulate(merged_result.begin(), merged_result.end(), 0);
    uint32_t true_infs = std::accumulate(delta_I.begin(), delta_I.end(), 0);
    assert(merged_infs == true_infs && "Sampled infections do not match true infections");
    return merged_result;
}
Dataframe::Dataframe_t<uint32_t, 2> sample_infections(const Dataframe::Dataframe_t<State_t, 2> &community_state, const Dataframe::Dataframe_t<uint32_t, 2> &events, const std::vector<Edge_t> &ccm, uint32_t seed)
{
    auto N_connections = ccm.size();
    auto delta_Is = get_delta_Is(community_state);
    auto Nt = delta_Is.size();
    Dataframe::Dataframe_t<uint32_t, 2> sampled_infections(Nt, N_connections * 2);
    auto t_vec = make_iota(Nt);
    std::transform(std::execution::par_unseq, t_vec.begin(), t_vec.end(), sampled_infections.data.begin(), [&events, &delta_Is, &ccm](auto t)
                   { return sample_timestep(events[t], delta_Is[t], ccm); });

    return sampled_infections;
}


Dataframe::Dataframe_t<uint32_t, 3> sample_infections(const Dataframe::Dataframe_t<State_t, 3> &community_state, const Dataframe::Dataframe_t<uint32_t, 3> &events, const std::vector<Edge_t> &ccm, uint32_t seed)
{
    Dataframe::Dataframe_t<uint32_t, 3> result(events.get_ranges());
    auto df_pack = Dataframe::dataframe_tie(community_state, events);
    auto seeds = generate_seeds(df_pack.size(), seed);
    std::transform(df_pack.begin(), df_pack.end(), seeds.begin(), result.data.begin(), [ccm](const auto& pack, auto seed)
    {
        auto [community_state, events] = pack;
        return sample_infections(community_state, events, ccm, seed);
    });
    return result;
}
Dataframe::Dataframe_t<uint32_t, 4> sample_infections(const Dataframe::Dataframe_t<State_t, 4> &community_state, const Dataframe::Dataframe_t<uint32_t, 4> &events, const Dataframe::Dataframe_t<Edge_t, 2> &ccm, uint32_t seed)
{
    Dataframe::Dataframe_t<uint32_t, 4> result(events.get_ranges());
    auto df_pack = Dataframe::dataframe_tie(community_state, events);
    auto seeds = generate_seeds(df_pack.size(), seed);
    auto N_graphs = community_state.size();
    for(int g_idx = 0; g_idx < N_graphs; g_idx++)
    {
        result[g_idx] = sample_infections(community_state[g_idx], events[g_idx], ccm[g_idx], seeds[g_idx]);
    }
    return result;
}
