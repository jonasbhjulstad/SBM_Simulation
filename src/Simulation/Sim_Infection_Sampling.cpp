
#include <Sycl_Graph/Simulation/Sim_Infection_Sampling.hpp>
#include <Sycl_Graph/Simulation/Sim_Types.hpp>
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
#include <Sycl_Graph/Utils/Dataframe.hpp>
#include <algorithm>
#include <execution>
#include <iostream>
#include <random>
#define INF_MAX_SAMPLE_LIMIT 10000

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

std::vector<std::vector<int>> get_delta_Is(const std::vector<std::vector<State_t>> &community_state)
{
    uint32_t N_communities = community_state[0].size();
    std::vector<std::vector<int>> I_trajectories(community_state.size(), std::vector<int>(N_communities));
    std::vector<std::vector<int>> R_trajectories(community_state.size(), std::vector<int>(N_communities));

    for (int i = 0; i < community_state.size(); i++)
    {
        for (int j = 0; j < N_communities; j++)
        {
            I_trajectories[i][j] = community_state[i][j][1];
            R_trajectories[i][j] = community_state[i][j][2];
        }
    }

    std::vector<std::vector<int>> delta_R = diff(R_trajectories);
    std::vector<std::vector<int>> delta_I = diff(I_trajectories);

    for (int i = 0; i < delta_I.size(); i++)
    {
        for (int j = 0; j < delta_I[i].size(); j++)
        {
            delta_I[i][j] += delta_R[i][j];
        }
    }

    for (int i = 0; i < delta_I.size(); i++)
    {
        assert(std::all_of(delta_I[i].begin(), delta_I[i].end(), [](auto x)
                           { return x >= 0; }) &&
               "Negative delta_I");
    }

    for (int i = 0; i < delta_R.size(); i++)
    {
        assert(std::all_of(delta_R[i].begin(), delta_R[i].end(), [](auto x)
                           { return x >= 0; }) &&
               "Negative delta_R");
    }

    return delta_I;
}

auto get_related_connections(size_t c_idx, const std::vector<std::pair<uint32_t, uint32_t>> &ccm, const std::vector<uint32_t> &ccm_weights)
{
    std::vector<uint32_t> connection_indices;
    std::vector<uint32_t> connection_weights;
    for (int i = 0; i < ccm.size(); i++)
    {
        if (ccm[i].first == c_idx)
        {
            connection_indices.push_back(2 * i);
            connection_weights.push_back(ccm_weights[i]);
        }
        if (ccm[i].second == c_idx)
        {
            connection_indices.push_back(2 * i + 1);
            connection_weights.push_back(ccm_weights[i]);
        }
    }
    return std::make_tuple(connection_indices, connection_weights);
}
auto get_related_events(size_t c_idx, const std::vector<std::pair<uint32_t, uint32_t>> &ccm, const std::vector<uint32_t> &ccm_weights, const std::vector<uint32_t> &events)
{
    auto [r_con, rw] = get_related_connections(c_idx, ccm, ccm_weights);
    std::vector<uint32_t> r_con_events(r_con.size(), 0);
    for (int i = 0; i < r_con_events.size(); i++)
    {
        r_con_events[i] = events[r_con[i]];
    }
    return r_con_events;
}
auto get_community_connections(size_t N_communities, const auto &ccm, const auto &ccm_weights)
{
    std::vector<uint32_t> community_indices(N_communities);
    std::iota(community_indices.begin(), community_indices.end(), 0);
    std::vector<std::vector<uint32_t>> indices(community_indices.size());
    std::vector<std::vector<uint32_t>> weights(community_indices.size());

    for (int i = 0; i < community_indices.size(); i++)
    {
        auto [rc, rw] = get_related_connections(community_indices[i], ccm, ccm_weights);
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

auto make_iota(auto N)
{
    std::vector<uint32_t> result(N);
    std::iota(result.begin(), result.end(), 0);
    return result;
}

std::vector<uint32_t> get_related_connection_events(const auto &related_connections, const std::vector<uint32_t> &events)
{
    std::vector<uint32_t> r_con_events(related_connections.size(), 0);
    for (int i = 0; i < r_con_events.size(); i++)
    {
        r_con_events[i] = events[related_connections[i]];
    }
    return r_con_events;
};

std::vector<uint32_t> sample_community(const auto &related_connections, const auto &related_weights, const auto &events, auto N_samples)
{
    auto N_connections = events.size() / 2;
    std::vector<uint32_t> result(2 * N_connections, 0);
    if (!N_samples)
        return result;
    auto r_con_events = get_related_connection_events(related_connections, events);
    auto sample_counts = constrained_weight_sample(N_samples, related_weights, r_con_events);
    for (int sample_idx = 0; sample_idx < sample_counts.size(); sample_idx++)
    {
        result[related_connections[sample_idx]] = sample_counts[sample_idx];
    }
    return result;
}

std::vector<uint32_t> sample_timestep(const auto &events, const auto &delta_I, const auto &ccm, const auto &ccm_weights)
{
    auto N_communities = delta_I.size();
    auto N_connections = events.size() / 2;
    auto merge_sample_result = [&](const std::vector<std::vector<uint32_t>> &sample_result)
    {
        std::vector<uint32_t> merged_result(N_connections * 2, 0);
        for (int i = 0; i < sample_result.size(); i++)
        {
            for (int j = 0; j < sample_result[i].size(); j++)
            {
                merged_result[j] += sample_result[i][j];
            }
        }
        return merged_result;
    };

    std::vector<std::vector<uint32_t>> result(N_communities, std::vector<uint32_t>(N_connections * 2, 0));
    auto community_idx = make_iota(N_communities);
    std::transform(std::execution::par_unseq, community_idx.begin(), community_idx.end(), result.begin(), [&](auto c_idx)
                   {
                    auto [r_con, r_weight] = get_related_connections(c_idx, ccm, ccm_weights);
                    auto dI = delta_I[c_idx];
                    return sample_community(r_con, r_weight, events, dI); });
    auto merged_result = merge_sample_result(result);
    uint32_t merged_infs = std::accumulate(merged_result.begin(), merged_result.end(), 0);
    uint32_t true_infs = std::accumulate(delta_I.begin(), delta_I.end(), 0);
    assert(merged_infs == true_infs && "Sampled infections do not match true infections");
    return merged_result;
}

std::vector<std::vector<uint32_t>> sample_infections(const Dataframe_t<State_t, 2> &community_state, const Dataframe_t<uint32_t, 2> &events, const std::vector<std::pair<uint32_t, uint32_t>> &ccm, const std::vector<uint32_t> &ccm_weights, uint32_t N_communities, uint32_t seed, uint32_t max_infection_samples)
{

    auto N_connections = ccm.size();
    auto delta_Is = get_delta_Is(community_state);
    auto Nt = delta_Is.size();
    std::vector<std::vector<uint32_t>>
        sampled_infections(Nt, std::vector<uint32_t>(N_connections * 2, 0));
    auto t_vec = make_iota(Nt);
    std::transform(std::execution::par_unseq, t_vec.begin(), t_vec.end(), sampled_infections.begin(), [events, delta_Is, ccm, ccm_weights](auto t)
                   { return sample_timestep(events[t], delta_Is[t], ccm, ccm_weights); });

    return sampled_infections;
}

Dataframe_t<uint32_t, 3> sample_infections(const Dataframe_t<State_t, 3> &&community_state, const Dataframe_t<uint32_t, 3> &&from_events, const Dataframe_t<uint32_t, 3> &&to_events, const std::vector<std::pair<uint32_t, uint32_t>> &ccm, const std::vector<uint32_t> &ccm_weights, uint32_t N_communities, uint32_t seed, uint32_t max_infection_samples)
{
    auto N_connections = ccm.size();
    auto N_sims = community_state.N_sims;
    auto Nt = community_state.Nt;
    Dataframe_t<uint32_t, 3> sampled_infections(N_sims, Nt, N_connections);
    for (int i = 0; i < N_sims; i++)
    {
        auto events = zip_merge(from_events[i], to_events[i]);
        sampled_infections[i] = sample_infections(community_state[i], events, ccm, ccm_weights, N_communities, seed, max_infection_samples);
    }
    return sampled_infections;
}

void validate_infection_graphseries(const Dataframe_t<State_t, 4> &community_state, const Dataframe_t<uint32_t, 4> &from_events, const Dataframe_t<uint32_t, 4> &to_events, const auto &ccms, const auto &ccm_weights)
{
    auto Ng = community_state.Ng;
    auto N_sims = community_state.N_sims;
    auto Nt = community_state.Nt;
    for (int g_idx = 0; g_idx < Ng; g_idx++)
    {
        auto N_communities = community_state[g_idx].N_cols;
        auto N_connections = ccms[g_idx].size();
        const auto &ccm = ccms[g_idx];
        const auto &ccm_w = ccm_weights[g_idx];
        for (int sim_idx = 0; sim_idx < N_sims; sim_idx++)
        {
            auto events = zip_merge(from_events[g_idx][sim_idx], to_events[g_idx][sim_idx]);
            auto dIs = get_delta_Is(community_state[g_idx][sim_idx]);
            for (int t = 0; t < Nt - 1; t++)
            {
                for (int c_idx = 0; c_idx < N_communities; c_idx++)
                {
                    auto r_events = get_related_events(c_idx, ccm, ccm_w, events[t]);
                    if (std::accumulate(r_events.begin(), r_events.end(), 0) < dIs[t][c_idx])
                    {
                        throw std::runtime_error("Error: too few events related to community in timeseries, (g_idx, sim_idx, t, c_idx): (" + std::to_string(g_idx) + "," + std::to_string(sim_idx) + "," + std::to_string(t) + "," + std::to_string(c_idx) + ")");
                    }
                }
            }
        }
    }
}

void event_inf_summary(const Dataframe_t<State_t, 4> &community_state, const Dataframe_t<uint32_t, 4> &events, const auto &ccms, const auto &ccm_weights)
{
    auto Ng = community_state.Ng;
    auto N_sims = community_state.N_sims;
    auto Nt = community_state.Nt;
    for (int g_idx = 0; g_idx < Ng; g_idx++)
    {
        std::cout << "Graph " << g_idx << "\n";
        auto N_communities = community_state[g_idx].N_cols;
        auto N_connections = ccms[g_idx].size();
        const auto &ccm = ccms[g_idx];
        const auto &ccm_w = ccm_weights[g_idx];
        for (int sim_idx = 0; sim_idx < N_sims; sim_idx++)
        {
            std::cout << "Simulation " << sim_idx << "\n";
            auto dIs = get_delta_Is(community_state[g_idx][sim_idx]);
            for (int t = 0; t < Nt - 1; t++)
            {
                std::cout << "Timestep " << t << "\n";
                for (int c_idx = 0; c_idx < N_communities; c_idx++)
                {
                    auto [r_idx, r_w] = get_related_connections(c_idx, ccm, ccm_w);
                    auto r_events = get_related_events(c_idx, ccm, ccm_w, events[g_idx][sim_idx][t]);
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

Dataframe_t<uint32_t, 4> sample_infections(const Dataframe_t<State_t, 4>& community_state, const Dataframe_t<uint32_t, 4>& events, const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &ccm, const std::vector<std::vector<uint32_t>> &ccm_weights, uint32_t seed, uint32_t max_infection_samples)
{
    auto N_graphs = community_state.Ng;
    auto N_sims = community_state.N_sims;
    auto Nt = community_state.Nt;
    event_inf_summary(community_state, from_events, to_events, ccm, ccm_weights);
    // validate_infection_graphseries(community_state, from_events, to_events, ccm, ccm_weights);
    Dataframe_t<uint32_t, 4> sampled_infections(N_graphs, N_sims, Nt, ccm.size() * 2);
    for (int i = 0; i < N_graphs; i++)
    {
        sampled_infections[i] = sample_infections(std::forward<const Dataframe_t<State_t, 3>>(community_state[i]),
                                                  std::forward<const Dataframe_t<uint32_t, 3>>(from_events[i]),
                                                  std::forward<const Dataframe_t<uint32_t, 3>>(to_events[i]),
                                                  ccm[i], ccm_weights[i], N_communities[i], seed, max_infection_samples);
    }
    return sampled_infections;
}
