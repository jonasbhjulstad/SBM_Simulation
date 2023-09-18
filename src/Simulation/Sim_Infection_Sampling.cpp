
#include <Sycl_Graph/Simulation/Sim_Infection_Sampling.hpp>
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
#include <algorithm>
#include <iostream>
#include <random>
#include <execution>
#include <Sycl_Graph/Simulation/Sim_Types.hpp>
#define INF_MAX_SAMPLE_LIMIT 10000

std::vector<uint32_t> constrained_weight_sample(size_t N_samples, const std::vector<uint32_t> weights, const std::vector<uint32_t>& max_values)
{
    if (N_samples == 0)
    {
        return std::vector<uint32_t>(weights.size(), 0);
    }
    std::vector<uint32_t> sample_counts(weights.size(), 0);
    std::discrete_distribution<uint32_t> dist(weights.begin(), weights.end());
    std::mt19937 rng(std::random_device{}());
    uint32_t N_sampled = 0;
    for(int i = 0; i < INF_MAX_SAMPLE_LIMIT;  i++)
    {
        auto idx = dist(rng);
        if(sample_counts[idx] < max_values[idx])
        {
            sample_counts[idx]++;
            N_sampled++;
        }
        if(N_sampled >= N_samples)
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


    for(int i = 0; i < delta_I.size();i++)
    {
        assert(std::all_of(delta_I[i].begin(), delta_I[i].end(), [](auto x){return x >= 0;}) && "Negative delta_I");
    }

    for(int i = 0; i < delta_R.size();i++)
    {
        assert(std::all_of(delta_R[i].begin(), delta_R[i].end(), [](auto x){return x >= 0;}) && "Negative delta_R");
    }

    return delta_I;
}



auto get_related_connections(size_t c_idx, const std::vector<std::pair<uint32_t, uint32_t>>& ccm, const std::vector<uint32_t>& ccm_weights)
{
    std::vector<uint32_t> connection_indices;
    std::vector<uint32_t> connection_weights;
    for(int i = 0; i < ccm.size(); i++)
    {
        if (ccm[i].first == c_idx)
        {
            connection_indices.push_back(2*i);
            connection_weights.push_back(ccm_weights[i]);
        }
        if(ccm[i].second == c_idx)
        {
            connection_indices.push_back(2*i + 1);
            connection_weights.push_back(ccm_weights[i]);
        }
    }
    return std::make_tuple(connection_indices, connection_weights);
}

auto zip_merge(const std::vector<uint32_t>& v0, const std::vector<uint32_t>& v1)
{
    std::vector<uint32_t> result(v0.size() + v1.size());
    for(int i = 0; i < v0.size(); i++)
    {
        result[2*i] = v0[i];
        result[2*i + 1] = v1[i];
    }
    return result;
}

auto zip_merge(const Timeseries_t<uint32_t>& ts0, const Timeseries_t<uint32_t>& ts1)
{
    auto Nt = ts0.Nt;
    auto Nc = ts0.N_cols;
    Timeseries_t<uint32_t> result(Nt, Nc*2);
    for(int t = 0; t < Nt; t++)
    {
        for(int c = 0; c < Nc; c++)
        {
            result[t][2*c] = ts0[t][c];
            result[t][2*c + 1] = ts1[t][c];
        }
    }
    return result;
}


std::vector<std::vector<uint32_t>> sample_infections(const Timeseries_t<State_t> &community_state, const Timeseries_t<uint32_t> &from_events, const Timeseries_t<uint32_t> &to_events, const std::vector<std::pair<uint32_t, uint32_t>> &ccm, const std::vector<uint32_t> &ccm_weights, uint32_t N_communities, uint32_t seed, uint32_t max_infection_samples)
{

    auto N_connections = ccm.size();
    auto delta_Is = get_delta_Is(community_state);
    auto events = zip_merge(from_events, to_events);
    auto Nt = delta_Is.size();
    std::vector<std::vector<uint32_t>> related_connections(N_communities);
    std::vector<std::vector<uint32_t>> related_weights(N_communities);
    for(int i = 0; i < N_communities; i++)
    {
        auto [rc, rw] = get_related_connections(i, ccm, ccm_weights);
        related_connections[i] = rc;
        related_weights[i] = rw;
    }

    auto sample_community = [N_connections, related_connections, related_weights](const auto& t_events, auto N_samples, uint32_t c_idx)
    {
        std::vector<uint32_t> max_values(related_connections[c_idx].size(), 0);
        for(int i = 0; i < max_values.size(); i++)
        {
            max_values[i] = t_events[related_connections[c_idx][i]];
        }
        auto mv_sum = std::accumulate(max_values.begin(), max_values.end(), 0);
        assert(mv_sum >= N_samples && "Not enough connection events to make up for infection samples");
        auto sample_counts = constrained_weight_sample(N_samples, related_weights[c_idx], max_values);
        std::vector<uint32_t> result(N_connections*2, 0);
        for(int sample_idx = 0; sample_idx < sample_counts.size(); sample_idx++)
        {
            result[related_connections[c_idx][sample_idx]] = sample_counts[sample_idx];
        }
        return result;
    };



    auto sample_timestep = [N_connections, related_connections, related_weights, N_communities, sample_community](const auto& t_events, const auto& delta_I_ts){
        std::vector<std::vector<uint32_t>> result(N_communities, std::vector<uint32_t>(N_connections*2, 0));
        std::vector<uint32_t> community_idx(N_communities);
        std::iota(community_idx.begin(), community_idx.end(), 0);
        auto event_sum = std::accumulate(t_events.begin(), t_events.end(), 0);
        auto delta_I_sum = std::accumulate(delta_I_ts.begin(), delta_I_ts.end(), 0);
        assert(event_sum == delta_I_sum && "Event sum does not match delta_I sum");
        std::transform(std::execution::par_unseq, community_idx.begin(), community_idx.end(), result.begin(), [t_events, delta_I_ts, sample_community](auto c_idx){return sample_community(t_events, delta_I_ts[c_idx], c_idx);});
        std::vector<uint32_t> merged_result(N_connections*2, 0);
        for(int i = 0; i < result.size(); i++)
        {
            for(int j = 0; j < result[i].size(); j++)
            {
                merged_result[j] += result[i][j];
            }
        }
        uint32_t merged_infs = std::accumulate(merged_result.begin(), merged_result.end(), 0);
        uint32_t true_infs = std::accumulate(delta_I_ts.begin(), delta_I_ts.end(), 0);
        assert(merged_infs == true_infs && "Sampled infections do not match true infections");
        return merged_result;
    };



    std::vector<std::vector<uint32_t>> sampled_infections(Nt, std::vector<uint32_t>(N_connections*2, 0));
    std::vector<uint32_t> t_vec(Nt);
    std::iota(t_vec.begin(), t_vec.end(), 0);
    std::transform(std::execution::par_unseq, t_vec.begin(), t_vec.end(), sampled_infections.begin(), [events, delta_Is, sample_timestep](auto t){return sample_timestep(events[t], delta_Is[t]);});

    return sampled_infections;
}

Simseries_t<uint32_t> sample_infections(const Simseries_t<State_t> &&community_state,const Simseries_t<uint32_t> &&from_events, const Simseries_t<uint32_t> &&to_events, const std::vector<std::pair<uint32_t, uint32_t>> &ccm, const std::vector<uint32_t> &ccm_weights, uint32_t N_communities, uint32_t seed, uint32_t max_infection_samples)
{
    auto N_connections = ccm.size();
    auto N_sims = community_state.N_sims;
    auto Nt = community_state.Nt;
    Simseries_t<uint32_t> sampled_infections(N_sims, Nt, N_connections);
    for (int i = 0; i < N_sims; i++)
    {
        sampled_infections[i] = sample_infections(community_state[i], from_events[i], to_events[i], ccm, ccm_weights, N_communities, seed, max_infection_samples);
    }
    return sampled_infections;
}

Graphseries_t<uint32_t> sample_infections(const Graphseries_t<State_t>&&community_state, const Graphseries_t<uint32_t>&& from_events, const Graphseries_t<uint32_t>&& to_events, const std::vector<std::vector<std::pair<uint32_t, uint32_t>>>& ccm, const std::vector<std::vector<uint32_t>>& ccm_weights, const std::vector<uint32_t>& N_communities, uint32_t seed, uint32_t max_infection_samples)
{
    auto N_graphs = community_state.Ng;
    auto N_sims = community_state.N_sims;
    auto Nt = community_state.Nt;
    Graphseries_t<uint32_t> sampled_infections(N_graphs, N_sims, Nt, ccm.size()*2);
    for(int i = 0; i < N_graphs; i++)
    {
        sampled_infections[i] = sample_infections(std::forward<const Simseries_t<State_t>>(community_state[i]),
        std::forward<const Simseries_t<uint32_t>>(from_events[i]),
        std::forward<const Simseries_t<uint32_t>>(to_events[i]),
        ccm[i], ccm_weights[i], N_communities[i], seed, max_infection_samples);
    }
    return sampled_infections;
}
