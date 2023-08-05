#include <Sycl_Graph/SIR_Infection_Sampling.hpp>
#include <random>
#include <iostream>
#include <algorithm>
std::vector<uint32_t> sample_connection_infections(Inf_Sample_Data_t &z)
{
    std::mt19937 rng(z.seed);
    std::vector<uint32_t> inf_samples(z.connection_weights.size(), 0);
    std::discrete_distribution<uint32_t> dist(z.connection_weights.begin(), z.connection_weights.end());

    uint32_t N_sampled = 0;
    if (z.N_infected == 0)
        return inf_samples;
    while (N_sampled < z.N_infected)
    {
        uint32_t idx = dist(rng);
        uint32_t connection_idx = z.connection_indices[idx];
        if (inf_samples[idx] < z.connection_events[connection_idx])
        {
            inf_samples[idx]++;
            N_sampled++;
        }
        else
        {
            z.connection_weights[idx] = 0;
        }
    }

    std::cout << "Samples for community " << z.community_idx << ": " << std::endl;
    for (int i = 0; i < inf_samples.size(); i++)
    {
        std::cout << "Index " << z.connection_indices[i] << ": " << inf_samples[i];
    }
    std::cout << std::endl;

    return inf_samples;
}

std::vector<uint32_t> sample_timestep_infections(const std::vector<int> &delta_Is, const std::vector<uint32_t> &from_events, const std::vector<uint32_t> &to_events, const std::vector<uint32_t> &ccm, const std::vector<uint32_t> &ccm_weights, uint32_t N_connections, uint32_t seed)
{
    uint32_t N_communities = delta_Is.size();
    std::mt19937 rng(seed);
    std::vector<uint32_t> seeds(N_communities);
    std::generate(seeds.begin(), seeds.end(), [&rng]()
                  { return rng(); });
    std::cout << "Infections : \n";
    for (int i = 0; i < to_events.size(); i++)
    {
        std::cout << ccm[2 * i] << " <- " << ccm[2 * i + 1] << " : " << from_events[i] << "\n";
        std::cout << ccm[2 * i] << " -> " << ccm[2 * i + 1] << " : " << to_events[i] << "\n";
    }

    std::vector<Inf_Sample_Data_t> zs(N_communities);
    std::vector<uint32_t> community_events(N_communities, 0);
    for (int i = 0; i < zs.size(); i++)
    {
        for (int j = 0; j < ccm.size(); j++)
        {
            if (ccm[j] == i)
            {
                zs[i].connection_indices.push_back(j);
                zs[i].connection_weights.push_back(ccm_weights[j]);
            }
        }
        zs[i].connection_events.resize(from_events.size() * 2);
        for (int j = 0; j < from_events.size(); j++)
        {
            if (ccm[2*j] == i)
            {
                community_events[i] += from_events[j];
            }
            if (ccm[2*j+1] == i)
            {
                community_events[i] += to_events[j];
            }
            zs[i].connection_events[2 * j] = from_events[j];
            zs[i].connection_events[2 * j + 1] = to_events[j];
        }

        zs[i].N_infected = delta_Is[i];
        zs[i].community_idx = i;
        zs[i].seed = seeds[i];
    }

    std::vector<std::vector<uint32_t>> connection_infections(N_communities);

    std::transform(zs.begin(), zs.end(), connection_infections.begin(), [](auto &z)
                   { return sample_connection_infections(z); });
    std::vector<uint32_t> merged_infections(N_connections);
    for (int i = 0; i < N_connections; i++)
    {
        for (int j = 0; j < N_communities; j++)
        {
            merged_infections[2 * i] += connection_infections[j][2 * i];
            merged_infections[2 * i + 1] += connection_infections[j][2 * i + 1];
        }
    }
    return merged_infections;
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

    assert(std::all_of(delta_I.begin(), delta_I.end(), [](auto &v)
                       { return std::all_of(v.begin(), v.end(), [](auto &x)
                                            { return x >= 0; }); }));
    assert(std::all_of(delta_R.begin(), delta_R.end(), [](auto &v)
                       { return std::all_of(v.begin(), v.end(), [](auto &x)
                                            { return x >= 0; }); }));

    return delta_I;
}

std::vector<std::vector<uint32_t>> sample_infections(const std::vector<std::vector<State_t>> &community_state, const std::vector<std::vector<uint32_t>> &from_events, const std::vector<std::vector<uint32_t>> &to_events, const std::vector<uint32_t> &ccm, const std::vector<uint32_t> &ccm_weights, uint32_t seed)
{
    std::vector<uint32_t> community_idx(community_state.size());
    std::iota(community_idx.begin(), community_idx.end(), 0);

    std::vector<std::vector<int>> delta_Is = get_delta_Is(community_state);

    std::vector<uint32_t> I_sample_seeds(delta_Is.size());
    std::mt19937 rng(seed);
    std::generate(I_sample_seeds.begin(), I_sample_seeds.end(), [&rng]()
                  { return rng(); });

    uint32_t N_connections = from_events[0].size();

    std::vector<std::tuple<std::vector<int>, std::vector<uint32_t>, std::vector<uint32_t>, uint32_t>> zip(delta_Is.size());
    for (int i = 0; i < zip.size(); i++)
    {
        zip[i] = std::make_tuple(delta_Is[i], from_events[i], to_events[i], I_sample_seeds[i]);
    }

    std::vector<std::vector<uint32_t>> sampled_infections(delta_Is.size());

    std::transform(zip.begin(), zip.end(), sampled_infections.begin(), [ccm, ccm_weights, N_connections](const auto &z)
                   { return sample_timestep_infections(std::get<0>(z), std::get<1>(z), std::get<2>(z), ccm, ccm_weights, N_connections, std::get<3>(z)); });

    return sampled_infections;
}
