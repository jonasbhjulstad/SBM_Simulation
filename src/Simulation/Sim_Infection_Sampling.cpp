
#include <Sycl_Graph/Simulation/Sim_Infection_Sampling.hpp>
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
#include <algorithm>
#include <iostream>
#include <random>
#include <execution>

std::vector<uint32_t> sample_connection_infections(Inf_Sample_Data_t &z, const uint32_t max_infection_samples)
{
    std::mt19937 rng(z.seed);
    std::vector<uint32_t> inf_samples(z.weights.size(), 0);
    std::discrete_distribution<uint32_t> dist(z.weights.begin(), z.weights.end());

    uint32_t N_sampled = 0;
    if (z.N_infected == 0)
        return inf_samples;
    uint32_t N_attempts = 0;
    while (N_sampled < z.N_infected)
    {
        uint32_t idx = dist(rng);
        uint32_t connection_idx = z.indices[idx];
        if (inf_samples[idx] < z.events[connection_idx])
        {
            inf_samples[idx]++;
            N_sampled++;
        }
        else
        {
            z.weights[idx] = 0;
        }
        N_attempts++;
        if (N_attempts > max_infection_samples)
        {
            std::cout << "Warning: max infection samples exceeded" << std::endl;
            std::cout << "N_infected: " << z.N_infected << std::endl;
            std::cout << "Events: ";
            for (int i = 0; i < z.events.size(); i++)
            {
                std::cout << z.events[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "Indices: ";
            for (int i = 0; i < z.indices.size(); i++)
            {
                std::cout << z.indices[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "Weights: ";
            for (int i = 0; i < z.weights.size(); i++)
            {
                std::cout << z.weights[i] << " ";
            }
            std::cout << std::endl;
            break;
        }
    }

#ifdef SIR_INFECT_SAMPLE_DEBUG
    std::cout << "Samples for community " << z.community_idx << ": " << std::endl;
    for (int i = 0; i < inf_samples.size(); i++)
    {
        std::cout << "Index " << z.indices[i] << ": " << inf_samples[i];
    }
    std::cout << std::endl;
#endif

    return inf_samples;
}

std::vector<uint32_t> dupe_vec(const std::vector<uint32_t> &vec)
{
    std::vector<uint32_t> res(vec.size() * 2);
    for (int i = 0; i < vec.size(); i++)
    {
        res[2 * i] = vec[i];
        res[2 * i + 1] = vec[i];
    }
    return res;
}

std::vector<uint32_t> events_combine(const std::vector<uint32_t> &from_events, const std::vector<uint32_t> &to_events)
{
    std::vector<uint32_t> comb(from_events.size() + to_events.size());
    for (int i = 0; i < from_events.size(); i++)
    {
        comb[2 * i] = from_events[i];
        comb[2 * i + 1] = to_events[i];
    }
    return comb;
}
std::vector<std::vector<uint32_t>> events_combine(const std::vector<std::vector<uint32_t>> &from_events, const std::vector<std::vector<uint32_t>> &to_events)
{
    std::vector<std::vector<uint32_t>> comb(from_events.size(), std::vector<uint32_t>(from_events[0].size() * 2));
    std::transform(from_events.begin(), from_events.end(), to_events.begin(), comb.begin(), [&](const auto &from, const auto &to)
                   { return events_combine(from, to); });
    return comb;
}

std::vector<uint32_t> sample_timestep_infections(const std::vector<int> &delta_Is, const std::vector<uint32_t> &from_events, const std::vector<uint32_t> &to_events, const std::vector<std::pair<uint32_t, uint32_t>> &ccm, const std::vector<uint32_t> &ccm_weights, uint32_t N_connections, uint32_t seed, uint32_t max_infection_samples)
{
    uint32_t N_communities = delta_Is.size();
    std::mt19937 rng(seed);
    std::vector<uint32_t> seeds(N_communities);
    std::generate(seeds.begin(), seeds.end(), [&rng]()
                  { return rng(); });
#ifdef SIR_INFECT_SAMPLE_DEBUG
    std::cout << "Infection-Events: \n";
    for (int i = 0; i < ccm.size(); i++)
    {
        std::cout << ccm[i].first << " -> " << ccm[i].second << " : " << from_events[i] << "\n";
    }
#endif
    auto ccm_flat = pairlist_to_vec(ccm);
    auto duped_ccm_weights = dupe_vec(ccm_weights);

    std::vector<Inf_Sample_Data_t> zs(N_communities);
    auto flat_events = events_combine(from_events, to_events);
    std::vector<uint32_t> community_events(N_communities, 0);
    for (int i = 0; i < zs.size(); i++)
    {
        zs[i].events = flat_events;
        for (int j = 0; j < ccm.size(); j++)
        {

            if (ccm_flat[2 * j] == i)
            {
                zs[i].indices.push_back(2 * j);
                zs[i].weights.push_back(duped_ccm_weights[2 * j]);
            }
            if (ccm_flat[2 * j + 1] == i)
            {
                zs[i].indices.push_back(2 * j + 1);
                zs[i].weights.push_back(duped_ccm_weights[2 * j + 1]);
            }
        }

        zs[i].N_infected = delta_Is[i];
        zs[i].community_idx = i;
        zs[i].seed = seeds[i];
    }

#ifdef SIR_INFECT_SAMPLE_DEBUG
    std::cout << "Community Infections/Events: \n";
    for (int i = 0; i < community_events.size(); i++)
    {
        std::cout << "Community " << i << ": " << zs[i].N_infected << ", " << community_events[i] << "\n";
    }
#endif

    std::vector<std::vector<uint32_t>> connection_infections(N_communities);

    std::transform(zs.begin(), zs.end(), connection_infections.begin(), [max_infection_samples](auto &z)
                   { return sample_connection_infections(z, max_infection_samples); });
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

std::vector<std::vector<uint32_t>> sample_infections(const std::vector<std::vector<State_t>> &community_state, const std::vector<std::vector<uint32_t>> &from_events, const std::vector<std::vector<uint32_t>> &to_events, const std::vector<std::pair<uint32_t, uint32_t>> &ccm, const std::vector<uint32_t> &ccm_weights, uint32_t seed, uint32_t max_infection_samples)
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

    std::transform(std::execution::par_unseq, zip.begin(), zip.end(), sampled_infections.begin(), [ccm, ccm_weights, N_connections, max_infection_samples](const auto &z)
                   { return sample_timestep_infections(std::get<0>(z), std::get<1>(z), std::get<2>(z), ccm, ccm_weights, N_connections, std::get<3>(z), max_infection_samples); });

    return sampled_infections;
}


std::vector<std::vector<std::vector<uint32_t>>> sample_from_connection_events(const std::vector<std::vector<std::vector<State_t>>> &community_state,
                                                                                         const std::vector<std::vector<std::vector<uint32_t>>> &from_events,
                                                                                         const std::vector<std::vector<std::vector<uint32_t>>> &to_events,
                                                                                        const std::vector<std::pair<uint32_t, uint32_t>>& ccm,
                                                                                        const std::vector<uint32_t>& ccm_weights,
                                                                                        uint32_t seed, uint32_t max_infection_samples)
{

    uint32_t N_sims = from_events.size();
    uint32_t Nt = from_events[0].size();
    uint32_t N_connections = from_events[0][0].size();
    std::vector<std::vector<std::vector<uint32_t>>> infections = std::vector<std::vector<std::vector<uint32_t>>>(N_sims, std::vector<std::vector<uint32_t>>(N_connections, std::vector<uint32_t>(Nt)));

    std::vector<std::tuple<std::vector<std::vector<State_t>>, std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>>> sim_data(N_sims);

    for (int sim_idx = 0; sim_idx < N_sims; sim_idx++)
    {
        sim_data[sim_idx] = std::make_tuple(community_state[sim_idx], from_events[sim_idx], to_events[sim_idx]);
    }
    std::transform(std::execution::par_unseq, sim_data.begin(), sim_data.end(), infections.begin(), [&](const auto &sim_d)
                   {
        const auto& sim_community_state = std::get<0>(sim_d);
        const auto& sim_from_events = std::get<1>(sim_d);
        const auto& sim_to_events = std::get<2>(sim_d);
        return sample_infections(sim_community_state, sim_from_events, sim_to_events, ccm, ccm_weights, seed, max_infection_samples); });

    return infections;
}

uint32_t floor_div(uint32_t a, uint32_t b)
{
    return static_cast<uint32_t>(std::floor(static_cast<double>(a) / static_cast<double>(b)));
}

std::vector<std::vector<std::vector<uint32_t>>> sample_from_connection_events(const std::vector<std::vector<std::vector<State_t>>> &community_state,
                                                                                         const std::vector<std::vector<std::vector<uint32_t>>> &from_events,
                                                                                         const std::vector<std::vector<std::vector<uint32_t>>> &to_events,
                                                                                        const std::vector<std::vector<std::pair<uint32_t, uint32_t>>>& ccms,
                                                                                        const std::vector<std::vector<uint32_t>>& ccm_weights,
                                                                                        uint32_t seed, uint32_t max_infection_samples, uint32_t N_compute)
{

    uint32_t N_sims = from_events.size();
    uint32_t Nt = from_events[0].size();
    uint32_t N_connections = from_events[0][0].size();
    std::vector<std::vector<std::vector<uint32_t>>> infections = std::vector<std::vector<std::vector<uint32_t>>>(N_sims, std::vector<std::vector<uint32_t>>(N_connections, std::vector<uint32_t>(Nt)));

    std::vector<std::tuple<std::vector<std::vector<State_t>>, std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>, uint32_t>> sim_data(N_sims);


    for (int sim_idx = 0; sim_idx < N_sims; sim_idx++)
    {
        sim_data[sim_idx] = std::make_tuple(community_state[sim_idx], from_events[sim_idx], to_events[sim_idx], floor_div(sim_idx, N_compute));
    }
    std::transform(std::execution::par_unseq, sim_data.begin(), sim_data.end(), infections.begin(), [&](const auto &sim_d)
                   {
        const auto& sim_community_state = std::get<0>(sim_d);
        const auto& sim_from_events = std::get<1>(sim_d);
        const auto& sim_to_events = std::get<2>(sim_d);
        const auto& ccm_idx = std::get<3>(sim_d);
        return sample_infections(sim_community_state, sim_from_events, sim_to_events, ccms[ccm_idx], ccm_weights[ccm_idx], seed, max_infection_samples); });

    return infections;
}
