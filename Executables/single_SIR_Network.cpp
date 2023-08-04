#define TBB_DEBUG 1
#include <CL/sycl.hpp>
#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/SBM_Generation.hpp>
// #include <Sycl_Graph/SBM_write.hpp>
// #include <Sycl_Graph/SIR_SBM_Network.hpp>
#include <Sycl_Graph/path_config.hpp>
#include <algorithm>
#include <cstdint>
#include <execution>
#include <filesystem>
#include <iostream>
#include <string>

using namespace Sycl_Graph::SBM;

std::tuple<uint32_t, uint32_t> get_susceptible_id_if_infected(const auto &v_acc, uint32_t id_from,
                                                              uint32_t id_to)
{
    if (((v_acc[id_to] == SIR_INDIVIDUAL_S) &&
         (v_acc[id_from] == SIR_INDIVIDUAL_I)))
    {
        return std::make_tuple(id_to, 1);
    }
    else if (((v_acc[id_from] == SIR_INDIVIDUAL_S) &&
              (v_acc[id_to] == SIR_INDIVIDUAL_I)))
    {
        return std::make_tuple(id_from, 0);
    }
    else
    {
        return std::make_tuple(std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max());
    }
}
void linewrite(std::ofstream &file, const std::vector<uint32_t> &state_iter)
{
    for (const auto &t_i_i : state_iter)
    {
        file << t_i_i;
        if (&t_i_i != &state_iter.back())
            file << ",";
        else
            file << "\n";
    }
}

void linewrite(std::ofstream &file, const std::vector<float> &val)
{
    for (const auto &t_i_i : val)
    {
        file << t_i_i;
        if (&t_i_i != &val.back())
            file << ",";
        else
            file << "\n";
    }
}

void linewrite(std::ofstream &file,
               const std::vector<std::array<uint32_t, 3>> &state_iter)
{
    for (const auto &t_i : state_iter)
    {
        for (const auto &t_i_i : t_i)
        {
            file << t_i_i;
            file << ",";
        }
    }
    file << "\n";
}

void linewrite(std::ofstream &file, const std::vector<Edge_t> &iter)
{
    for (auto &t_i_i : iter)
    {
        file << t_i_i.from << "," << t_i_i.to;
        if (&t_i_i != &iter.back())
            file << ",";
        else
            file << "\n";
    }
}

void columnwrite(std::ofstream &file, const std::vector<Edge_t> &iter)
{
    for (auto &t_i_i : iter)
    {
        file << t_i_i.from << "," << t_i_i.to << "\n";
    }
}

template <typename T>
std::tuple<sycl::buffer<T, 1>, sycl::event> buffer_create_1D(sycl::queue &q, const std::vector<T> &data)
{
    sycl::buffer<T> tmp(data.data(), data.size());
    sycl::buffer<T> result(sycl::range<1>(data.size()));

    auto event = q.submit([&](sycl::handler &h)
                          {
                auto tmp_acc = tmp.template get_access<sycl::access::mode::read>(h);
                auto res_acc = result.template get_access<sycl::access::mode::write>(h);
                h.copy(tmp_acc, res_acc); });
    return std::make_tuple(result, event);
}

template <typename T>
std::tuple<sycl::buffer<T, 2>, sycl::event> buffer_create_2D(sycl::queue &q, const std::vector<std::vector<T>> &data)
{
    assert(std::all_of(data.begin(), data.end(), [&](const auto subdata)
                       { return subdata.size() == data[0].size(); }));

    std::vector<T> data_flat(data.size() * data[0].size());
    for (uint32_t i = 0; i < data.size(); ++i)
    {
        std::copy(data[i].begin(), data[i].end(), data_flat.begin() + i * data[0].size());
    }

    sycl::buffer<T, 2> tmp(data_flat.data(), sycl::range<2>(data.size(), data[0].size()));
    sycl::buffer<T, 2> result(sycl::range<2>(data.size(), data[0].size()));
    auto event = q.submit([&](sycl::handler &h)
                          {
        auto tmp_acc = tmp.template get_access<sycl::access::mode::read>(h);
        auto res_acc = result.template get_access<sycl::access::mode::write>(h);
        h.parallel_for(result.get_range(), [=](sycl::id<2> idx)
                       { res_acc[idx] = tmp_acc[idx]; }); });

    return std::make_tuple(result, event);
}
sycl::buffer<uint32_t> generate_seeds(sycl::queue &q, uint32_t N_rng,
                                      uint32_t seed)
{
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> dis(0, 1000000);
    std::vector<uint32_t> rngs(N_rng);
    std::generate(rngs.begin(), rngs.end(), [&]()
                  { return dis(gen); });

    sycl::buffer<uint32_t> tmp(rngs.data(), rngs.size());
    sycl::buffer<uint32_t> result(sycl::range<1>(rngs.size()));

    q.submit([&](sycl::handler &h)
             {
        auto tmp_acc = tmp.get_access<sycl::access::mode::read>(h);
        auto res_acc = result.get_access<sycl::access::mode::write>(h);

        h.parallel_for(result.get_range(), [=](sycl::id<1> idx)
                       { res_acc[idx] = tmp_acc[idx]; }); });

    return result;
}

sycl::event initialize_vertices(float p_I0, float p_R0, sycl::queue &q,
                                sycl::buffer<SIR_State, 2> &buf, auto &seed_buf, auto event)
{
    uint32_t N_vertices = buf.get_range()[1];
    uint32_t N_seeds = seed_buf.get_range()[0];
    auto N_threads = std::min({N_vertices, N_seeds});
    uint32_t N_vertices_per_thread = N_vertices / N_threads + 1;
    q.submit([&](sycl::handler &h)
             {
                h.depends_on(event);
    auto state_acc = buf.template get_access<sycl::access::mode::write>(h);
    h.parallel_for(buf.get_range(),
                   [=](sycl::id<2> id) { state_acc[id] = SIR_INDIVIDUAL_S; }); });

    return q.submit([&](sycl::handler &h)
                    {
    auto state_acc = buf.template get_access<sycl::access::mode::write>(h);
    auto seed_acc =
        seed_buf.template get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(N_threads, [=](sycl::id<1> id) {

        for(int i = 0; i < N_vertices_per_thread; i++)
        {
            auto idx = id * N_vertices_per_thread + i;
            if (idx >= N_vertices)
                break;
            Static_RNG::default_rng rng(seed_acc[id]);
            seed_acc[idx]++;
            Static_RNG::bernoulli_distribution<float> bernoulli_I(p_I0);
            Static_RNG::bernoulli_distribution<float> bernoulli_R(p_R0);

            if (bernoulli_I(rng)) {
                state_acc[0][idx] = SIR_INDIVIDUAL_I;
            } else if (bernoulli_R(rng)) {
                state_acc[0][idx] = SIR_INDIVIDUAL_R;
            } else {
                state_acc[0][idx] = SIR_INDIVIDUAL_S;
            }
        }
    }); });
}

sycl::event recover(sycl::queue &q, uint32_t t, sycl::event &dep_event, float p_R, auto &seed_buf, auto &trajectory, auto &vcm_buf)
{
    uint32_t N_vertices = trajectory.get_range()[1];
    uint32_t N_seeds = seed_buf.size();
    sycl::buffer<bool> rec_buf(N_vertices);
    uint32_t N_threads = std::min({N_vertices, N_seeds});
    uint32_t N_vertex_per_thread = N_vertices / N_threads + 1;
    auto event = q.submit([&](sycl::handler &h)
                          {
    h.depends_on(dep_event);
    auto seed_acc =
        seed_buf.template get_access<sycl::access_mode::atomic>(h);
    auto v_acc =
        trajectory.template get_access<sycl::access::mode::read_write>(h);
    //   sycl::stream out(1024, 256, h);
    h.parallel_for(N_threads, [=](sycl::id<1> id) {
        // uint32_t seed = sycl::atomic_fetch_add<uint32_t>(seed_acc[id], 1);
        uint32_t seed = 0;
        Static_RNG::default_rng rng(seed);
        for(int i = 0; i < N_vertex_per_thread; i++)
        {
            auto v_idx = N_vertex_per_thread*id[0] + i;
            if(v_idx >= N_vertices)
                break;
      auto state_prev = v_acc[t][v_idx];
      v_acc[t + 1][v_idx] = state_prev;
      if (state_prev == SIR_INDIVIDUAL_I) {
        Static_RNG::bernoulli_distribution<float> bernoulli_R(p_R);
        if (bernoulli_R(rng)) {
          v_acc[t + 1][v_idx] = SIR_INDIVIDUAL_R;
        }
      }
        }
    }); });
    return event;
}

sycl::event infect(sycl::queue &q, auto &ecm_buf, auto &p_I_buf, auto &seed_buf, auto &connection_events_buf, auto &trajectory, auto &edges, uint32_t N_wg, uint32_t t, uint32_t N_connections, auto dep_event)
{
    uint32_t N_edges = ecm_buf.size();
    std::vector<uint32_t> infection_indices_init(N_edges, std::numeric_limits<uint32_t>::max());
    auto [infection_indices_buf, ii_event] = buffer_create_1D(q, infection_indices_init);
    auto inf_event = q.submit([&](sycl::handler &h)

                              {
                                  h.depends_on(ii_event);
    h.depends_on(dep_event);
    auto ecm_acc = ecm_buf.template get_access<sycl::access::mode::read>(h);
    auto p_I_acc = p_I_buf.template get_access<sycl::access::mode::read>(h);
    auto seed_acc =
        seed_buf.template get_access<sycl::access_mode::atomic>(h);
    auto v_acc =
        trajectory.template get_access<sycl::access::mode::read_write>(h);
    auto edge_acc = edges.template get_access<sycl::access::mode::read>(h);
    auto infection_indices_acc = infection_indices_buf.template get_access<sycl::access::mode::write>(h);
    h.parallel_for(N_wg, [=](sycl::id<1> id) {
      auto seed = sycl::atomic_fetch_add<uint32_t>(seed_acc[id], 1);
      uint32_t N_edge_per_wg = (N_edges / N_wg) + 1;
      for(int i = 0; i < N_edge_per_wg; i++)
      {
        auto edge_idx = id * N_edge_per_wg + i;
        if (edge_idx >= N_edges)
          break;
      auto id_from = edge_acc[edge_idx].from;
      auto id_to = edge_acc[edge_idx].to;
      auto [sus_id, direction] = get_susceptible_id_if_infected(v_acc[t], id_from, id_to);
      if (sus_id != std::numeric_limits<uint32_t>::max()) {
        Static_RNG::default_rng rng(seed);
        auto p_I = p_I_acc[t][ecm_acc[edge_idx]];
        Static_RNG::bernoulli_distribution<float> bernoulli_I(p_I);
        if (bernoulli_I(rng)) {
          v_acc[t + 1][sus_id] = SIR_INDIVIDUAL_I;
            infection_indices_acc[edge_idx] = direction;
        }
      }
      }
    }); });

    auto accumulate_event = q.submit([&](sycl::handler &h)
                                     {
                                        auto infection_indices_acc = infection_indices_buf.template get_access<sycl::access::mode::read>(h);
                                        auto events_acc = connection_events_buf.template get_access<sycl::access::mode::read_write>(h);
                                        auto ecm_acc = ecm_buf.template get_access<sycl::access::mode::read>(h);
                                        h.parallel_for(N_connections, [=](sycl::id<1> id)
                                                       {
                                                        events_acc[t][id].from = 0;
                                                        events_acc[t][id].to = 0;
                                                        for(int i = 0; i < N_edges; i++)
                                                        {
                                                            if (infection_indices_acc[i] != std::numeric_limits<uint32_t>::max())
                                                            {
                                                                if (infection_indices_acc[i] == 0)
                                                                    {
                                                                        events_acc[t][id].from++;
                                                                    }
                                                                else
                                                                    {
                                                                        events_acc[t][id].to++;
                                                                    }
                                                            }
                                                        }
                                                       }); });
    return accumulate_event;
}

template <typename T>
std::vector<std::vector<T>> read_buffer(sycl::queue &q, sycl::buffer<T, 2> &buf,
                           auto events = {})
{

    auto range = buf.get_range();
    auto rows = range[0];
    auto cols = range[1];

    std::vector<T> data(cols * rows);
    T *p_data = data.data();

    q.submit([&](sycl::handler &h)
             {
        //create accessor
        h.depends_on(events);
        auto acc = buf.template get_access<sycl::access::mode::read>(h);
        h.copy(acc, p_data); })
        .wait();

    //transform to 2D vector
    std::vector<std::vector<T>> data_2d(rows);
    for (int i = 0; i < rows; i++)
    {
        data_2d[i] = std::vector<T>(cols);
        for (int j = 0; j < cols; j++)
        {
            data_2d[i][j] = data[i * cols + j];
        }
    }

    return data_2d;
}

std::vector<Edge_t> sample_connection_infections(
    uint32_t community_idx, uint32_t N_infected,
    const std::vector<Edge_t> &connection_events, const auto &ccm,
    uint32_t seed)
{
    if (N_infected == 0)
        return std::vector<Edge_t>(ccm.size() * 2, Edge_t(0, 0, 0));
    std::mt19937 rng(seed);

    std::vector<uint32_t> flattened_connection_events(2 * connection_events.size());
    for (int i = 0; i < connection_events.size(); i++)
    {
        flattened_connection_events[2 * i] = connection_events[i].from;
        flattened_connection_events[2 * i + 1] = connection_events[i].to;
    }

    std::vector<uint32_t> connection_weights(ccm.size() * 2, 0);

    for (int i = 0; i < ccm.size(); i++)
    {
        if ((ccm[i].from == community_idx) && flattened_connection_events[2 * i])
        {
            connection_weights[2 * i] = ccm[i].weight;
        }
        if ((ccm[i].to == community_idx) && flattened_connection_events[2 * i + 1])
        {
            connection_weights[2 * i + 1] = ccm[i].weight;
        }
    }

    if (std::all_of(connection_weights.begin(), connection_weights.end(), [](auto &w)
                    { return w == 0; }))
    {
        return std::vector<Edge_t>(ccm.size() * 2, Edge_t(0, 0, 0));
    }

    std::discrete_distribution<uint32_t> discrete_dist(connection_weights.begin(), connection_weights.end());
    uint32_t N_samples = 0;

    std::vector<uint32_t> sampled_infections(2 * ccm.size(), 0);

    while (N_samples < N_infected)
    {
        auto idx = discrete_dist(rng);
        if (sampled_infections[idx] >= flattened_connection_events[idx])
        {
            connection_weights[idx] = 0;
            continue;
        }
        else
        {
            sampled_infections[idx]++;
            N_samples++;
        }
    }

    std::vector<Edge_t> result(ccm.size() * 2, Edge_t(0, 0, 0));
    // transform sampled_infections into result
    for (int i = 0; i < ccm.size(); i++)
    {
        result[i].from = sampled_infections[2 * i];
        result[i].to = sampled_infections[2 * i + 1];
    }

    return result;
}

struct Inf_Sample_Data_t
{
    uint32_t community_idx;
    uint32_t N_infected;
    uint32_t seed;
    std::vector<Edge_t> connection_events;

    static std::vector<Inf_Sample_Data_t> make(const auto &c_idx, const auto &N_infs, const auto &seeds, const auto &c_events)
    {
        std::vector<Inf_Sample_Data_t> res(c_idx.size());
        for (int i = 0; i < c_idx.size(); i++)
        {
            res[i].community_idx = c_idx[i];
            res[i].N_infected = N_infs[i];
            res[i].seed = seeds[i];
            res[i].connection_events = c_events[i];
        }
        return res;
    }
};

std::vector<std::vector<Edge_t>> sample_connection_infections(
    const std::vector<std::vector<State_t>> &community_trajectory,
    const std::vector<std::vector<Edge_t>> &connection_events,
    const std::vector<std::vector<uint32_t>> &community_recoveries,
    const auto &ccm,
    uint32_t seed)
{
    uint32_t N_communities = community_trajectory[1].size();
    uint32_t Nt = community_trajectory.size() - 1;
    uint32_t N_connections = connection_events[0].size();

    std::vector<std::vector<Edge_t>> connection_infections(Nt);
    std::vector<std::vector<int>> delta_Is(
        Nt, std::vector<int>(N_communities, 0));

    for (int i = 0; i < Nt; i++)
    {
        for (int j = 0; j < N_communities; j++)
        {
            delta_Is[i][j] =
                community_trajectory[i + 1][j][1] - community_trajectory[i][j][1] + community_recoveries[i][j];
        }
    }

    std::vector<uint32_t> community_idx(N_communities);
    std::iota(community_idx.begin(), community_idx.end(), 0);

    auto generate_seeds = [](uint32_t N, uint32_t seed)
    {
        std::vector<uint32_t> seeds(N);
        std::mt19937 rng(seed);
        std::generate(seeds.begin(), seeds.end(), [&rng]()
                      { return rng(); });
        return seeds;
    };

    auto rngs = generate_seeds(Nt, seed);

    auto zip = Inf_Sample_Data_t::make(community_idx, delta_Is[0], rngs, connection_events);
    std::transform(zip.begin(), zip.end(), connection_infections.begin(), [ccm](auto &z)
                   { return sample_connection_infections(z.community_idx, z.N_infected, z.connection_events, ccm, z.seed); });

    return connection_infections;
}

int main()
{
    uint32_t N_clusters = 2;
    uint32_t N_pop = 100;
    uint32_t N_pop_tot = N_pop * N_clusters;
    float p_in = 1.0f;
    float p_out = 1.0f;
    uint32_t N_sims = 2;
    uint32_t Ng = 1;
    // sycl::queue q(sycl::gpu_selector_v);
    sycl::queue q(sycl::cpu_selector_v);
    // get work group size
    auto device = q.get_device();
    auto N_wg = device.get_info<sycl::info::device::max_work_group_size>();

    uint32_t Nt = 3;
    uint32_t seed = 100;
    uint32_t N_threads = 10;

    auto G =
        create_planted_SBM(N_pop, N_clusters, p_in, p_out, N_threads, seed);

    float p_I_min = 1e-3f;
    float p_I_max = 1e-2f;
    float p_R = 1e-1f;
    SIR_SBM_Param_t param;
    param.p_R = p_R;

    auto p_I0 = 0.1f;
    auto p_R0 = 0.0f;
    auto [edges, e0] = buffer_create_1D(q, G.edge_list);
    auto seed_buf = generate_seeds(q, N_wg, seed);
    auto [ecm_buf, e1] = buffer_create_1D(q, G.ecm);
    auto [vcm_buf, e2] = buffer_create_1D(q, G.vcm);

    auto trajectory = sycl::buffer<SIR_State, 2>(sycl::range<2>(Nt + 1, N_pop_tot));
    std::vector<std::vector<float>> p_I_vec = generate_p_Is(G.N_connections, p_I_min, p_I_max, Nt, seed);
    auto [p_I_buf, e3] = buffer_create_2D(q, p_I_vec);
    std::vector<std::vector<Edge_t>> connection_events_init(Nt + 1, std::vector<Edge_t>(G.N_connections, {0, 0}));
    auto [connection_events_buf, e5] = buffer_create_2D(q, connection_events_init);
    std::vector<std::vector<State_t>> community_state_init(Nt + 1, std::vector<State_t>(N_clusters, {0, 0, 0}));
    auto [community_state_buf, e6] = buffer_create_2D(q, community_state_init);
    std::vector<sycl::event> buffer_creation_events = {e0, e1, e2, e3, e5, e6};

    auto advance_event = initialize_vertices(p_I0, p_R0, q, trajectory, seed_buf, buffer_creation_events);

    // recovery
    uint32_t t = 0;

    sycl::event rec_event;
    sycl::event inf_event;

    for (int t = 0; t < Nt; t++)
    {
        auto rec_event = recover(q, t, inf_event, p_R, seed_buf, trajectory, vcm_buf);

        inf_event = infect(q, ecm_buf, p_I_buf, seed_buf, connection_events_buf, trajectory, edges, N_wg, t, G.N_connections, rec_event);
    }
    inf_event.wait();
    auto vertex_state = read_buffer(q, trajectory, inf_event);

    std::vector<std::vector<State_t>> community_state(Nt + 1, std::vector<State_t>(N_clusters, {0, 0, 0}));
    std::transform(std::execution::par_unseq, vertex_state.begin(), vertex_state.end(), community_state.begin(), [=](auto v_state)
                   {

        std::vector<State_t> state(N_clusters, {0, 0, 0});
        for(int i = 0; i < v_state.size(); i++)
        {
            auto community_idx = G.vcm[i];
            state[community_idx][v_state[i]]++;
        }
        return state; });

    std::vector<std::vector<uint32_t>> community_recoveries(Nt, std::vector<uint32_t>(N_clusters, 0));
    for(int t = 0; t < Nt; t++)
    {
        for(int j = 0; j < N_clusters; j++)
        {
            community_recoveries[t][j] = community_state[t+1][j][2] - community_state[t][j][2];
        }
    }
    auto connection_events = read_buffer(q, connection_events_buf, inf_event);



    auto connection_infections = sample_connection_infections(community_state, connection_events, community_recoveries, G.connection_community_map, seed);
    std::string output_dir = std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/SIR_sim/Graph_" + std::to_string(0) + "/";
    std::filesystem::create_directory(
        std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/SIR_sim/");
    std::filesystem::create_directories(output_dir);
    uint32_t sim_idx = 0;

    std::ofstream community_traj_f(output_dir + "community_trajectory_" +
                                   std::to_string(sim_idx) + ".csv");
    std::ofstream connection_events_f(output_dir + "connection_events_" +
                                      std::to_string(sim_idx) + ".csv");

    std::ofstream connection_infections_f(output_dir + "connection_infections_" +
                                          std::to_string(sim_idx) + ".csv");

    std::for_each(community_state.begin(), community_state.end(),
                  [&](auto &community_trajectory_i)
                  {
                      linewrite(community_traj_f, community_trajectory_i);
                  });
    std::for_each(connection_events.begin(),
                  connection_events.end(),
                  [&](auto &connection_events_i)
                  {
                      linewrite(connection_events_f, connection_events_i);
                  });

    std::for_each(connection_infections.begin(),
                  connection_infections.end(),
                  [&](auto &connection_infections_i)
                  {
                      linewrite(connection_infections_f, connection_infections_i);
                  });
    std::ofstream p_I_f(output_dir + "p_Is_" + std::to_string(sim_idx) + ".csv");
    std::vector<std::vector<float>> p_I_duplicated;
    std::transform(param.p_I.begin(), param.p_I.end(),
                   std::back_inserter(p_I_duplicated),
                   [&](const auto &p_I_i)
                   {
        std::vector<float> p_I_dup = p_I_i;
        p_I_dup.insert(p_I_dup.end(), p_I_i.begin(), p_I_i.end());
        return p_I_dup; });
    std::for_each(p_I_duplicated.begin(), p_I_duplicated.end(),
                  [&](auto &p_I_i)
                  { linewrite(p_I_f, p_I_i); });

    return 0;
}
