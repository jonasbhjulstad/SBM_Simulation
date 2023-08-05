#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Buffer_Utils.hpp>
#include <Sycl_Graph/SIR_Dynamics.hpp>
#include <random>
#include <execution>
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

sycl::event initialize_vertices(float p_I0, float p_R0, sycl::queue &q,
                                sycl::buffer<SIR_State, 2> &buf, sycl::buffer<uint32_t> &seed_buf, std::vector<sycl::event> event)
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

sycl::event recover(sycl::queue &q, uint32_t t, sycl::event &dep_event, float p_R, sycl::buffer<uint32_t> &seed_buf, sycl::buffer<SIR_State, 2> &trajectory, sycl::buffer<uint32_t> &vcm_buf)
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
        uint32_t seed = sycl::atomic_fetch_add<uint32_t>(seed_acc[id], 1);
        Static_RNG::default_rng rng(seed);
        Static_RNG::bernoulli_distribution<float> bernoulli_R(p_R);
        for(int i = 0; i < N_vertex_per_thread; i++)
        {
            auto v_idx = N_vertex_per_thread*id[0] + i;
            if(v_idx >= N_vertices)
                break;
      auto state_prev = v_acc[t][v_idx];
      v_acc[t + 1][v_idx] = state_prev;
      if (state_prev == SIR_INDIVIDUAL_I) {
        if (bernoulli_R(rng)) {
          v_acc[t + 1][v_idx] = SIR_INDIVIDUAL_R;
        }
      }
        }
    }); });
    return event;
}

sycl::event infect(sycl::queue &q, sycl::buffer<uint32_t> &ecm_buf, sycl::buffer<float, 2> &p_I_buf, sycl::buffer<uint32_t> &seed_buf, sycl::buffer<uint32_t, 2> &event_from_buf, sycl::buffer<uint32_t, 2> &event_to_buf, sycl::buffer<SIR_State, 2> &trajectory, sycl::buffer<uint32_t, 1> &edge_from_buf, sycl::buffer<uint32_t, 1> &edge_to_buf, uint32_t N_wg, uint32_t t, uint32_t N_connections, sycl::event dep_event)
{
    uint32_t N_edges = ecm_buf.size();
    std::vector<uint32_t> infection_indices_init(N_edges, std::numeric_limits<uint32_t>::max());
    sycl::event ii_event;
    auto infection_indices_buf = buffer_create_1D(q, infection_indices_init, ii_event);
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
    auto edge_from_acc = edge_from_buf.template get_access<sycl::access::mode::read>(h);
    auto edge_to_acc = edge_to_buf.template get_access<sycl::access::mode::read>(h);
    auto infection_indices_acc = infection_indices_buf.template get_access<sycl::access::mode::write>(h);
    h.parallel_for(N_wg, [=](sycl::id<1> id) {
      auto seed = sycl::atomic_fetch_add<uint32_t>(seed_acc[id], 1);
        Static_RNG::default_rng rng(seed);
    Static_RNG::bernoulli_distribution<float> bernoulli_I(0.0f);
      uint32_t N_edge_per_wg = (N_edges / N_wg) + 1;
      for(int i = 0; i < N_edge_per_wg; i++)
      {
        auto edge_idx = id * N_edge_per_wg + i;
        if (edge_idx >= N_edges)
          break;
      auto id_from = edge_from_acc[edge_idx];
      auto id_to = edge_to_acc[edge_idx];
      auto [sus_id, direction] = get_susceptible_id_if_infected(v_acc[t], id_from, id_to);
      if (sus_id == std::numeric_limits<uint32_t>::max())
        continue;
    auto p_I = p_I_acc[t][ecm_acc[edge_idx]];
    bernoulli_I.p = p_I;
    if (bernoulli_I(rng)) {
        v_acc[t + 1][sus_id] = SIR_INDIVIDUAL_I;
        infection_indices_acc[edge_idx] = direction;
    }
      }
    }); });

    auto accumulate_event = q.submit([&](sycl::handler &h)
                                     {
                                        auto infection_indices_acc = infection_indices_buf.template get_access<sycl::access::mode::read>(h);
                                        auto event_from_acc = event_from_buf.template get_access<sycl::access::mode::read_write>(h);
                                        auto event_to_acc = event_to_buf.template get_access<sycl::access::mode::read_write>(h);
                                        auto ecm_acc = ecm_buf.template get_access<sycl::access::mode::read>(h);
                                        h.parallel_for(N_connections, [=](sycl::id<1> id)
                                                       {
                                                        event_from_acc[t][id] = 0;
                                                        event_to_acc[t][id] = 0;
                                                        for(int i = 0; i < N_edges; i++)
                                                        {
                                                            if ((infection_indices_acc[i] != std::numeric_limits<uint32_t>::max()) && (ecm_acc[i] == id))
                                                            {
                                                                if (infection_indices_acc[i] == 0)
                                                                {
                                                                    event_from_acc[t][id]++;
                                                            }
                                                                else
                                                                {
                                                                    event_to_acc[t][id]++;}
                                                            }
                                                        }
                                                       }); });
    // accumulate_event = q.submit([&](sycl::handler& h)
    // {
    //     sycl::stream out (1024, 256, h);
    //     h.depends_on(accumulate_event);
    //     auto event_from_acc = event_from_buf.template get_access<sycl::access::mode::read>(h);
    //     auto event_to_acc = event_to_buf.template get_access<sycl::access::mode::read>(h);

    //     h.single_task([=](){
    //         for(int i = 0; i < event_from_buf.size(); i++)
    //         {

    //         }
    //     });
    // });
    return accumulate_event;
}

  std::vector<std::vector<float>> generate_p_Is(uint32_t N_community_connections,
                                                float p_I_min, float p_I_max,
                                                uint32_t Nt, uint32_t seed)
  {
    std::vector<Static_RNG::default_rng> rngs(Nt);
    Static_RNG::default_rng rd(seed);
    std::vector<uint32_t> seeds(Nt);
    std::generate(seeds.begin(), seeds.end(), [&rd]()
                  { return rd(); });
    std::transform(seeds.begin(), seeds.end(), rngs.begin(),
                   [](auto seed)
                   { return Static_RNG::default_rng(seed); });

    std::vector<std::vector<float>> p_Is(
        Nt, std::vector<float>(N_community_connections));

    std::transform(
        std::execution::par_unseq, rngs.begin(), rngs.end(), p_Is.begin(),
        [&](auto &rng)
        {
          Static_RNG::uniform_real_distribution<> dist(p_I_min, p_I_max);
          std::vector<float> p_I(N_community_connections);
          std::generate(p_I.begin(), p_I.end(), [&]()
                        { return dist(rng); });
          return p_I;
        });

    return p_Is;
  }

  std::vector<uint32_t> create_ecm(const std::vector<std::vector<std::pair<uint32_t, uint32_t>>>& edge_lists)
  {
    std::vector<uint32_t> list_sizes(edge_lists.size());
    std::transform(edge_lists.begin(), edge_lists.end(), list_sizes.begin(), [](auto& edge_list){return edge_list.size();});
    uint32_t N_edges = std::accumulate(list_sizes.begin(), list_sizes.end(), 0);
    std::vector<uint32_t> ecm(N_edges);
    uint32_t offset = 0;
    for(int i = 0; i < edge_lists.size(); i++)
    {
        std::fill(ecm.begin() + offset, ecm.begin() + offset + list_sizes[i], i);
        offset += list_sizes[i];
    }
    return ecm;
  }

  std::vector<uint32_t> create_vcm(const std::vector<std::vector<uint32_t>> node_lists)
  {
    std::vector<uint32_t> list_sizes(node_lists.size());
    std::transform(node_lists.begin(), node_lists.end(), list_sizes.begin(), [](auto& node_list){return node_list.size();});
    uint32_t N_nodes = std::accumulate(list_sizes.begin(), list_sizes.end(), 0);
    std::vector<uint32_t> vcm(N_nodes);
    uint32_t offset = 0;
    for(int i = 0; i < node_lists.size(); i++)
    {
        std::fill(vcm.begin() + offset, vcm.begin() + offset + list_sizes[i], i);
        offset += list_sizes[i];
    }
    return vcm;
  }
