#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Buffer_Utils.hpp>
#include <Sycl_Graph/SIR_Dynamics.hpp>
#include <execution>
#include <random>

sycl::event initialize_vertices(float p_I0, float p_R0, sycl::queue &q,
                                sycl::buffer<SIR_State, 3> &buf, sycl::buffer<Static_RNG::default_rng, 2> &rng_buf, std::vector<sycl::event> event)
{
    uint32_t N_sims = buf.get_range()[0];
    uint32_t N_vertices = buf.get_range()[2];
    uint32_t N_seeds = seed_buf.get_range()[0];
    auto N_threads = std::min({N_vertices, N_seeds});
    uint32_t N_vertices_per_thread = N_vertices / N_threads + 1;
    auto S_event = q.submit([&](sycl::handler &h)
                            {
                h.depends_on(event);
    auto state_acc = buf.template get_access<sycl::access::mode::write>(h);
    h.parallel_for(buf.get_range(),
                   [=](sycl::id<3> id) { state_acc[id] = SIR_INDIVIDUAL_S; }); });

    return q.submit([&](sycl::handler &h)
                    {
    h.depends_on(S_event);
    auto state_acc = buf.template get_access<sycl::access::mode::write>(h);
    auto seed_acc =
        seed_buf.template get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<2>(N_vertices, N_sims), [=](sycl::id<1> id) {

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
    }); });
}

sycl::event recover(sycl::queue &q, uint32_t t, const std::vector<sycl::event> &dep_event, float p_R, sycl::buffer<uint32_t> &seed_buf, sycl::buffer<SIR_State, 2> &trajectory)
{
    uint32_t N_vertices = trajectory.get_range()[1];
    uint32_t N_seeds = seed_buf.size();
    uint32_t N_threads = std::min({N_vertices, N_seeds});
    uint32_t N_vertex_per_thread = N_vertices / N_threads + 1;
    auto event = q.submit([&](sycl::handler &h)
                          {
    h.depends_on(dep_event);
    auto seed_acc =
        seed_buf.template get_access<sycl::access_mode::atomic>(h);
    auto v_acc =
        trajectory.template get_access<sycl::access::mode::read_write>(h);
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

std::vector<sycl::event> infect(sycl::queue &q, const std::shared_ptr<sycl::buffer<uint32_t>> &ecm_buf, sycl::buffer<float, 2> &p_I_buf, sycl::buffer<uint32_t> &seed_buf, sycl::buffer<uint32_t, 2> &event_from_buf, sycl::buffer<uint32_t, 2> &event_to_buf, sycl::buffer<SIR_State, 2> &trajectory, const std::shared_ptr<sycl::buffer<uint32_t>> &edge_from_buf, const std::shared_ptr<sycl::buffer<uint32_t>> &edge_to_buf, sycl::buffer<uint8_t> &inf_buf_from, sycl::buffer<uint8_t> &inf_buf_to, uint32_t t, uint32_t N_connections, sycl::event dep_event)
{

    uint32_t N_edges = ecm_buf->size();
    uint32_t N_wg = seed_buf.size();
    uint32_t N_vertices = trajectory.get_range()[1];
    auto inf_kernel = [&](auto &edge_buf_0, auto &edge_buf_1, auto &inf_buf)
    {
        return [&](sycl::handler &h)
        {
          h.depends_on(dep_event);
          auto ecm_acc = ecm_buf->template get_access<sycl::access::mode::read>(h);
          auto p_I_acc = sycl::accessor<float, 2, sycl::access::mode::read>(p_I_buf, h, sycl::range<2>(1, N_connections), sycl::range<2>(t, 0));
          auto seed_acc =
              seed_buf.template get_access<sycl::access_mode::atomic>(h);
          auto v_acc = sycl::accessor<SIR_State, 2, sycl::access::mode::read_write>(trajectory, h, sycl::range<2>(2, N_vertices), sycl::range<2>(t, 0));

          auto e_acc_0 = edge_buf_0->template get_access<sycl::access::mode::read>(h);
          auto e_acc_1 = edge_buf_0->template get_access<sycl::access::mode::read>(h);
          auto inf_acc = inf_buf.template get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(N_wg, [=](sycl::id<1> id) {
                auto seed = sycl::atomic_fetch_add<uint32_t>(seed_acc[id], 1);
                Static_RNG::default_rng rng(seed);
                Static_RNG::bernoulli_distribution<float> bernoulli_I(0.0f);
                uint32_t N_edge_per_wg = (N_edges / N_wg) + 1;
                for (int i = 0; i < N_edge_per_wg; i++)
                {
                    auto edge_idx = id * N_edge_per_wg + i;
                    if (edge_idx >= N_edges)
                        break;
                    if ((v_acc[0][e_acc_1[edge_idx]] == SIR_INDIVIDUAL_S) && (v_acc[0][e_acc_0[edge_idx]] == SIR_INDIVIDUAL_I))
                    {
                        auto p_I = p_I_acc[0][ecm_acc[edge_idx]];
                        bernoulli_I.p = p_I;
                        if (bernoulli_I(rng))
                        {
                            v_acc[1][e_acc_1[edge_idx]] = SIR_INDIVIDUAL_I;
                            inf_acc[edge_idx] = true;
                        }
                    }
                }
      }); };
    };
    auto acc_kernel = [&](auto &event_buf, auto &inf_buf, auto &i_event)
    { return [&](sycl::handler &h)
      {

                                        h.depends_on(i_event);
                                        auto infection_acc = inf_buf.template get_access<sycl::access::mode::read>(h);
                                        auto event_acc = sycl::accessor<uint32_t, 2, sycl::access::mode::write>(event_buf, h, sycl::range<2>(1, N_connections), sycl::range<2>(t, 0));
                                        auto ecm_acc = ecm_buf->template get_access<sycl::access::mode::read>(h);
                                        h.parallel_for(N_connections, [=](sycl::id<1> id)
                                                       {
                                                        event_acc[0][id] = 0;
                                                        for(int i = 0; i < N_edges; i++)
                                                        {
                                                            if (infection_acc[i])
                                                            {
                                                                event_acc[0][ecm_acc[i]]++;
                                                            }
                                                        }
                                                       }); }; };

    std::vector<sycl::event> events(4);
    auto kernel_0 = inf_kernel(edge_from_buf, edge_to_buf, inf_buf_to);
    auto kernel_1 = inf_kernel(edge_to_buf, edge_from_buf, inf_buf_from);
    events[0] = q.submit(kernel_0);
    events[1] = q.submit(kernel_1);
    events[2] = q.submit(acc_kernel(event_to_buf, inf_buf_to, events[0]));
    events[3] = q.submit(acc_kernel(event_from_buf, inf_buf_from, events[1]));
    return events;
}

std::vector<std::vector<float>> generate_p_Is(uint32_t N_community_connections, float p_I_min, float p_I_max, uint32_t Nt, uint32_t seed)
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
