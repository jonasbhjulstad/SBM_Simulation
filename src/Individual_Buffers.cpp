#include <Sycl_Graph/Individual_Buffers.hpp>
#include <Sycl_Graph/SIR_Dynamics.hpp>
#include <algorithm>


struct Individual_Buffers
{
    Individual_Buffers(sycl::queue &q, const std::vector<std::vector<float>> &p_Is, uint32_t N_connections, uint32_t Nt, uint32_t N_communities, uint32_t N_pop, uint32_t N_edges, uint32_t N_wg, uint32_t seed)

        : events(4), rngs{generate_rngs(q, N_wg, events[0])}, events_to{sycl::buffer<uint32_t, 2>(sycl::range<2>(Nt, N_connections))},
          trajectory{sycl::buffer<SIR_State, 3>(sycl::range<2>(Nt + 1, N_pop * N_communities))},
          events_from{sycl::buffer<uint32_t, >(sycl::range<2>(Nt, N_connections))},
          infections_from{buffer_create_1D(q, std::vector<uint8_t>(N_pop * N_communities, false), events[1])},
          infections_to{buffer_create_1D(q, std::vector<uint8_t>(N_pop * N_communities, false), events[2])},
          p_Is{buffer_create_2D(q, p_Is, events[3])}
    {
    }
    Individual_Buffers(const Individual_Buffers &) = default;


    sycl::event initialize_trajectory(sycl::queue &q, float p_I0, float p_R0, std::vector<sycl::event> dep_event)
    {
        return initialize_vertices(p_I0, p_R0, q, trajectory, seeds, dep_event);
    }

    static std::vector<Individual_Buffers> make(sycl::queue &q, const std::vector<std::vector<std::vector<float>>> &p_Is_vec, uint32_t N_connections, uint32_t Nt, uint32_t N_communities, uint32_t N_pop, uint32_t N_edges, uint32_t N_wg, const std::vector<uint32_t> &init_seeds)
    {
        std::vector<Individual_Buffers> result;
        result.reserve(init_seeds.size());
        std::transform(p_Is_vec.begin(), p_Is_vec.end(), init_seeds.begin(), std::back_inserter(result), [&](const auto &p_Is, uint32_t init_seed)
                       { return Individual_Buffers(q, p_Is, N_connections, Nt, N_communities, N_pop, N_edges, N_wg, init_seed); });
        return result;
    }

    static std::vector<Individual_Buffers> make(sycl::queue &q, const std::vector<std::vector<float>> &p_Is, uint32_t N_connections, uint32_t Nt, uint32_t N_communities, uint32_t N_pop, uint32_t N_edges, uint32_t N_wg, const std::vector<uint32_t> &init_seeds)
    {
        std::vector<Individual_Buffers> result;
        result.reserve(init_seeds.size());
        std::transform(init_seeds.begin(), init_seeds.end(), std::back_inserter(result), [&](uint32_t init_seed)
                       { return Individual_Buffers(q, p_Is, N_connections, Nt, N_communities, N_pop, N_edges, N_wg, init_seed); });
        return result;
    }
    Individual_Buffer_Data read_buffers(sycl::queue &q, auto dep_event)
    {
        std::vector<sycl::event> read_events(3);
        auto p_vertex_state = read_buffer(q, this->trajectory, dep_event, read_events[0]);
        auto p_from_events = read_buffer(q, this->events_from, dep_event, read_events[1]);
        auto p_to_events = read_buffer(q, this->events_to, dep_event, read_events[2]);

        std::for_each(read_events.begin(), read_events.end(), [](auto &e)
                      { e.wait(); });
        return {read_pointer(p_vertex_state, this->trajectory.get_range()[0], this->trajectory.get_range()[1]),
                read_pointer(p_from_events, this->events_from.get_range()[0], this->events_from.get_range()[1]),
                read_pointer(p_to_events, this->events_to.get_range()[0], this->events_to.get_range()[1])};
    };
};
