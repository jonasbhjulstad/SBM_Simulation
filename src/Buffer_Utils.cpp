#include "Sycl_Graph/Buffer_Utils_impl.hpp"
#include <random>
#include <execution>
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

std::vector<uint32_t> generate_seeds(uint32_t N_rng, uint32_t seed)
{
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> dis(0, 1000000);
    std::vector<uint32_t> rngs(N_rng);
    std::generate(rngs.begin(), rngs.end(), [&]()
                  { return dis(gen); });
    return rngs;
}

sycl::buffer<uint32_t> generate_seeds(sycl::queue &q, uint32_t N_rng,
                                      uint32_t seed, sycl::event& event)
{
    auto rngs = generate_seeds(N_rng, seed);
    sycl::buffer<uint32_t> tmp(rngs.data(), rngs.size());
    sycl::buffer<uint32_t> result(sycl::range<1>(rngs.size()));

    event = q.submit([&](sycl::handler &h)
             {
        auto tmp_acc = tmp.get_access<sycl::access::mode::read>(h);
        auto res_acc = result.get_access<sycl::access::mode::write>(h);

        h.copy(tmp_acc, res_acc);
        // h.parallel_for(result.get_range(), [=](sycl::id<1> idx)
        //                { res_acc[idx] = tmp_acc[idx]; }); });
             });
    return result;
}

sycl::buffer<Static_RNG::default_rng> generate_rngs(sycl::queue& q, uint32_t N_rng, uint32_t seed, sycl::event& event)
{
    auto seeds = generate_seeds(N_rng, seed);
    std::vector<Static_RNG::default_rng> rngs;
    rngs.reserve(N_rng);
    std::transform(seeds.begin(), seeds.end(), std::back_inserter(rngs), [](auto seed){return Static_RNG::default_rng(seed);});
    sycl::buffer<Static_RNG::default_rng> result((sycl::range<1>(rngs.size())));
    event = q.submit([&](sycl::handler &h)
             {
        auto res_acc = result.get_access<sycl::access::mode::write>(h);

        h.copy(rngs.data(), res_acc);
             });
    event.wait();
    return result;
}

sycl::buffer<Static_RNG::default_rng, 2> generate_rngs(sycl::queue& q, sycl::range<2> size, uint32_t seed, sycl::event& event)
{
    uint32_t N_rngs = size[0] * size[1];
    auto seeds = generate_seeds(N_rngs, seed);
    std::vector<Static_RNG::default_rng> rngs;
    rngs.reserve(N_rngs);
    std::transform(seeds.begin(), seeds.end(), std::back_inserter(rngs), [](auto seed){return Static_RNG::default_rng(seed);});
    sycl::buffer<Static_RNG::default_rng, 2> result(size);
    event = q.submit([&](sycl::handler &h)
             {
        auto res_acc = result.get_access<sycl::access::mode::discard_write>(h);

        h.copy(rngs.data(), res_acc);
             });
    return result;
}




std::vector<uint32_t> pairlist_to_vec(const std::vector<std::pair<uint32_t, uint32_t>> &pairlist)
{
    std::vector<uint32_t> res(pairlist.size() * 2);
    for(int i = 0; i < pairlist.size(); i++)
    {
        res[i * 2] = pairlist[i].first;
        res[i * 2 + 1] = pairlist[i].second;
    }
    return res;
}
std::vector<std::pair<uint32_t, uint32_t>> vec_to_pairlist(const std::vector<uint32_t> &vec)
{
    std::vector<std::pair<uint32_t, uint32_t>> res(vec.size() / 2);
    for(int i = 0; i < vec.size() / 2; i++)
    {
        res[i].first = vec[i * 2];
        res[i].second = vec[i * 2 + 1];
    }
    return res;
}
std::vector<float> generate_floats(uint32_t N, float min, float max, uint32_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(min, max);
    std::vector<float> result(N);
    std::generate(result.begin(), result.end(), [&](){return dist(rng);});
    return result;
}

std::vector<std::vector<float>> generate_floats(uint32_t rows, uint32_t cols, float min, float max, uint32_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(min, max);
    std::vector<std::vector<float>> floats(rows, std::vector<float>(cols));
    std::for_each(floats.begin(), floats.end(), [&](auto& row){std::generate(row.begin(), row.end(), [&](){return dist(rng);});});
    return floats;
}

std::vector<std::vector<std::vector<float>>> generate_floats(uint32_t N0, uint32_t N1, uint32_t N2, float min, float max, uint32_t seed)
{
    //generate p_Is
    using p_I_mat = std::vector<std::vector<float>>;
    std::vector<p_I_mat> p_Is(N0);
    auto seeds = generate_seeds(N0, seed);

    std::transform(std::execution::par_unseq, seeds.begin(), seeds.end(), p_Is.begin(), [=](auto seed)
    {
        return generate_floats(N1, N2, min, max, seed);
    });
    return p_Is;
}


template std::vector<uint32_t> merge_vectors(const std::vector<std::vector<uint32_t>>&);
template std::vector<int> merge_vectors(const std::vector<std::vector<int>>&);
template std::vector<float> merge_vectors(const std::vector<std::vector<float>>&);

template std::vector<std::vector<uint32_t>> diff(const std::vector<std::vector<uint32_t>> &v);
template std::vector<std::vector<int>> diff(const std::vector<std::vector<int>> &v);

template sycl::buffer<uint32_t, 1> create_device_buffer<uint32_t, 1>(sycl::queue& q, const std::vector<uint32_t> &host_data, const sycl::range<1>& range, sycl::event& event);
template sycl::buffer<uint32_t, 2> create_device_buffer<uint32_t, 2>(sycl::queue& q, const std::vector<uint32_t> &host_data, const sycl::range<2>& range, sycl::event& event);
template sycl::buffer<uint32_t, 3> create_device_buffer<uint32_t, 3>(sycl::queue& q, const std::vector<uint32_t> &host_data, const sycl::range<3>& range, sycl::event& event);

template sycl::buffer<float, 1> create_device_buffer<float, 1>(sycl::queue& q, const std::vector<float> &host_data, const sycl::range<1>& range, sycl::event& event);
template sycl::buffer<float, 2> create_device_buffer<float, 2>(sycl::queue& q, const std::vector<float> &host_data, const sycl::range<2>& range, sycl::event& event);
template sycl::buffer<float, 3> create_device_buffer<float, 3>(sycl::queue& q, const std::vector<float> &host_data, const sycl::range<3>& range, sycl::event& event);

template sycl::buffer<SIR_State, 3> create_device_buffer<SIR_State, 3>(sycl::queue& q, const std::vector<SIR_State> &host_data, const sycl::range<3>& range, sycl::event& event);

template std::vector<SIR_State> read_buffer<SIR_State, 3>(sycl::buffer<SIR_State,3>& buf, sycl::queue& q, sycl::event& event);
template std::vector<State_t> read_buffer<State_t, 3>(sycl::buffer<State_t,3>& buf, sycl::queue& q, sycl::event& event);
