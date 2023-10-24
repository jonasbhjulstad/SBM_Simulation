#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
#include <random>

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

std::vector<uint32_t> Buffer_Routines::generate_seeds(uint32_t N_rng, uint32_t seed)
{
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> dis(0, 1000000);
    std::vector<uint32_t> rngs(N_rng);
    std::generate(rngs.begin(), rngs.end(), [&]()
                  { return dis(gen); });
    return rngs;
}

sycl::buffer<uint32_t> Buffer_Routines::generate_seeds(sycl::queue &q, uint32_t N_rng,
                                      uint32_t seed, sycl::event& event)
{
    auto rngs = Buffer_Routines::generate_seeds(N_rng, seed);
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
    auto seeds = Buffer_Routines::generate_seeds(N_rng, seed);
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
    auto seeds = Buffer_Routines::generate_seeds(N_rngs, seed);
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

std::vector<float> generate_floats(uint32_t N, float min, float max, uint32_t N_rngs, uint32_t seed)
{
    std::vector<uint32_t> seeds;
    std::mt19937 rng(seed);
    std::generate_n(std::back_inserter(seeds), N_rngs, [&](){return rng();});
    std::vector<std::mt19937> rngs;
    std::transform(seeds.begin(), seeds.end(), std::back_inserter(rngs), [](auto seed){return std::mt19937(seed);});
    auto N_per_rng = N / N_rngs;
    std::vector<std::vector<float>> result(N_rngs);
    std::transform(rngs.begin(), rngs.end(), result.begin(), [&](auto& rng)
                   {
        std::uniform_real_distribution<float> dist(min, max);
        std::vector<float> res(N_per_rng);
        std::generate(res.begin(), res.end(), [&](){return dist(rng);});
        return res;
                   });
    std::vector<float> flat_result;
    for(auto& r : result)
    {
        flat_result.insert(flat_result.end(), r.begin(), r.end());
    }
    return flat_result;
}

sycl::event generate_floats(sycl::queue& q, std::vector<float>& result, float min, float max, uint32_t seed)
{
    auto device = q.get_device();
    auto wg_size = device.get_info<sycl::info::device::max_work_group_size>();
    std::mt19937 rng(seed);
    std::vector<uint32_t> seeds(wg_size);
    std::generate(seeds.begin(), seeds.end(), [&](){return rng();});

    std::vector<Static_RNG::default_rng> rngs(wg_size);
    std::transform(seeds.begin(), seeds.end(), rngs.begin(), [](auto seed){return Static_RNG::default_rng(seed);});

    sycl::buffer<Static_RNG::default_rng> rngs_buf(rngs.data(), rngs.size());
    sycl::buffer<float> result_buf((sycl::range<1>(result.size())));
    auto N_per_thread = result.size() / wg_size;

    auto rng_event = q.submit([&](sycl::handler& h)
    {
        auto result_acc = result_buf.template get_access<sycl::access::mode::write>(h);
        auto rng_acc = rngs_buf.template get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::range<1>(wg_size), [=](sycl::id<1> idx)
        {
            Static_RNG::uniform_real_distribution<float> dist(min, max);
            for(int i = 0; i < N_per_thread; i++)
            {
                result_acc[idx * N_per_thread + i] = dist(rng_acc[idx]);
            }
        });
    });

    auto cpy_event = q.submit([&](sycl::handler& h)
    {
        auto res_acc = result_buf.template get_access<sycl::access::mode::read>(h);
        h.copy(res_acc, result.data());
    });

    return cpy_event;
}
