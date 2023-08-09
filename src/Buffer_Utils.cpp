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

template std::vector<std::vector<uint32_t>> read_buffer(sycl::queue &q, sycl::buffer<uint32_t, 2> &buf,
                                                        sycl::event events);
template std::vector<std::vector<float>> read_buffer(sycl::queue &q, sycl::buffer<float, 2> &buf, sycl::event events);

template std::vector<std::vector<uint32_t>> read_buffer(sycl::queue &q, sycl::buffer<uint32_t, 2> &buf,
                                                        sycl::event events, std::ofstream&);
template std::vector<std::vector<float>> read_buffer(sycl::queue &q, sycl::buffer<float, 2> &buf, sycl::event events, std::ofstream&);


template std::vector<std::vector<uint32_t>> diff(const std::vector<std::vector<uint32_t>> &v);
template std::vector<std::vector<int>> diff(const std::vector<std::vector<int>> &v);

template sycl::buffer<uint32_t, 1> buffer_create_1D(sycl::queue &q, const std::vector<uint32_t> &data, sycl::event &res_event);
template sycl::buffer<uint32_t, 2> buffer_create_2D(sycl::queue &q, const std::vector<std::vector<uint32_t>> &data, sycl::event &res_event);
template sycl::buffer<int, 1> buffer_create_1D(sycl::queue &q, const std::vector<int> &data, sycl::event &res_event);
template sycl::buffer<int, 2> buffer_create_2D(sycl::queue &q, const std::vector<std::vector<int>> &data, sycl::event &res_event);
template sycl::buffer<float, 1> buffer_create_1D(sycl::queue &q, const std::vector<float> &data, sycl::event &res_event);
template sycl::buffer<float, 2> buffer_create_2D(sycl::queue &q, const std::vector<std::vector<float>> &data, sycl::event &res_event);
template sycl::buffer<SIR_State, 2> buffer_create_2D(sycl::queue &q, const std::vector<std::vector<SIR_State>> &data, sycl::event &res_event);



template std::shared_ptr<sycl::buffer<uint32_t, 1>> shared_buffer_create_1D(sycl::queue &q, const std::vector<uint32_t> &data, sycl::event &res_event);
template std::shared_ptr<sycl::buffer<float, 1>> shared_buffer_create_1D(sycl::queue &q, const std::vector<float> &data, sycl::event &res_event);
