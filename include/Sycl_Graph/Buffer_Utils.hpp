#ifndef BUFFER_UTILS_HPP
#define BUFFER_UTILS_HPP
#include <fstream>
#include <Sycl_Graph/SIR_Types.hpp>
#include <CL/sycl.hpp>
void linewrite(std::ofstream &file, const std::vector<uint32_t> &state_iter);

void linewrite(std::ofstream &file, const std::vector<float> &val);

void linewrite(std::ofstream &file,
               const std::vector<std::array<uint32_t, 3>> &state_iter);

std::vector<uint32_t> pairlist_to_vec(const std::vector<std::pair<uint32_t, uint32_t>> &pairlist);
std::vector<std::pair<uint32_t, uint32_t>> vec_to_pairlist(const std::vector<uint32_t> &vec);

template <typename T>
sycl::buffer<T, 1> buffer_create_1D(sycl::queue &q, const std::vector<T> &data, sycl::event &res_event)
{
    sycl::buffer<T> tmp(data.data(), data.size());
    sycl::buffer<T> result(sycl::range<1>(data.size()));

    res_event = q.submit([&](sycl::handler &h)
                         {
                auto tmp_acc = tmp.template get_access<sycl::access::mode::read>(h);
                auto res_acc = result.template get_access<sycl::access::mode::write>(h);
                h.copy(tmp_acc, res_acc); });
    return result;
}

template <typename T>
sycl::buffer<T, 2> buffer_create_2D(sycl::queue &q, const std::vector<std::vector<T>> &data, sycl::event &res_event)
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
    res_event = q.submit([&](sycl::handler &h)
                         {
        auto tmp_acc = tmp.template get_access<sycl::access::mode::read>(h);
        auto res_acc = result.template get_access<sycl::access::mode::write>(h);
        h.parallel_for(result.get_range(), [=](sycl::id<2> idx)
                       { res_acc[idx] = tmp_acc[idx]; }); });

    return result;
}

sycl::buffer<uint32_t> generate_seeds(sycl::queue &q, uint32_t N_rng,
                                      uint32_t seed);

template <typename T>
std::vector<std::vector<T>> read_buffer(sycl::queue &q, sycl::buffer<T, 2> &buf,
                                        sycl::event event)
{

    auto range = buf.get_range();
    auto rows = range[0];
    auto cols = range[1];

    std::vector<T> data(cols * rows);
    T *p_data = data.data();

    q.submit([&](sycl::handler &h)
             {
        //create accessor
        h.depends_on(event);
        auto acc = buf.template get_access<sycl::access::mode::read>(h);
        h.copy(acc, p_data); })
        .wait();

    // transform to 2D vector
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
template <typename T>
std::vector<std::vector<T>> diff(const std::vector<std::vector<T>> &v)
{
    std::vector<std::vector<T>> res(v.size() - 1, std::vector<T>(v[0].size()));
    for (int i = 0; i < v.size() - 1; i++)
    {
        for (int j = 0; j < v[i].size(); j++)
        {
            res[i][j] = v[i + 1][j] - v[i][j];
        }
    }
    return res;
}
#endif
