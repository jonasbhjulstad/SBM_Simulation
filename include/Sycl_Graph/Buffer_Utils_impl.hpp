#ifndef BUFFER_UTILS_IMPL_HPP
#define BUFFER_UTILS_IMPL_HPP
#include <fstream>
#include <Sycl_Graph/SIR_Types.hpp>
#include <CL/sycl.hpp>


template <typename T>
std::vector<T> merge_vectors(const std::vector<std::vector<T>> &vectors)
{
    std::vector<T> merged;
    uint32_t size = 0;
    for(int i = 0; i < vectors.size(); i++)
    {
        size += vectors[i].size();
    }
    merged.reserve(size);
    for (auto &v : vectors)
    {
        merged.insert(merged.end(), v.begin(), v.end());
    }
    return merged;
}
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
std::vector<std::vector<T>> read_buffer(sycl::queue &q, sycl::buffer<T, 2> &buf,
                                        sycl::event event, std::ofstream& log_file)
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
    if(log_file.is_open())
        log_file << "2D buffer read of size " << rows << "x" << cols << std::endl;
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
