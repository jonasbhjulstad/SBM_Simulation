#ifndef SYCL_GRAPH_DATAFRAME_BUFFER_UTILS_HPP
#define SYCL_GRAPH_DATAFRAME_BUFFER_UTILS_HPP
#include <Sycl_Graph/Dataframe/Dataframe.hpp>
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>

template <typename T>
sycl::event initialize_device_buffer(sycl::queue &q, const Dataframe_t<T, 2> &data, sycl::buffer<T, 2> &buf)
{
    auto N_rows = buf.get_range()[0];
    auto N_cols = buf.get_range()[1];
    if_false_throw(data.size() == N_rows, "Dataframe does not have the same number of rows as the buffer: " + std::to_string(data.size()) + " != " + std::to_string(N_rows));
    std::vector<T> tmp(N_cols * N_rows, T{});
    for (int i = 0; i < data.size(); i++)
    {
        for (int j = 0; j < data[i].size(); j++)
        {
            tmp[i * N_cols + j] = data(i, j);
        }
    }

    return q.submit([&](sycl::handler &h)
             {
        auto acc = buf.template get_access<sycl::access::mode::write, sycl::access::target::global_buffer>(h);
        h.copy(tmp.data(), acc); });
}

template <typename T>
sycl::event initialize_device_buffer(sycl::queue &q, const Dataframe_t<T, 2> &data, sycl::buffer<T, 1> &buf)
{
    auto flat_vec = data.flatten();

    return q.submit([&](sycl::handler &h)
             {
        auto acc = buf.template get_access<sycl::access::mode::write, sycl::access::target::global_buffer>(h);
        h.copy(flat_vec.data(), acc); });
}

template <typename T>
sycl::event initialize_device_buffer(sycl::queue &q, const Dataframe_t<T, 3> &data, sycl::buffer<T, 3> &buf)
{
    auto N0 = buf.get_range()[0];
    auto N1 = buf.get_range()[1];
    auto N2 = buf.get_range()[2];

    auto N0_data = data.size();
    auto N1_data = data[0].size();
    auto N2_data = data[0][0].size();

    if_false_throw((N0 == N0_data) && (N1 == N1_data) && (N2 == N2_data), "Dataframe does not have the same dimensions as the buffer: " + std::to_string(N0_data) + " != " + std::to_string(N0) + " || " + std::to_string(N1_data) + " != " + std::to_string(N1) + " || " + std::to_string(N2_data) + " != " + std::to_string(N2));

    auto data_flat = data.flatten();

    return q.submit([&](sycl::handler &h)
             {
        auto acc = buf.template get_access<sycl::access::mode::write, sycl::access::target::global_buffer>(h);
        h.copy(data_flat.data(), acc); });
}


#endif
