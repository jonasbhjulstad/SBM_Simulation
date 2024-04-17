#pragma once
#include <Eigen/Dense>

template <typename T>
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T>
using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

#include <CL/sycl.hpp>

template <typename T>
Vec<T> buffer_copy(sycl::queue& q, sycl::buffer<T>& buf)
{
    Vec<T> vec(buf.size());
    q.submit([&](sycl::handler& h)
    {
        h.copy(buf, vec.data());
    }).wait();
    return vec;
}

template <typename T>
Mat<T> buffer_copy(sycl::queue& q, sycl::buffer<T, 2>& buf)
{
    std::vector<T> data(buf.size());
    q.submit([&](sycl::handler& h)
    {
        h.copy(buf, data.data());
    }).wait();

    return Eigen::Map<Mat<T>>(data.data(), buf.get_range()[0], buf.get_range()[1]);
}

template <typename T>
std::vector<Mat<T>> buffer_copy(sycl::queue& q, sycl::buffer<T, 3>& buf)
{
    std::vector<T> data(buf.size());
    q.submit([&](sycl::handler& h)
    {
        h.copy(buf, data.data());
    }).wait();

    auto range = buf.get_range();
    std::vector<Mat<T>> mats;
    for (int i = 0; i < range[0]; ++i)
    {
        mats.push_back(Eigen::Map<Mat<T>>(data.data() + i * range[1] * range[2], range[1], range[2]));
    }
    return mats;
}
