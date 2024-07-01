// eigen_sycl.hpp
//

#ifndef LZZ_SIR_SBM_LZZ_eigen_sycl_hpp
#define LZZ_SIR_SBM_LZZ_eigen_sycl_hpp
#line 3 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen_sycl.hpp"
template <typename T>
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T>
using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
#include <CL/sycl.hpp>
#define LZZ_INLINE inline
#line 11 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen_sycl.hpp"
template <typename T>
#line 12 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen_sycl.hpp"
Vec <T> buffer_copy (sycl::queue & q, sycl::buffer <T> & buf);
#line 22 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen_sycl.hpp"
template <typename T>
#line 23 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen_sycl.hpp"
Mat <T> buffer_copy (sycl::queue & q, sycl::buffer <T, 2> & buf);
#line 34 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen_sycl.hpp"
template <typename T>
#line 35 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen_sycl.hpp"
std::vector <(Mat<T>)> buffer_copy (sycl::queue & q, sycl::buffer <T, 3> & buf);
#line 11 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen_sycl.hpp"
template <typename T>
#line 12 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen_sycl.hpp"
Vec <T> buffer_copy (sycl::queue & q, sycl::buffer <T> & buf)
#line 13 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen_sycl.hpp"
{
    Vec<T> vec(buf.size());
    q.submit([&](sycl::handler& h)
    {
        h.copy(buf, vec.data());
    }).wait();
    return vec;
}
#line 22 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen_sycl.hpp"
template <typename T>
#line 23 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen_sycl.hpp"
Mat <T> buffer_copy (sycl::queue & q, sycl::buffer <T, 2> & buf)
#line 24 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen_sycl.hpp"
{
    std::vector<T> data(buf.size());
    q.submit([&](sycl::handler& h)
    {
        h.copy(buf, data.data());
    }).wait();

    return Eigen::Map<Mat<T>>(data.data(), buf.get_range()[0], buf.get_range()[1]);
}
#line 34 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen_sycl.hpp"
template <typename T>
#line 35 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen_sycl.hpp"
std::vector <(Mat<T>)> buffer_copy (sycl::queue & q, sycl::buffer <T, 3> & buf)
#line 36 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//eigen_sycl.hpp"
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
#undef LZZ_INLINE
#endif
