#include <CL/sycl.hpp>
#include <iostream>

template <typename T>
std::vector<std::vector<std::vector<T>>> vector_remap(std::vector<T> &input, size_t N0, size_t N1, size_t N2)
{
    size_t M = N0 * N1 * N2;
    if (input.size() != M)
    {
        throw std::runtime_error("Input vector size does not match 3D dimensions.");
    }

    std::vector<std::vector<std::vector<T>>> output(N0, std::vector<std::vector<T>>(N1, std::vector<T>(N2)));

    //linear position defined by i2 + i1*n2 + i0*n1*n2
    for (size_t i0 = 0; i0 < N0; i0++)
    {
        for (size_t i1 = 0; i1 < N1; i1++)
        {
            for (size_t i2 = 0; i2 < N2; i2++)
            {
                output[i0][i1][i2] = input[i2 + i1 * N2 + i0 * N1 * N2];
            }
        }
    }
    return output;
}
int main()
{
    sycl::queue q(sycl::cpu_selector_v);
    sycl::buffer<uint32_t, 3> buf(sycl::range<3>(3, 3, 3));
    q.submit([&](sycl::handler& h)
    {
        auto acc = buf.template get_access<sycl::access::mode::write>(h);
        h.single_task([=]()
        {
            for(int i = 0; i < 3; i++)
            {
                for(int j = 0; j < 3; j++)
                {
                    for(int k = 0; k < 3; k++)
                    {
                        acc[i][j][k] = i*9 + j*3 + k;
                    }
                }
            }
        });
    }).wait();

    std::vector<uint32_t> vec_flat(27);

    q.submit([&](sycl::handler& h)
    {
        auto acc = buf.template get_access<sycl::access::mode::read>(h);
        h.copy(acc, vec_flat.data());
    }).wait();


    auto vec = vector_remap(vec_flat, 3, 3, 3);
    //print
    for (auto &i0 : vec)
    {
        for (auto &i1 : i0)
        {
            for (auto &i2 : i1)
            {
                std::cout << i2 << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    return 0;
}
