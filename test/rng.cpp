#include <Static_RNG/distributions.hpp>
#include <iostream>
#include <CL/sycl.hpp>

int main()
{
    uint32_t N = 10;
    Static_RNG::default_rng rng(100);

    Static_RNG::normal_distribution dist(0.0, 1.0);
    std::vector<Static_RNG::default_rng> rngs;
    for (int i = 0; i < N; i++)
    {
        std::cout << dist(rng) << std::endl;
        rngs.push_back(Static_RNG::default_rng(rng()));
    }
    sycl::queue q(sycl::cpu_selector_v);
    sycl::buffer<Static_RNG::default_rng> rng_buf((sycl::range<1>(N)));
    q.submit([&](sycl::handler& h)
    {
        auto acc = rng_buf.template get_access<sycl::access::mode::write>(h);
        h.copy(rngs.data(), acc);
    }).wait();

    sycl::buffer<float, 2> res(sycl::range<2>(N, N));
    q.submit([&](sycl::handler& h)
    {
        auto rng_acc = rng_buf.template get_access<sycl::access::mode::read_write>(h);
        auto res_acc = res.template get_access<sycl::access::mode::write>(h);
        h.parallel_for(sycl::range<1>(N), [=](sycl::item<1> it)
        {
            Static_RNG::normal_distribution<float> dist(0.0, 1.0);
            auto& r = rng_acc[it];
            for(int i = 0; i < N; i++)
            {
                res_acc[it][i] = dist(r);
            }

        });

    }).wait();

    std::vector<float> res_vec(N*N);
    q.submit([&](sycl::handler& h)
    {
        auto res_acc = res.template get_access<sycl::access::mode::read>(h);
        h.copy(res_acc, res_vec.data());
    }).wait();

    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
            std::cout << res_vec[i*N + j] << " ";
        std::cout << std::endl;
    }


    return 0;
}
