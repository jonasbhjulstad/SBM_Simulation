#include <oneapi/dpl/execution>
#include <oneapi/dpl/pstl/execution_defs.h>
namespace sycl::property{
    struct noinit{};
}
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <mutex>
#include <iostream>
// #include <oneapi/dpl/pstl/hetero/dpcpp/execution_sycl_defs.h>
#include <execution>

int main()
{
    std::mutex m;
    std::vector<int> v(10000000);
    int j = 0;
    // std::for_each(oneapi::dpl::execution::par_unseq,v.begin(), v.end(), [&](int& i)
    std::for_each(std::execution::par_unseq,v.begin(), v.end(), [&](int& i)
    {   
        // std::lock_guard g(m);
        j++;
    });
    std::cout << j << std::endl;
    return 0;
}