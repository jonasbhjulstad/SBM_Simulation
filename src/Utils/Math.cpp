#include <Sycl_Graph/Utils/Math.hpp>
long long n_choose_k
{
    long long product = 1;
    for (int i = 1; i <= k; i++)
        product = product * (n - k + i) / i;
    return product;
}
