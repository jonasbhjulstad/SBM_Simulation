#ifndef SYCL_GRAPH_RANDOM_HPP
#define SYCL_GRAPH_RANDOM_HPP

#ifdef SYCL_GRAPH_USE_INTEL_SYCL
#include<oneapi/dpl/random>
#else
#include <random>
#endif
namespace Sycl_Graph::random {
#ifdef SYCL_GRAPH_USE_INTEL_SYCL
    using default_rng = oneapi::dpl::ranlux48;
    using oneapi::dpl::uniform_real_distribution;
#else
    using std::mt19937_64;
    using std::uniform_real_distribution;
    typedef mt19937_64 default_rng;
#endif

    template <typename T, typename RNG = default_rng>
    struct binomial_distribution {
        binomial_distribution(T n, T p) : n(n), p(p) {}
        T n;
        T p;
        T operator()(RNG &rng) {
            T count = 0;
            for (T i = 0; i < n; i++) {
                if (uniform_real_distribution<T>(0, 1)(rng) < p) {
                    count++;
                }
            }
            return count;
        }
        
    };

    template <typename T>
    struct poisson_distribution {
        poisson_distribution(T lambda) : lambda(lambda) {}
        T lambda;
        T operator()(default_rng &rng) {
            T count = 0;
            T p = 1;
            T L = std::exp(-lambda);
            while (p > L) {
                p *= uniform_real_distribution<T>(0, 1)(rng);
                count++;
            }
            return count - 1;
        }
    };
}
#endif