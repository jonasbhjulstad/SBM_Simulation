#ifndef SYCL_GRAPH_RANDOM_HPP
#define SYCL_GRAPH_RANDOM_HPP

#ifdef SYCL_GRAPH_USE_ONEAPI
#include<oneapi/dpl/random>
#else
#include <random>
#include <tinymt/tinymt.h>
#endif
namespace Sycl_Graph::random {
#ifdef SYCL_GRAPH_USE_ONEAPI
    #ifdef cl_khr_fp64
    using oneapi::dpl::uniform_real_distribution;
    using oneapi::dpl::bernoulli_distribution;
    using oneapi::dpl::normal_distribution;
    using default_rng = oneapi::dpl::minstd_rand;
    #else
    using default_rng = tinymt::tinymt32;
    #endif
#else
    template <typename dType = float>
    struct uniform_real_distribution
    {
        uniform_real_distribution() = default;
        uniform_real_distribution(dType a, dType b): a(a), b(b){}
        dType a = 0; dType b = 1;
        template <typename RNG = tinymt::tinymt32>
        float operator()(RNG &rng) {
            auto val = rng();
            //convert val to random uniform float
            return ((dType)val / (dType)rng.max()) * (b-a) + a;
        }
        void set_a(dType a) { this->a = a; }
        void set_b(dType b) { this->b = b; }
    };
    template <typename dType = float>
    struct bernoulli_distribution
    {
        bernoulli_distribution(dType p): p(p){}
        dType p = 0;
        template <typename RNG = tinymt::tinymt32>
        float operator()(RNG &rng) {
            auto val = rng();
            float uniform = (val / (dType)rng.max());
            return uniform < p;
        }
    };

    template <typename dType = float>
    struct normal_distribution
    {
        normal_distribution() = default;
        normal_distribution(dType mean, dType stddev): mean(mean), stddev(stddev){}
        dType mean = 0; dType stddev = 1;
        template <typename RNG = tinymt::tinymt32>
        float operator()(RNG &rng) {
            auto val = rng();
            //convert val to random uniform float
            dType uniform = (val / (dType)rng.max());
            return std::sqrt(-2 * std::log(uniform)) * std::cos(2 * M_PIf * uniform) * stddev + mean;
        }
    };
    template <typename T = float>
    struct binomial_distribution {
        binomial_distribution(T n, T p) : n(n), dist(p) {}
        bernoulli_distribution<T> dist;
        T n;
        template <typename RNG = tinymt::tinymt32>
        T operator()(RNG &rng) {
            T count = 0;
            for (T i = 0; i < n; i++) {
                count += dist(rng);
            }
            return count;
        }
        void set_trials(T n) {
            this->n = n;
        }
        void set_probability(T p) {
            dist.p = p;
        }
        
    };

    template <typename T>
    struct poisson_distribution {
        poisson_distribution(T lambda) : lambda(lambda) {}
        T lambda;
        template <typename RNG = tinymt::tinymt32>
        T operator()(RNG &rng) {
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
    typedef tinymt::tinymt32 default_rng;
#endif

}
#endif