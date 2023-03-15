#ifndef SYCL_GRAPH_MATH_MATH_HPP
#define SYCL_GRAPH_MATH_MATH_HPP
#include <cstddef>
#include <stdint.h>
#include <vector>
#include <array>
#include <algorithm>
namespace Sycl_Graph
{

    auto n_choose_k(uint32_t n, uint32_t k) -> uint32_t;

    auto linspace(float min, float max, int N) -> std::vector<float>;
    constexpr auto n_choose_k(std::size_t n, std::size_t k) -> std::size_t
    {
        return (k == 0) ? 1 : (n * n_choose_k(n - 1, k - 1)) / k;
    }
    template <typename T>
    auto arange(T min, T max, T step) -> std::vector<T>
    {
        T s = min;
        std::vector<T> res;
        while (s <= max)
        {
            res.push_back(s);
            s += step;
        }
        return res;
    }
    template <typename T = uint32_t>
    auto range(auto start, auto end) -> std::vector<T>
    {
        std::vector<T> res(end - start);
        for (T i = start; i < end; i++)
        {
            res[i - start] = i;
        }
        return res;
    }

    template <typename T>
    inline auto zip(const std::vector<T> &a, const std::vector<T> &b) -> std::vector<std::pair<T, T>>
    {
        std::vector<std::pair<T, T>> res(a.size());
        for (int i = 0; i < a.size(); i++)
        {
            res[i] = std::make_pair(a[i], b[i]);
        }
        return res;
    }

    template <typename T>
    inline auto unzip(const std::vector<std::pair<T, T>> &a) -> std::pair<std::vector<T>, std::vector<T>>
    {
        std::pair<std::vector<T>, std::vector<T>> res;
        res.first.reserve(a.size());
        res.second.reserve(a.size());
        for (int i = 0; i < a.size(); i++)
        {
            res.first.push_back(a[i].first);
            res.second.push_back(a[i].second);
        }
        return res;
    }

    template <typename T>
    inline auto transpose(const std::vector<std::vector<T>> &vec) -> std::vector<std::vector<T>>
    {
        // transpose vector
        std::vector<std::vector<T>> res(vec[0].size());
        for (int i = 0; i < vec[0].size(); i++)
        {
            res[i].reserve(vec.size());
            for (int j = 0; j < vec.size(); j++)
            {
                res[i].push_back(vec[j][i]);
            }
        }
        return res;
    }

    template <typename T, std::size_t N0, std::size_t N1>
    inline auto transpose(const std::array<std::array<T, N0>, N1> &vec) -> std::array<std::array<T, N1>, N0>
    {
        std::array<std::array<T, N1>, N0> res;
        for (int i = 0; i < N1; i++)
        {
            for (int j = 0; j < N0; j++)
            {
                res[j][i] = vec[i][j];
            }
        }
        return res;
    }

    auto filtered_range(const std::vector<uint32_t> &filter_idx, uint32_t min, uint32_t max) -> std::vector<uint32_t>;

    template <uint32_t N, typename T, typename T_out = T>
    auto diff(const std::array<T, N> &vec) -> std::array<T_out, N - 1>
    {
        std::array<T_out, N - 1> res;
        for (int i = 0; i < N; i++)
        {
            res[i] = (T_out)vec[i + 1] - (T_out)vec[i];
        }
        return res;
    }

    template <uint32_t N, typename T, typename T_out = T>
    auto integrate(const T v0, const std::array<T, N - 1> &vec) -> std::array<T_out, N>
    {
        std::array<T_out, N> res;
        res[0] = v0;
        for (int i = 0; i < N; i++)
        {
            res[i + 1] = res[i] + vec[i];
        }
        return res;
    }

    template <typename T>
    auto enumerate(const std::vector<T> &data) -> std::vector<std::pair<uint32_t, T>>
    {
        std::vector<std::pair<uint32_t, T>> enumerated_data(data.size());
        std::transform(data.begin(), data.end(), enumerated_data.begin(), [&, n = -1](const auto &d) mutable
                       { return std::make_pair(n++, d); });
        return enumerated_data;
    }
} // namespace Sycl_Graph

#endif