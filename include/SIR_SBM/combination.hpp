#pragma once
#include <SIR_SBM/common.hpp>
#include <string>
#include <unordered_map>
 
size_t n_choose_k(size_t n, size_t k)
{
    if (k > n)
    {
        return 0;
    }
    if (k * 2 > n)
    {
        k = n - k;
    }
    if (k == 0)
    {
        return 1;
    }
 
    size_t result = n;
    for (size_t i = 2; i <= k; ++i)
    {
        result *= (n - i + 1);
        result /= i;
    }
    return result;
}

std::vector<std::vector<int>> combinations_with_replacement(int n, int k)
{
    if(k == 0)
    {
        return {{}};
    }
    if(k == 1)
    {
        std::vector<std::vector<int>> res(n);
        for(int i = 0; i < n; i++)
        {
            res[i] = {i};
        }
        return res;
    }

    auto res = combinations_with_replacement(n, k-1);

    std::vector<std::vector<int>> new_res;
    for(auto& v : res)
    {
        for(int i = v.back(); i < n; i++)
        {
            auto new_v = v;
            new_v.push_back(i);
            new_res.push_back(new_v);
        }
    }

    return new_res;
}