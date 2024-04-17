#include <iostream>
#include <unordered_set>
#include <algorithm>
#include <vector>
#include <numeric>
struct DummyHash
{
    size_t operator()(const std::vector<int>& key) const
    {
        return std::reduce(key.begin(), key.end());
    }
};

int main()
{
    return 0;
}