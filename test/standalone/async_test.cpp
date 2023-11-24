#include <future>
#include <iostream>
#include <vector>
#include <algorithm>
void print(uint32_t id)
{
    std::cout <<   "id " << id << std::endl;
}

int main()
{
    //std::async future test
    uint32_t N = 4;
    std::vector<std::future<void>> f(N);
    for(int i = 0; i < 4; i++)
    {
        f[i] = std::async(std::launch::async, [i](){print(i);});
    }

    std::for_each(f.begin(), f.end(), [](auto& f_i){f_i.wait();});
    return 0;
}