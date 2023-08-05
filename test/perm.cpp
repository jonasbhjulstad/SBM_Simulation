#include <algorithm>
#include <iostream>
#include <string>
#include <itertools.hpp>
#include <cstdint>
int main()
{
    std::vector<uint32_t> num = {1, 2, 3, 4, 5};
    std::cout << "Combination: \n";
    for(auto&& i : iter::combinations(num, 2))
    {
        std::cout << i[0] << " " << i[1] << std::endl;
    }
    std::cout << "Combination_with_replacement: \n";
    for(auto&& i : iter::combinations_with_replacement(num, 2))
    {
        std::cout << i[0] << " " << i[1] << std::endl;
    }

    std::cout << "Product: \n";
    for(auto&& i: iter::product(num, num))
    {
        std::cout << std::get<0>(i) << " " << std::get<1>(i) << std::endl;
    }

    return 0;
}
