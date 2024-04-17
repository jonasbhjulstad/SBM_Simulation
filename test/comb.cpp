#include <SIR_SBM/combination.hpp>
#include <iostream>

int main()
{
    auto c = combinations_with_replacement(10, 2);

    for(auto && ci: c)
    {
        std::cout << ci[0] << "," << ci[1] << "\n";
    }
    return 0;
}