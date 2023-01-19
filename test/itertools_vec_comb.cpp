#include <itertools.hpp>
#include <vector>
#include <iostream>
int main()
{
    std::vector<int> a = {1, 2, 3, 4, 5};
    std::vector<int> b = {6, 7, 8, 9, 10};
    auto c = iter::product(a, b);
    for (auto &&[i0, i1] : iter::product(a, b)) {
        std::cout << i0 << " " << i1 << std::endl;
    }

    return 0;
}