#include <Sycl_Graph/Algorithms/Generation/Graph_Generation.hpp>

int main()
{
    auto edges = Sycl_Graph::Dynamic::Network_Models::generate_planted_partition(100, 10, 1.0,0.1,false);

    return 0;
}