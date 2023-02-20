#include <Sycl_Graph/Graph/Dynamic/Graph_Types.hpp>
#include <iostream>
int main()
{
    //create boost graph
    using namespace boost;
    using namespace Sycl_Graph::Dynamic;

    //create adjacency list with no data
    typedef adjacency_list<vecS, vecS, bidirectionalS> Adjlist_t;

    std::cout << is_boost_graph<Adjlist_t> << std::endl;

    return 0;

}