#include <Graph/Graph.hpp>
#include <SIR_Bernoulli_Network.hpp>
int main()
{
    //generate Erdos-Renyi Graph
    Sycl::Graph::Erdos_Renyi_Generator<Network> erdos_renyi_gen(100, 0.5);
    auto graph = erdos_renyi_gen.generate();

}