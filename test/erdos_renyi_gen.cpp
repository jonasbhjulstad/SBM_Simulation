#include <Sycl_Graph.hpp>
#include <SIR_Bernoulli_Network.hpp>
#include <Sycl/Sycl_Graph_Random.hpp>
int main()
{

    using namespace Network_Models::Dynamic;
    using namespace Network_Models;
//   SIR_Bernoulli_Network(SIR_Graph &G, dType p_I0, dType p_R0,
                            //    RNG rng)
    Sycl::Graph::random::default_rng rng;
    SIR_Graph G(100, 100);
    SIR_Bernoulli_Network sir(G, 0.1, 0.1, rng);
    //generate sir_param
    size_t Nt = 100;
    std::vector<SIR_Param<float>> sir_param(Nt);
    std::generate(sir_param.begin(), sir_param.end(), [&]() {
        return SIR_Param<float>{0.1, 0.1, 100, 10};
    });



    sir.initialize();
    auto traj = sir.simulate(sir_param, Nt);
    //create random number generator
    //create SIR_bernoulli network



}