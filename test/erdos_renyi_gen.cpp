#include <Sycl_Graph/Graph/Graph.hpp>
#include <Sycl_Graph/Network/SIR_Bernoulli_Network.hpp>
#include <Sycl_Graph/random.hpp>
int main()
{

    using namespace Network_Models::Dynamic;
    using namespace Network_Models;
    size_t N_pop = 100;
    float p_ER = 0.1;

    Sycl_Graph::random::default_rng rng;
    SIR_Graph G(100, 1000);
    SIR_Bernoulli_Network sir(G, 0.1, 0.1, rng);
    //generate sir_param
    size_t Nt = 100;
    std::vector<SIR_Param<float>> sir_param(Nt);
    std::generate(sir_param.begin(), sir_param.end(), [&]() {
        return SIR_Param<float>{0.1, 0.1, 100, 10};
    });

    generate_erdos_renyi(G, N_pop, p_ER, SIR_S, rng);

    sir.initialize();
    auto traj = sir.simulate(sir_param,Nt);
    //print traj
    for (auto &x : traj) {
        for (auto &y : x) {
            std::cout << y << " ";
        }
        std::cout << std::endl;
    }
    //create random number generator
    //create SIR_bernoulli network



}