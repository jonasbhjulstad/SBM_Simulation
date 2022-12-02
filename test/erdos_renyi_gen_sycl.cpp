#include <Sycl_Graph/Graph/Graph.hpp>
#include <Sycl_Graph/Network/SIR_Bernoulli/SIR_Bernoulli_Network.hpp>
#include <Sycl_Graph/Graph/Graph_Generation.hpp>
#include <Sycl_Graph/random.hpp>
static constexpr uint32_t N_VERTICES = 100;
static constexpr uint32_t Nt = 100;
static constexpr uint32_t N_EDGES = N_VERTICES * N_VERTICES;
int main()
{
    using namespace Sycl_Graph::Network_Models;
    using namespace Sycl_Graph::Sycl::Network_Models;
    using RNG = Sycl_Graph::random::default_rng;
    using Network_t = SIR_Bernoulli_Network<RNG, Nt, N_VERTICES, N_EDGES, float>;
    size_t N_pop = 100;
    float p_ER = 0.1;

    //create a sycl queue
    sycl::queue q;


    RNG rng;
    SIR_Graph<N_VERTICES, N_EDGES> G(q);

    Network_t sir(G, 0.1f, 0.1f, rng);
    // generate sir_param
    std::array<SIR_Bernoulli_Param<float>, Nt> sir_param;
    std::generate(sir_param.begin(), sir_param.end(), [&]()
                  { return SIR_Bernoulli_Param<float>{0.1, 0.1, 100, 10}; });

    Sycl_Graph::Dynamic::Network_Models::generate_erdos_renyi(G, N_pop, p_ER, SIR_S, rng);

    sir.initialize();
    auto traj = sir.simulate(sir_param, Nt);
    // print traj
    for (auto &x : traj)
    {
        for (auto &y : x)
        {
            std::cout << y << " ";
        }
        std::cout << std::endl;
    }
    // create random number generator
    // create SIR_bernoulli network
}