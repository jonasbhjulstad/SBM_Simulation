
#include "Bernoulli_SIR_MC.hpp"
#include <quantiles.hpp>
#include <FROLS_Path_Config.hpp>
#include <FROLS_Graph.hpp>
#include <FROLS_Random.hpp>
#include <functional>
#include <FROLS_Sycl.hpp>
#include <Sycl_SIR_Bernoulli_Network.hpp>

template <uint32_t Nt>
void traj_to_file(const FROLS::MC_SIR_Params<> &p, const FROLS::MC_SIR_ArrayData<Nt> &d, uint32_t iter)
{
    FROLS::DataFrame df;
    std::array<float, Nt> p_Is;
    std::array<float, Nt> p_Rs;
    std::fill(p_Rs.begin(), p_Rs.end(), p.p_R);
    std::transform(d.p_vec.begin(), d.p_vec.end(), p_Is.begin(), [](const auto &pv)
                   { return pv.p_I; });
    df.assign("S", d.traj[0]);
    df.assign("I", d.traj[1]);
    df.assign("R", d.traj[2]);
    df.assign("p_I", p_Is);
    df.assign("p_R", p_Rs);
    df.assign("t", FROLS::range(0, Nt + 1));
    df.write_csv(FROLS::MC_filename(p.N_pop, p.p_ER, iter, "SIR"),
                 ",", p.csv_termination_tol);
}

constexpr uint32_t N_pop = 20;
constexpr float p_ER = 1.0;
constexpr uint32_t Nt = 20;
constexpr uint32_t NV = N_pop;
constexpr uint32_t NE = 1000 * NV * NV;

int main()
{
    using namespace FROLS;
    using namespace Network_Models;
    MC_SIR_Params<> p;
    p.N_pop = 20;
    p.p_ER = 1.0;

    std::random_device rd{};
    std::mt19937_64 rng(rd());

    FROLS::random::default_rng generator(rng());
    // Network_Models::SIR_Bernoulli_Network<decltype(generator), Nt, NV, NE> G(G_structure, p.p_I0, p.p_R0,
    //  generator);
    MC_SIR_ArrayData<Nt> data;

    data.p_vec = generate_interaction_probabilities<decltype(generator), Nt>(p, generator);

    // list available sycl devices
    auto devices = sycl::device::get_devices();
    for (auto &dev : devices)
    {
        std::cout << dev.template get_info<sycl::info::device::name>() << std::endl;
    }
    const size_t ER_seed = 777;
    sycl::queue q(sycl::cpu_selector{});
    std::vector<MC_SIR_ArrayData<Nt>> sim_data(p.N_sim);
    sycl::buffer<MC_SIR_ArrayData<Nt>, 1> sim_buffer{sim_data.data(), sycl::range<1>(sim_data.size())};
    sycl::buffer<MC_SIR_Params<>, 1> param_buffer{&p, sycl::range<1>(1)};
    std::vector<FROLS::random::default_rng> rng_vec(NV);
    std::generate(rng_vec.begin(), rng_vec.end(), [&]()
                  { return FROLS::random::default_rng(rd()); });
    sycl::buffer<FROLS::random::default_rng, 1> rng_buffer{rng_vec.data(), sycl::range<1>(NV)};
    std::cout << "Running MC-SIR simulations..." << std::endl;

    std::array<size_t, NV> seeds;
    std::generate(seeds.begin(), seeds.end(), [&]()
                  { return generator(); });
    SIR_VectorGraph G(v_mx, e_mx);
    generate_erdos_renyi<SIR_VectorGraph, decltype(generator)>(G_structure, p.N_pop, p.p_ER, SIR_S, generator);
    auto network = Network_Models::Sycl_SIR_Bernoulli_Network<FROLS::random::default_rng, Nt, NV, NE>{G_structure, p.p_I0, p.p_R0, q, seeds};
    network.reset();
    while (network.population_count()[1] == 0)
    {
        network.initialize();
    }

    auto trajectory = network.simulate(sim_data[0].p_vec);

std::for_each(sim_data.begin(), sim_data.end(), [&, n = 0](auto &sim) mutable
              { traj_to_file(p, sim, n++); });

using namespace std::placeholders;
const std::vector<std::string> colnames = {"S", "I", "R"};
auto MC_fname_f = std::bind(MC_filename, p.N_pop, p.p_ER, _1, "SIR");
auto q_fname_f = std::bind(quantile_filename, p.N_pop, p.p_ER, _1, "SIR");
quantiles_to_file(p.N_sim, colnames, MC_fname_f, q_fname_f);

return 0;
}