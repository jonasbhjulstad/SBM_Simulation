
#include "Bernoulli_SIR_MC.hpp"
#include <quantiles.hpp>
#include <FROLS_Path_Config.hpp>
#include <FROLS_Graph.hpp>
#include <FROLS_Random.hpp>
#include <functional>
#include <CL/sycl.hpp>

template<size_t Nt>
void traj_to_file(const FROLS::MC_SIR_Params<>&p, const FROLS::MC_SIR_SimData<Nt> &d, size_t iter) {
    FROLS::DataFrame df;
    std::array<double, Nt> p_Is;
    std::array<double, Nt> p_Rs;
    std::fill(p_Rs.begin(), p_Rs.end(), p.p_R);
    std::transform(d.p_vec.begin(), d.p_vec.end(), p_Is.begin(), [](const auto &pv) { return pv.p_I; });
    df.assign("S", d.traj[0]);
    df.assign("I", d.traj[1]);
    df.assign("R", d.traj[2]);
    df.assign("p_I", p_Is);
    df.assign("p_R", p_Rs);
    df.assign("t", FROLS::range(0, Nt + 1));
    df.write_csv(FROLS::MC_filename(p.N_pop, p.p_ER, iter, "SIR"),
                 ",", p.csv_termination_tol);
}

constexpr size_t N_pop = 20;
constexpr double p_ER = 1.0;
constexpr size_t Nt = 20;
constexpr size_t NV = N_pop;
constexpr size_t NE = NV * NV;

int main() {
    using namespace FROLS;
    using namespace Network_Models;
    MC_SIR_Params<>p;
    p.N_pop = 20;
    p.p_ER = 1.0;

    std::random_device rd{};
    std::vector<size_t> seeds(p.N_sim);
    std::generate(seeds.begin(), seeds.end(), [&]() { return rd(); });
    std::mt19937_64 rng(rd());
    auto G_structure = generate_erdos_renyi<SIR_Graph<NV, NE>, decltype(rng)>(p.N_pop, p.p_ER, SIR_S, rng);

    oneapi::dpl::ranlux48 generator(seeds[0]);
    Network_Models::SIR_Bernoulli_Network<decltype(generator), Nt, NV, NE> G(G_structure, p.p_I0, p.p_R0,
                                                                             generator);
    MC_SIR_SimData<Nt> data;
    G.reset();
    while (G.population_count()[1] == 0) {
        G.initialize();
    }

    data.p_vec = generate_interaction_probabilities<decltype(generator), Nt>(p, generator);

    //list available sycl devices
    auto devices = cl::sycl::device::get_devices();
    for (auto &dev: devices) {
        std::cout << dev.get_info<cl::sycl::info::device::name>() << std::endl;
    }
    sycl::queue q(sycl::default_selector{});
    sycl::buffer<size_t, 1> seed_buffer{seeds.data(), sycl::range<1>(seeds.size())};
    std::vector<MC_SIR_SimData<Nt>> sim_data(p.N_sim);
    sycl::buffer<MC_SIR_SimData<Nt>, 1> sim_buffer{sim_data.data(), sycl::range<1>(sim_data.size())};
    sycl::buffer<MC_SIR_Params<>, 1> param_buffer{&p, sycl::range<1>(1)};
    sycl::buffer<SIR_Graph<NV, NE>, 1> graph_buffer{&G_structure, sycl::range<1>(1)};
    std::cout << "Running MC-SIR simulations..." << std::endl;
    q.submit([&](sycl::handler &h) {
        auto seed = seed_buffer.template get_access<sycl::access_mode::read>(h);
        auto sim = sycl::accessor{sim_buffer, h, sycl::write_only};
        auto params = sycl::accessor{param_buffer, h, sycl::read_only};
        auto graph = sycl::accessor{graph_buffer, h, sycl::read_only};
        h.parallel_for<class nstream>(sycl::range<1>{p.N_sim}, [=](sycl::id<1> it) {
            const int i = it[0];
            sim[i] = MC_SIR_simulation<Nt>(graph[0], params[0], seed[i]);

        });

    });

    std::for_each(sim_data.begin(), sim_data.end(), [&, n=0](auto &sim) mutable{
        traj_to_file(p, sim, n++);
    });

    using namespace std::placeholders;
    const std::vector<std::string> colnames = {"S", "I", "R"};
    auto MC_fname_f = std::bind(MC_filename, p.N_pop, p.p_ER, _1, "SIR");
    auto q_fname_f = std::bind(quantile_filename, p.N_pop, p.p_ER, _1, "SIR");
    quantiles_to_file(p.N_sim, colnames, MC_fname_f, q_fname_f);

    return 0;
}