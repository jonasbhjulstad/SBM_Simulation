
#include "Bernoulli_SIR_MC.hpp"
#include <quantiles.hpp>
#include <FROLS_Path_Config.hpp>
#include <FROLS_Graph.hpp>
#include <FROLS_Execution.hpp>
#include <functional>
// #include <oneapi/dpl/algorithm>
template <size_t Nt, typename dType=float>
void traj_to_file(const FROLS::MC_SIR_Params<>& p, const FROLS::MC_SIR_SimData<Nt>& d, size_t iter)
{
    //print iter
    FROLS::DataFrame df;
    std::array<dType, Nt+1> p_Is;
    std::array<dType, Nt+1> p_Rs;
    std::fill(p_Rs.begin(), p_Rs.end(), p.p_R);
    std::transform(d.p_vec.begin(), d.p_vec.end(), p_Is.begin(), [](const auto& pv){return pv.p_I;});
    p_Is.back() = 0.;
    df.assign("S", d.traj[0]);
    df.assign("I", d.traj[1]);
    df.assign("R", d.traj[2]);
    df.assign("p_I", p_Is);
    df.assign("p_R", p_Rs);
    df.assign("t", FROLS::range(0, Nt+1));
    df.write_csv(FROLS::MC_filename(p.N_pop, p.p_ER, iter, "SIR"),
                 ",", p.csv_termination_tol);
}
constexpr size_t N_pop = 20;
constexpr float p_ER = 1.0;
constexpr size_t Nt = 20;
constexpr size_t NV = N_pop;
constexpr size_t NE = NV*NV;
int main() {
    using namespace FROLS;
    using namespace Network_Models;

//    params[1].N_pop = 20;
//    params[2].p_ER = 0.1;
//    params[3].p_ER = 0.1;
//    params[3].N_pop = 20;
    MC_SIR_Params<>p;
    p.N_pop = N_pop;
    p.p_ER = p_ER;

    std::random_device rd{};
    std::vector<size_t> seeds(p.N_sim);
    std::generate(seeds.begin(), seeds.end(), [&](){return rd();});
    auto enum_seeds = enumerate(seeds);
    std::cout << "Running MC-SIR simulations..." << std::endl;
    std::mutex mx;
    size_t MC_iter = 0;
    std::mt19937_64 rng(rd());

    typedef Network_Models::SIR_Bernoulli_Network<decltype(rng), Nt, N_pop, NE> SIR_Bernoulli_Network;
//    auto policy_d = make_device_policy<class PolicyD>(oneapi::dpl::execution::dpcpp_default);

    auto G = generate_erdos_renyi<SIR_Graph<NV, NE>, decltype(rng)>(p.N_pop, p.p_ER, SIR_S, rng);
    std::vector<MC_SIR_SimData<Nt>> simdatas(p.N_sim);
    std::transform(enum_seeds.begin(), enum_seeds.end(), simdatas.begin(), [=](auto& es){
        size_t iter = es.first;
        size_t seed = es.second;
        return MC_SIR_simulation<Nt, NV, NE>(G, p, seed);
//        traj_to_file(p, simdata, iter);
//        std::lock_guard<std::mutex> lg(mx);
//        MC_iter++;
//        if (!(MC_iter % (p.N_sim / 10)))
//        {
//            std::cout << "Simulation " << MC_iter << " of " << p.N_sim << std::endl;
//        }
    });

    std::for_each(simdatas.begin(), simdatas.end(), [&, n= 0](const auto& simdata)mutable
    {
        traj_to_file<Nt>(p, simdata, n++);
    });

    using namespace std::placeholders;
    const std::vector<std::string> colnames = {"S", "I", "R"};
    auto MC_fname_f = std::bind(MC_filename, p.N_pop, p.p_ER, _1, "SIR");
    auto q_fname_f = std::bind(quantile_filename, p.N_pop, p.p_ER, _1, "SIR");
    quantiles_to_file(p.N_sim, colnames, MC_fname_f, q_fname_f);

    return 0;
}