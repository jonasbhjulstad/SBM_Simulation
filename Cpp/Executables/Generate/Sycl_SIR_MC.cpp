
#include "Bernoulli_SIR_MC.hpp"
#include <quantiles.hpp>
#include <FROLS_Path_Config.hpp>
#include <functional>
#include <CL/sycl.hpp>
template <size_t Nt>
void traj_to_file(const FROLS::MC_SIR_Params& p, const FROLS::MC_SIR_SimData<Nt>& d, size_t iter)
{
    FROLS::DataFrame df;
    std::array<double, Nt> p_Is;
    std::array<double, Nt> p_Rs;
    std::fill(p_Rs.begin(), p_Rs.end(), p.p_R);
    std::transform(d.p_vec.begin(), d.p_vec.end(), p_Is.begin(), [](const auto& pv){return pv.p_I;});
    df.assign("S", d.traj[0]);
    df.assign("I", d.traj[1]);
    df.assign("R", d.traj[2]);
    df.assign("p_I", p_Is);
    df.assign("p_R", p_Rs);
    df.assign("t", FROLS::range(0, Nt+1));
    df.write_csv(FROLS::MC_filename(p.N_pop, p.p_ER, iter, "SIR"),
                 ",", p.csv_termination_tol);
}
constexpr size_t Nt = 20;

int main() {
    using namespace FROLS;
//    params[1].N_pop = 20;
//    params[2].p_ER = 0.1;
//    params[3].p_ER = 0.1;
//    params[3].N_pop = 20;
    MC_SIR_Params p;
    p.N_pop = 20;
    p.p_ER = 1.0;

    std::random_device rd{};
    std::vector<size_t> seeds(p.N_sim);
    std::generate(seeds.begin(), seeds.end(), [&](){return rd();});
    auto enum_seeds = FROLS::enumerate(seeds);

    sycl::queue q(sycl::default_selector{});
    sycl::buffer<size_t, 1> seed_buffer{seeds.data(), sycl::range<1>(seeds.size())};

    std::cout << "Running MC-SIR simulations..." << std::endl;
    q.submit([&](sycl::handler& h)
    {
        auto seed = seed_buffer.template get_access<sycl::access_mode::read>(h);
        h.parallel_for<class nstream>(sycl::range<1>{p.N_sim}, [=](sycl::id<1> it)
        {
            const int i = it[0];
            auto simdata = MC_SIR_simulation<Nt>(p, seed[i]);
            traj_to_file(p, simdata, i);

        });
    });

    std::for_each(std::execution::par_unseq, enum_seeds.begin(), enum_seeds.end(), [&](auto& es){
        size_t iter = es.first;
        size_t seed = es.second;
        auto simdata = MC_SIR_simulation<Nt>(p, seed);
        traj_to_file(p, simdata, iter);
    });

    using namespace std::placeholders;
    const std::vector<std::string> colnames = {"S", "I", "R"};
    auto MC_fname_f = std::bind(MC_filename, p.N_pop, p.p_ER, _1, "SIR");
    auto q_fname_f = std::bind(quantile_filename, p.N_pop, p.p_ER, _1, "SIR");
    quantiles_to_file(p.N_sim, colnames, MC_fname_f, q_fname_f);

    return 0;
}