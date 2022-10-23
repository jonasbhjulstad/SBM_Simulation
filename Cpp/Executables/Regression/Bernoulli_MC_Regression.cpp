
#include "../Generate/Bernoulli_SIR_MC.hpp"
#include <quantiles.hpp>
#include <FROLS_Path_Config.hpp>
#include <FROLS_Graph.hpp>
#include <FROLS_Execution.hpp>
#include <functional>
#include <utility>
#include <chrono>

#include <FROLS_Eigen.hpp>
#include <Quantile_Regressor.hpp>
#include <ERR_Regressor.hpp>
#include <Regression_Algorithm.hpp>
#include <Polynomial_Discrete.hpp>
#include <memory>

std::string err_simulation_filename(uint32_t N_pop, float p_ER, uint32_t iter, std::string network_type)
{
    std::stringstream ss;
    ss << FROLS::FROLS_DATA_DIR << "/ERR_Simulation_" << network_type << "_" << N_pop << "_" << p_ER << "/trajectory_" << iter
       << ".csv";
    return ss.str();
}

std::string quantile_simulation_filename(uint32_t N_pop, float p_ER, uint32_t iter, std::string network_type)
{
    std::stringstream ss;
    ss << FROLS::FROLS_DATA_DIR << "/Quantile_Simulation_" << network_type << "_" << N_pop << "_" << p_ER << "/trajectory_" << iter
       << ".csv";
    return ss.str();
}

template <uint32_t Nt, typename dType = float>
void traj_to_file(const FROLS::MC_SIR_Params<> &p, const FROLS::MC_SIR_SimData<Nt> &d, uint32_t iter)
{
    // print iter
    FROLS::DataFrame df;
    std::array<dType, Nt + 1> p_Is;
    std::transform(d.p_vec.begin(), d.p_vec.end(), p_Is.begin(), [](auto &p)
                   { return p.p_I; });
    std::array<dType, Nt + 1> p_Rs;
    std::fill(p_Rs.begin(), p_Rs.end(), p.p_R);
    p_Is.back() = 0.;
    df.assign("S", d.traj[0]);
    df.assign("I", d.traj[1]);
    df.assign("R", d.traj[2]);
    df.assign("p_I", p_Is);
    df.assign("p_R", p_Rs);
    auto t = FROLS::range(0, Nt + 1);
    df.assign("t", t);
    df.resize(Nt + 1);
    df.write_csv(FROLS::MC_filename(p.N_pop, p.p_ER, iter, "SIR"),
                 ",", p.csv_termination_tol);
}

constexpr uint32_t Nt = 50;
constexpr uint32_t N_sims = 50;
const std::string network_type = "SIR";
void simulation_loop(uint32_t N_pop, float p_ER)
{

    using namespace std::placeholders;

    auto MC_fname_f = std::bind(FROLS::MC_filename, N_pop, p_ER, _1, network_type);
    auto er_outfile_f = std::bind(err_simulation_filename, N_pop, p_ER, _1, network_type);
    auto qr_outfile_f = std::bind(quantile_simulation_filename, N_pop, p_ER, _1, network_type);

    using namespace FROLS;
    using namespace Network_Models;
    MC_SIR_Params<> p;
    p.N_pop = N_pop;
    p.p_ER = p_ER;
    uint32_t NV = N_pop;
    size_t nk = FROLS::n_choose_k(NV, 2);
    uint32_t NE = 1.5 * nk;
    std::cout << "Simulating for network with N_pop = " << N_pop << " and p_ER = " << p_ER << std::endl;

    // mark start timestamp
    auto start = std::chrono::high_resolution_clock::now();

    std::random_device rd{};
    std::vector<uint32_t> seeds(p.N_sim);
    std::generate(seeds.begin(), seeds.end(), [&]()
                  { return rd(); });
    auto enum_seeds = enumerate(seeds);

    std::mt19937_64 rng(rd());
    typedef Network_Models::SIR_Bernoulli_Network<SIR_VectorGraph, decltype(rng), Nt> SIR_Bernoulli_Network;
    std::vector<std::shared_ptr<std::mutex>> v_mx(NV + 1);
    // create mutexes
    for (auto &mx : v_mx)
    {
        mx = std::make_unique<std::mutex>();
    }
    std::vector<std::shared_ptr<std::mutex>> e_mx(NE + 1);
    // create mutexes
    for (auto &mx : e_mx)
    {
        mx = std::make_unique<std::mutex>();
    }

    SIR_VectorGraph G(v_mx, e_mx);
    generate_erdos_renyi<SIR_VectorGraph, decltype(rng)>(G, p.N_pop, p.p_ER, SIR_S, rng);
    std::vector<MC_SIR_SimData<Nt>> simdatas(p.N_sim);
    std::transform(enum_seeds.begin(), enum_seeds.end(), simdatas.begin(), [&](auto &es)
                   {
        uint32_t iter = es.first;
        uint32_t seed = es.second;
        if ((iter % (p.N_sim / 10)) == 0)
        {
            std::cout << "Simulation " << iter << " of " << p.N_sim << std::endl;
        }
        return MC_SIR_simulation<decltype(G), Nt>(G, p, seed); });

    std::for_each(simdatas.begin(), simdatas.end(), [&, n = 0](const auto &simdata) mutable
                  { traj_to_file<Nt>(p, simdata, n++); });

    const std::vector<std::string> colnames = {"S", "I", "R", "p_I"};
    auto q_fname_f = std::bind(quantile_filename, p.N_pop, p.p_ER, _1, "SIR");
    quantiles_to_file(p.N_sim, colnames, MC_fname_f, q_fname_f);

    // mark end timestamp
    auto end = std::chrono::high_resolution_clock::now();
    // print time
    std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;


    std::cout << "Performing ERR-Regression" << std::endl;
    // ERR-Regression

    uint32_t d_max = 2;
    uint32_t N_output_features = 16;
    uint32_t Nu = 1;
    uint32_t Nx = 3;
    std::vector<std::vector<Feature>> preselected_features(4);
    FROLS::Features::Polynomial_Model model(Nx, Nu, N_output_features, d_max);
    FROLS::Regression::Regressor_Param er_param;
    er_param.tol = 1e-4;
    er_param.theta_tol = 1e-10;
    er_param.N_terms_max = 4;
    FROLS::Regression::ERR_Regressor er_regressor(er_param);

    Regression::from_file_regression(MC_fname_f, colnames, {"p_I"}, N_sims, er_regressor, model, er_outfile_f, true);

    model.write_csv(path_dirname(er_outfile_f(0).c_str()) + std::string("/param.csv"));

    std::cout << "Performing Quantile-Regression" << std::endl;

    // Quantile-Regression

    float MAE_tol = 1e-6;
    FROLS::Regression::Quantile_Param qr_param;
    qr_param.N_terms_max = 4;
    qr_param.tol = MAE_tol;
    qr_param.tau = 0.95;

    FROLS::Regression::Quantile_Regressor qr_regressor(qr_param);

    from_file_regression(MC_fname_f, colnames, {"p_I"}, N_sims, qr_regressor, model, qr_outfile_f);
    model.write_csv(path_dirname(qr_outfile_f(0).c_str()) + std::string("/param.csv"));
}

int main(int argc, char **argv)
{
    auto N_pop_vec = FROLS::arange((uint32_t)10, (uint32_t)100, (uint32_t)10);
    std::vector<float> p_ER_vec = {0.1, 1.0};
    for (const float &p_ER : p_ER_vec)
    {
        for (const uint32_t N_pop : N_pop_vec)
        {
            simulation_loop(N_pop, p_ER);
        }
    }

    return 0;
}
