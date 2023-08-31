#include <Sycl_Graph/Regression.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <tuple>
#include <utility>
#include <ortools/linear_solver/linear_solver.h>

Eigen::MatrixXf openData(const std::string& fileToOpen)
{
    using namespace Eigen;
    std::vector<float> matrixEntries;

    std::ifstream matrixDataFile(fileToOpen);

    std::string matrixRowString;

    std::string matrixEntry;

    int matrixRowNumber = 0;

    while (std::getline(matrixDataFile, matrixRowString)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
    {
        std::stringstream matrixRowStringStream(matrixRowString); // convert matrixRowString that is a string to a stream variable.

        while (std::getline(matrixRowStringStream, matrixEntry, ',')) // here we read pieces of the stream matrixRowStringStream until every comma, and store the resulting character into the matrixEntry
        {
            matrixEntries.push_back(stod(matrixEntry)); // here we convert the string to double and fill in the row vector storing all the matrix entries
        }
        matrixRowNumber++; // update the column numbers
    }

    return Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
}

std::pair<Eigen::MatrixXf, Eigen::MatrixXf> load_beta_regression(const std::string& datapath, uint32_t idx)
{
    using Mat = Eigen::MatrixXf;
    using Vec = Eigen::VectorXf;
    auto community_infs_filename = [](uint32_t idx){return std::string("community_infections_" + std::to_string(idx) + ".csv");};

    auto connection_infs_filename = [](uint32_t idx){return std::string("connection_infections_" + std::to_string(idx) + ".csv");};

    auto community_traj_filename = [](uint32_t idx){return std::string("community_trajectory_" + std::to_string(idx) + ".csv");};

    auto infection_events_filename = [](uint32_t idx){return std::string("connection_events_" + std::to_string(idx) + ".csv");};
    auto connection_community_map_filename = [](uint32_t idx){return std::string("ccm.csv");};

    auto p_Is_filename = [](uint32_t idx){return std::string("p_I_" + std::to_string(idx) + ".csv");};


    Mat connection_infs = openData(datapath + connection_infs_filename(idx));
    Mat connection_community_map = openData(datapath + connection_community_map_filename(idx));
    Mat community_traj = openData(datapath + community_traj_filename(idx));

    assert((connection_infs.array() <= 1000).all());
    assert((connection_infs.array() >= 0).all());
    assert((community_traj.array() <= 1000).all());
    auto N_connections = connection_infs.cols();
    auto Nt = community_traj.rows() - 1;
    uint32_t const N_communities = community_traj.cols() / 3;
    Mat p_Is = openData(datapath + p_Is_filename(idx));

    Mat F_beta_rs_mat(Nt, N_connections);
    for (int i = 0; i < connection_community_map.rows(); i++)
    {
        auto target_idx = connection_community_map(i, 1);
        auto source_idx = connection_community_map(i, 0);
        // community_traj[:, 3*target_idx:3*target_idx+3]
        Mat target_traj = community_traj(Eigen::seqN(0, Nt), Eigen::seqN(3 * target_idx, 3));
        Mat source_traj = community_traj(Eigen::seqN(0, Nt), Eigen::seqN(3 * source_idx, 3));

        Vec const S_r = source_traj.col(0);
        Vec I_r = source_traj.col(1);
        Vec const R_r = source_traj.col(2);
        Vec S_s = target_traj.col(0);
        Vec const I_s = target_traj.col(1);
        Vec R_s = target_traj.col(2);
        Vec p_I = p_Is.col(i);

        Vec denom = (S_s.array() + I_r.array() + R_s.array()).matrix();
        Vec nom = (S_s.array() * I_r.array()).matrix();
        Vec const connection_inf = connection_infs.col(i);
        F_beta_rs_mat.col(i) = p_I.array() * nom.array() / denom.array();
    }
    // not all zero
    assert(F_beta_rs_mat.array().sum() != 0);
    assert(connection_infs.array().sum() != 0);
    return std::make_pair(F_beta_rs_mat, connection_infs);
}

std::tuple<Eigen::MatrixXf, Eigen::MatrixXf> load_N_datasets(const std::string &datapath, uint32_t N, uint32_t offset)
{
    using Mat = Eigen::MatrixXf;
    using Vec = Eigen::VectorXf;
    std::vector<uint32_t> idx(N);
    std::iota(idx.begin(), idx.end(), offset);
    std::vector<std::pair<Mat, Mat>> datasets;
    std::transform(idx.begin(), idx.end(), std::back_inserter(datasets), [&](auto idx)
                   { return load_beta_regression(datapath, idx); });

    std::vector<std::pair<uint32_t, uint32_t>> sizes;
    std::transform(datasets.begin(), datasets.end(), std::back_inserter(sizes), [](auto &dataset)
                   { return std::make_pair(dataset.first.rows(), dataset.first.cols()); });
    uint32_t const tot_rows = std::accumulate(sizes.begin(), sizes.end(), 0, [](auto acc, auto &size)
                                        { return acc + size.first; });
    uint32_t const cols = sizes[0].second;
    Mat connection_infs_tot(tot_rows, cols);
    Mat F_beta_rs_mat(tot_rows, cols);
    // fill mats with datasets
    uint32_t row_offset = 0;
    for (int i = 0; i < datasets.size(); i++)
    {
        auto &dataset = datasets[i];
        auto &size = sizes[i];
        connection_infs_tot(Eigen::seqN(row_offset, size.first), Eigen::all) = dataset.second;
        F_beta_rs_mat(Eigen::seqN(row_offset, size.first), Eigen::all) = dataset.first;
        row_offset += size.first;
    }
    assert(F_beta_rs_mat.array().sum() != 0);
    assert(connection_infs_tot.array().sum() != 0);

    return std::make_tuple(F_beta_rs_mat, connection_infs_tot);
}

std::pair<std::vector<float>, std::vector<float>> beta_regression(const Eigen::MatrixXf &F_beta_rs_mat, const Eigen::MatrixXf &connection_infs, float tau)
{

    uint32_t const N_connections = connection_infs.cols();
    std::vector<float> thetas_LS(N_connections);
    std::vector<float> thetas_QR(N_connections);
    for (int i = 0; i < thetas_QR.size(); i++)
    {
        thetas_LS[i] = connection_infs.col(i).dot(F_beta_rs_mat.col(i)) / F_beta_rs_mat.col(i).dot(F_beta_rs_mat.col(i));
        thetas_QR[i] = quantile_regression(F_beta_rs_mat.col(i), connection_infs.col(i), tau);
    }

    return std::make_pair(thetas_LS, thetas_QR);
}

float alpha_regression(const Eigen::VectorXf &x, const Eigen::VectorXf &y)
{
    return y.dot(x) / x.dot(x);
}
std::tuple<std::vector<float>, std::vector<float>> regression_on_datasets(const std::string &datapath, uint32_t N, float tau, uint32_t offset)
{
    auto [F_beta_rs_mat, connection_infs] = load_N_datasets(datapath, N, offset);
    // float alpha = alpha_regression(x_recovery, y_recovery);
    auto [thetas_LS, thetas_QR] = beta_regression(F_beta_rs_mat, connection_infs, tau);
    return std::make_tuple(thetas_LS, thetas_QR);
}

std::tuple<std::vector<float>, std::vector<float>> regression_on_datasets(const std::vector<std::string> &datapaths, uint32_t N, float tau, uint32_t offset)
{
    using Mat = Eigen::MatrixXf;
    std::vector<std::tuple<Mat, Mat>> datasets(datapaths.size());
    std::transform(datapaths.begin(), datapaths.end(), datasets.begin(), [&](auto &datapath)
                   { return load_N_datasets(datapath, N, offset); });

    Mat F_beta_rs_mat_tot;
    Mat connection_infs_tot;

    uint32_t row_offset = 0;
    // stack datasets
    for (auto & dataset : datasets)
    {
        auto &F_beta_rs_mat = std::get<0>(dataset);
        auto &connection_infs = std::get<1>(dataset);
        uint32_t const N_rows = F_beta_rs_mat.rows();

        F_beta_rs_mat_tot(Eigen::seqN(row_offset, N_rows), Eigen::all) = F_beta_rs_mat;
        connection_infs_tot(Eigen::seqN(row_offset, N_rows), Eigen::all) = connection_infs;
        row_offset += N_rows;
    }

    auto [thetas_LS, thetas_QR] = beta_regression(F_beta_rs_mat_tot, connection_infs_tot, tau);
    return std::make_tuple(thetas_LS, thetas_QR);
}



float quantile_regression(const Eigen::VectorXf& x, const Eigen::VectorXf& y, float tau, float y_tol, float x_tol)
{
    using Mat = Eigen::MatrixXf;
    using Vec = Eigen::VectorXf;
    using namespace operations_research;
    operations_research::MPSolver::OptimizationProblemType problem_type = operations_research::MPSolver::GLOP_LINEAR_PROGRAMMING;
    // if all y is 0, return 0
    if (y.template lpNorm<Eigen::Infinity>() == 0 || x.template lpNorm<Eigen::Infinity>() == 0)
    {
        return 0.0F;
    }

    // std::cout << "x: " << x.transpose() << std::endl;
    // std::cout << "y: " << y.transpose() << std::endl;

    if ((y.template lpNorm<Eigen::Infinity>() < y_tol) || (x.template lpNorm<Eigen::Infinity>() < x_tol))
    {
        float const ynorm = y.lpNorm<Eigen::Infinity>();
        float const xnorm = x.lpNorm<Eigen::Infinity>();
        return std::numeric_limits<float>::infinity();
    }

    uint32_t const N_rows = x.rows();
    static uint32_t count = 0;
    std::string const solver_name = "Quantile_Solver_" + std::to_string(count++);
    std::unique_ptr<MPSolver> solver = std::make_unique<MPSolver>(solver_name, problem_type);
    const float infinity = solver->infinity();
    // theta_neg = solver->MakeNumVar(0.0, infinity, "theta_neg");
    // theta_pos = solver->MakeNumVar(0.0, infinity, "theta_pos");
    operations_research::MPVariable *theta =
        solver->MakeNumVar(-infinity, infinity, "theta");
    std::vector<operations_research::MPVariable *> u_pos;
    std::vector<operations_research::MPVariable *> u_neg;
    std::vector<operations_research::MPConstraint *> g(N_rows);

    operations_research::MPObjective *objective = nullptr;
    objective = solver->MutableObjective();
    objective->SetMinimization();
    solver->MakeNumVarArray(N_rows, 0.0, infinity, "u_pos", &u_pos);
    solver->MakeNumVarArray(N_rows, 0.0, infinity, "u_neg", &u_neg);

    std::for_each(u_pos.begin(), u_pos.end(),
                  [=](auto &u)
                  { objective->SetCoefficient(u, tau); });
    std::for_each(u_neg.begin(), u_neg.end(),
                  [=](auto &u)
                  { objective->SetCoefficient(u, (1 - tau)); });
    // std::generate(g.begin(), g.end(),
    //               [&]() { return solver->MakeRowConstraint(); });

    for (int i = 0; i < N_rows; i++)
    {
        // g[i]->SetCoefficient(theta_pos, x(i));
        // g[i]->SetCoefficient(theta_neg, -x(i));
        g[i] = solver->MakeRowConstraint(y(i), y(i));
        g[i]->SetCoefficient(theta, x(i));
        g[i]->SetCoefficient(u_pos[i], 1);
        g[i]->SetCoefficient(u_neg[i], -1);
        // g[i]->SetBounds(y[i], y[i]);
    }
    //            MX g = xi * (theta_pos - theta_neg) + u_pos - u_neg - dm_y;
    const bool solver_status = solver->Solve() == MPSolver::OPTIMAL;

    float theta_sol = std::numeric_limits<float>::infinity();

    if (solver_status)
    {
        float const f = objective->Value();

        std::vector<float> u_neg_sol(N_rows);
        std::vector<float> u_pos_sol(N_rows);
        for (int i = 0; i < N_rows; i++)
        {
            u_neg_sol[i] = u_neg[i]->solution_value();
            u_pos_sol[i] = u_pos[i]->solution_value();
        }

        theta_sol = theta->solution_value();
    }
    else
    {

        std::cout << "[Quantile_Regressor] Warning: Quantile regression failed"
                  << std::endl;
        std::for_each(g.begin(), g.end(), [](auto &gi)
                      { gi->Clear(); });
    }

    return theta_sol;
}
