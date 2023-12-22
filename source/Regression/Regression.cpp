#include <SBM_Graph/Graph.hpp>
#include <SBM_Simulation/Regression/Regression.hpp>
#include <SBM_Simulation/Utils/Math.hpp>
#include <SBM_Simulation/Regression/Filesystem.hpp>
#include <SBM_Database/Graph/Graph_Tables.hpp>
#include <SBM_Database/Regression/Regression_Tables.hpp>
#include <SBM_Database/Simulation/Size_Queries.hpp>
#include <fstream>
#include <iostream>
#include <orm/db.hpp>
#include <ortools/linear_solver/linear_solver.h>
#include <sstream>
#include <tuple>
#include <utility>
namespace SBM_Regression
{

  // std::tuple<Mat, Mat>
  // load_database_datasets(uint32_t p_out, uint32_t graph,
  //                        const QString &control_type)
  // {

  //   auto N_sims = SBM_Database::>get_N_sims("community_state_excitation");
  //   std::vector<std::pair<Mat, Mat>> datasets(N_sims);
  //   std::vector<uint32_t> idx(N_sims);
  //   std::iota(idx.begin(), idx.end(), 0);
  //   std::transform(idx.begin(), idx.end(), std::back_inserter(datasets),
  //                  [&](auto idx)
  //                  {
  //                    return load_beta_regression(p_out, graph, idx, control_type);
  //                  });

  //   std::vector<SBM_Graph::Edge_t> sizes;
  //   std::transform(datasets.begin(), datasets.end(), std::back_inserter(sizes),
  //                  [](auto &dataset)
  //                  {
  //                    return SBM_Graph::Edge_t(dataset.first.rows(),
  //                                             dataset.second.cols());
  //                  });
  //   uint32_t const tot_rows =
  //       std::accumulate(sizes.begin(), sizes.end(), 0,
  //                       [](auto acc, auto &size)
  //                       { return acc + size.from; });
  //   uint32_t const cols = sizes[0].to;
  //   Mat connection_infs_tot(tot_rows, cols);
  //   Mat F_beta_rs_mat(tot_rows, cols);
  //   uint32_t row_offset = 0;
  //   for (int i = 0; i < datasets.size(); i++)
  //   {
  //     auto &dataset = datasets[i];
  //     auto &size = sizes[i];
  //     connection_infs_tot(Eigen::seqN(row_offset, size.from), Eigen::all) =
  //         dataset.second;
  //     F_beta_rs_mat(Eigen::seqN(row_offset, size.from), Eigen::all) =
  //         dataset.first;
  //     row_offset += size.from;
  //   }
  //   assert(F_beta_rs_mat.array().sum() != 0);
  //   assert(connection_infs_tot.array().sum() != 0);

  //   return std::make_tuple(F_beta_rs_mat, connection_infs_tot);
  // }

  auto compute_MSE(const Vec &x, const Vec &y)
  {
    return (x - y).squaredNorm() / x.rows();
  }

  auto compute_MAE(const Vec &x, const Vec &y)
  {
    return (x - y).cwiseAbs().sum() / x.rows();
  }

  std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
             std::vector<float>>
  beta_regression(const Mat &F_beta_rs_mat,
                  const Mat &connection_infs, float tau)
  {

    uint32_t const N_connections = connection_infs.cols();
    std::vector<float> thetas_LS(N_connections);
    std::vector<float> thetas_QR(N_connections);
    std::vector<float> MSE(N_connections);
    std::vector<float> MAE(N_connections);
    for (int i = 0; i < thetas_QR.size(); i++)
    {
      thetas_LS[i] = connection_infs.col(i).dot(F_beta_rs_mat.col(i)) /
                     F_beta_rs_mat.col(i).dot(F_beta_rs_mat.col(i));
      thetas_QR[i] =
          quantile_regression(F_beta_rs_mat.col(i), connection_infs.col(i), tau);

      MSE[i] = compute_MSE(F_beta_rs_mat.col(i) * thetas_LS[i],
                           connection_infs.col(i));
      MAE[i] = compute_MAE(F_beta_rs_mat.col(i) * thetas_QR[i],
                           connection_infs.col(i));
    }

    return std::make_tuple(thetas_LS, thetas_QR, MSE, MAE);
  }

  float alpha_regression(const Vec &x, const Vec &y)
  {
    return y.dot(x) / x.dot(x);
  }

  std::vector<uint32_t> get_nonzero_rows(const Mat &X_data)
  {
    std::vector<uint32_t> result;
    for (int i = 0; i < X_data.rows(); i++)
    {
      if (X_data.row(i).array().sum() > 0)
      {
        result.push_back(i);
      }
    }
    return result;
  }

  void filter_data(Mat &X_data, Mat &Y_data, Mat &U_data, Mat& connection_infs)
  {
    auto indices = get_nonzero_rows(X_data);
    X_data = X_data(indices, Eigen::all);
    Y_data = Y_data(indices, Eigen::all);
    U_data = U_data(indices, Eigen::all);
    connection_infs = connection_infs(indices, Eigen::all);

  }


  


  void regression_on_database(float tau, uint32_t tau_id, uint32_t p_out,
                              uint32_t graph, const char* control_type)
  {
    Mat X_data = read_community_X_data(p_out, graph, control_type);
    Mat Y_data = read_community_Y_data(p_out, graph, control_type);
    Mat U_data = read_community_U_data(p_out, graph, control_type);
    Mat connection_infs = read_connection_infections(p_out, graph, control_type);
    filter_data(X_data, Y_data, U_data, connection_infs);

    auto [thetas_LS, thetas_QR, MSE, MAE] =
        beta_regression(X_data, connection_infs, tau);

    auto alpha = alpha_regression(X_data, Y_data);
  
    SBM_Database::regression_param_insert(p_out, graph, thetas_LS, alpha, MSE, tau, tau_id, control_type, "LS");
    SBM_Database::regression_param_insert(p_out, graph, thetas_QR, alpha, MAE, tau, tau_id, control_type, "QR");


  }

  void regression_on_database(float tau, uint32_t tau_id, const char* control_type)
  {
      auto Np = SBM_Database::get_N_p_out("infection_events");
      auto Ng = SBM_Database::get_N_graphs("infection_events");
      for (int p_out_id = 0; p_out_id < Np; p_out_id ++)
      {
        for(int graph_id = 0; graph_id < Ng; graph_id++)
        {
          regression_on_database(tau, tau_id, p_out_id, graph_id, control_type);
        }
      }


  }



  float quantile_regression(const Vec &x, const Vec &y,
                            float tau, float y_tol, float x_tol)
  {

    using namespace operations_research;
    operations_research::MPSolver::OptimizationProblemType problem_type =
        operations_research::MPSolver::GLOP_LINEAR_PROGRAMMING;
    if (y.template lpNorm<Eigen::Infinity>() == 0 ||
        x.template lpNorm<Eigen::Infinity>() == 0)
    {
      return 0.0F;
    }

    if ((y.template lpNorm<Eigen::Infinity>() < y_tol) ||
        (x.template lpNorm<Eigen::Infinity>() < x_tol))
    {
      float const ynorm = y.lpNorm<Eigen::Infinity>();
      float const xnorm = x.lpNorm<Eigen::Infinity>();
      return std::numeric_limits<float>::infinity();
    }

    uint32_t const N_rows = x.rows();
    static uint32_t count = 0;
    std::string const solver_name = "Quantile_Solver_" + std::to_string(count++);
    std::unique_ptr<MPSolver> solver =
        std::make_unique<MPSolver>(solver_name, problem_type);
    const float infinity = solver->infinity();
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

    for (int i = 0; i < N_rows; i++)
    {
      g[i] = solver->MakeRowConstraint(y(i), y(i));
      g[i]->SetCoefficient(theta, x(i));
      g[i]->SetCoefficient(u_pos[i], 1);
      g[i]->SetCoefficient(u_neg[i], -1);
    }
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
} // namespace SBM_Regression