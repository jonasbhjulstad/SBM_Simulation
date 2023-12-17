#include <SBM_Graph/Graph.hpp>
#include <SBM_Simulation/Regression/Regression.hpp>
#include <SBM_Simulation/Utils/Math.hpp>

#include <SBM_Database/Graph/Graph_Tables.hpp>
#include <SBM_Database/Regression/Regression_Tables.hpp>
#include <SBM_Database/Simulation/Simulation_Tables.hpp>
#include <SBM_Database/Simulation/Size_Queries.hpp>
#include <fstream>
#include <iostream>
#include <ortools/linear_solver/linear_solver.h>
#include <sstream>
#include <tuple>
#include <utility>
namespace SBM_Regression {
Eigen::MatrixXf openData(const std::string &fileToOpen) {
  using namespace Eigen;
  std::vector<float> matrixEntries;

  std::ifstream matrixDataFile(fileToOpen);

  std::string matrixRowString;

  std::string matrixEntry;

  int matrixRowNumber = 0;

  while (std::getline(matrixDataFile, matrixRowString)) {
    std::stringstream matrixRowStringStream(matrixRowString);

    while (std::getline(matrixRowStringStream, matrixEntry, ',')) {
      matrixEntries.push_back(stod(matrixEntry));
    }
    matrixRowNumber++;
  }

  return Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(
      matrixEntries.data(), matrixRowNumber,
      matrixEntries.size() / matrixRowNumber);
}

std::pair<Eigen::MatrixXf, Eigen::MatrixXf>
load_beta_regression(const std::string &datapath, uint32_t idx, bool truncate) {
  using Mat = Eigen::MatrixXf;
  using Vec = Eigen::VectorXf;
  auto connection_infs_filename = [](uint32_t idx) {
    return std::string("Connection_Infections/connection_infections_" +
                       std::to_string(idx) + ".csv");
  };

  auto community_traj_filename = [](uint32_t idx) {
    return std::string("Trajectories/community_trajectory_" +
                       std::to_string(idx) + ".csv");
  };

  auto infection_events_filename = [](uint32_t idx) {
    return std::string("Connection_Events/connection_events_" +
                       std::to_string(idx) + ".csv");
  };
  auto connection_community_map_filename = [](uint32_t idx) {
    return std::string("ccm.csv");
  };

  auto p_Is_filename = [](uint32_t idx) {
    return std::string("p_Is/p_I_" + std::to_string(idx) + ".csv");
  };

  Mat connection_infs = openData(datapath + connection_infs_filename(idx));
  auto N_connections = connection_infs.cols();

  if ((connection_infs.array() == 0).all()) {
    return std::make_pair(Mat(0, N_connections), Mat(0, N_connections));
  }

  Mat connection_community_map =
      openData(datapath + connection_community_map_filename(idx));
  Mat community_traj = openData(datapath + community_traj_filename(idx));
  assert((connection_infs.array() >= 0).all());
  auto Nt = community_traj.rows() - 1;
  auto N_communities = community_traj.cols() / 3;
  if (truncate) {
    auto t_trunc = 0;
    for (t_trunc = 0; t_trunc < Nt; t_trunc++) {
      if (N_communities == 1) {
        if (community_traj(t_trunc, 1) == 0)
          break;
      } else if (community_traj.row(t_trunc)(Eigen::seqN(1, 3)).sum() == 0) {
        break;
      }
    }
    connection_infs.conservativeResize(t_trunc, Eigen::NoChange);
    community_traj.conservativeResize(t_trunc, Eigen::NoChange);
    Nt = t_trunc;
  }
  assert((connection_infs.array() <= 1000).all());
  assert((community_traj.array() <= 1000).all());
  Mat p_Is = openData(datapath + p_Is_filename(idx));

  auto compute_beta_rs_col = [&](auto from_idx, auto to_idx, const Vec &p_I) {
    Mat target_traj =
        community_traj(Eigen::seqN(0, Nt), Eigen::seqN(3 * to_idx, 3));
    Mat source_traj =
        community_traj(Eigen::seqN(0, Nt), Eigen::seqN(3 * from_idx, 3));
    Vec p_I_trunc = p_I(Eigen::seqN(0, Nt));
    Vec const S_r = source_traj.col(0);
    Vec I_r = source_traj.col(1);
    Vec const R_r = source_traj.col(2);
    Vec S_s = target_traj.col(0);
    Vec const I_s = target_traj.col(1);
    Vec R_s = target_traj.col(2);

    Vec denom = (S_s.array() + I_r.array() + R_s.array()).matrix();
    Vec nom = (S_s.array() * I_r.array()).matrix();
    return Vec(p_I_trunc.array() * nom.array() / denom.array());
  };

  Mat F_beta_rs_mat(Nt, N_connections);
  for (int i = 0; i < connection_community_map.rows(); i++) {
    F_beta_rs_mat.col(i) =
        compute_beta_rs_col(connection_community_map(i, 0),
                            connection_community_map(i, 1), p_Is.col(i));
  }
  assert(F_beta_rs_mat.array().sum() != 0);
  assert(connection_infs.array().sum() != 0);
  return std::make_pair(F_beta_rs_mat, connection_infs);
}

std::pair<Eigen::MatrixXf, Eigen::MatrixXf>
load_beta_regression(uint32_t p_out, uint32_t graph, uint32_t sim_id,
                     const QString &control_type, bool truncate) {
  using Mat = Eigen::MatrixXf;
  using Vec = Eigen::VectorXf;
  auto connection_community_map_filename = [](uint32_t idx) {
    return std::string("ccm.csv");
  };

  auto p_Is_filename = [](uint32_t idx) {
    return std::string("p_Is/p_I_" + std::to_string(idx) + ".csv");
  };

  auto N_connections = SBM_Database::get_N_connections(
      "infection_events_excitation", p_out, graph);
  
  auto connection_infs_df = SBM_Database::connection_read<uint32_t>(
      "infection_events_excitation", p_out, graph, sim_id, "value",
      control_type, "");
  Mat connection_infs = Dataframe::to_eigen_matrix<uint32_t>(connection_infs_df);
  auto ccm = SBM_Database::ccm_read(p_out, graph);
  auto community_traj_df = SBM_Database::community_state_read(
      p_out, graph, sim_id, control_type, "");
  Mat community_traj = Dataframe::to_eigen_matrix<uint32_t>(community_traj_df);
  if ((connection_infs.array() == 0).all()) {
    return std::make_pair(Mat(0, N_connections), Mat(0, N_connections));
  }
  assert((connection_infs.array() >= 0).all());
  auto Nt = community_traj.rows() - 1;
  auto N_communities = community_traj.cols() / 3;
  if (truncate) {
    auto t_trunc = 0;
    for (t_trunc = 0; t_trunc < Nt; t_trunc++) {
      if (N_communities == 1) {
        if (community_traj(t_trunc, 1) == 0)
          break;
      } else if (community_traj.row(t_trunc)(Eigen::seqN(1, 3)).sum() == 0) {
        break;
      }
    }
    connection_infs.conservativeResize(t_trunc, Eigen::NoChange);
    community_traj.conservativeResize(t_trunc, Eigen::NoChange);
    Nt = t_trunc;
  }
  assert((connection_infs.array() <= 1000).all());
  assert((community_traj.array() <= 1000).all());
  auto p_I_df = SBM_Database::connection_read<float>(
      "p_Is_excitation", p_out, graph, "value", control_type, "");
  Mat p_Is = Dataframe::to_eigen_matrix<float>(p_I_df);

  auto compute_beta_rs_col = [&](auto from_idx, auto to_idx, const Vec &p_I) {
    Mat target_traj =
        community_traj(Eigen::seqN(0, Nt), Eigen::seqN(3 * to_idx, 3));
    Mat source_traj =
        community_traj(Eigen::seqN(0, Nt), Eigen::seqN(3 * from_idx, 3));
    Vec p_I_trunc = p_I(Eigen::seqN(0, Nt));
    Vec const S_r = source_traj.col(0);
    Vec I_r = source_traj.col(1);
    Vec const R_r = source_traj.col(2);
    Vec S_s = target_traj.col(0);
    Vec const I_s = target_traj.col(1);
    Vec R_s = target_traj.col(2);

    Vec denom = (S_s.array() + I_r.array() + R_s.array()).matrix();
    Vec nom = (S_s.array() * I_r.array()).matrix();
    return Vec(p_I_trunc.array() * nom.array() / denom.array());
  };

  Mat F_beta_rs_mat(Nt, N_connections);
  for (int i = 0; i < ccm.size(); i++) {
    F_beta_rs_mat.col(i) =
        compute_beta_rs_col(ccm[i].from, ccm[i].to, p_Is.col(i));
  }
  assert(F_beta_rs_mat.array().sum() != 0);
  assert(connection_infs.array().sum() != 0);
  return std::make_pair(F_beta_rs_mat, connection_infs);
}

std::tuple<Eigen::MatrixXf, Eigen::MatrixXf>
load_N_datasets(const std::string &datapath, uint32_t N, uint32_t offset) {
  using Mat = Eigen::MatrixXf;
  using Vec = Eigen::VectorXf;
  std::vector<uint32_t> idx(N);
  std::iota(idx.begin(), idx.end(), offset);
  std::vector<std::pair<Mat, Mat>> datasets;
  std::transform(idx.begin(), idx.end(), std::back_inserter(datasets),
                 [&](auto idx) { return load_beta_regression(datapath, idx); });

  std::vector<SBM_Graph::Edge_t> sizes;
  std::transform(datasets.begin(), datasets.end(), std::back_inserter(sizes),
                 [](auto &dataset) {
                   return SBM_Graph::Edge_t(dataset.first.rows(),
                                            dataset.second.cols());
                 });
  uint32_t const tot_rows =
      std::accumulate(sizes.begin(), sizes.end(), 0,
                      [](auto acc, auto &size) { return acc + size.from; });
  uint32_t const cols = sizes[0].to;
  Mat connection_infs_tot(tot_rows, cols);
  Mat F_beta_rs_mat(tot_rows, cols);
  uint32_t row_offset = 0;
  for (int i = 0; i < datasets.size(); i++) {
    auto &dataset = datasets[i];
    auto &size = sizes[i];
    connection_infs_tot(Eigen::seqN(row_offset, size.from), Eigen::all) =
        dataset.second;
    F_beta_rs_mat(Eigen::seqN(row_offset, size.from), Eigen::all) =
        dataset.first;
    row_offset += size.from;
  }
  assert(F_beta_rs_mat.array().sum() != 0);
  assert(connection_infs_tot.array().sum() != 0);

  return std::make_tuple(F_beta_rs_mat, connection_infs_tot);
}

std::tuple<Eigen::MatrixXf, Eigen::MatrixXf>
load_database_datasets(uint32_t p_out, uint32_t graph,
                       const QString &control_type) {
  using Mat = Eigen::MatrixXf;
  using Vec = Eigen::VectorXf;
  auto N_sims = SBM_Database::get_N_sims("community_state_excitation");
  std::vector<std::pair<Mat, Mat>> datasets(N_sims);
  std::vector<uint32_t> idx(N_sims);
  std::iota(idx.begin(), idx.end(), 0);
  std::transform(idx.begin(), idx.end(), std::back_inserter(datasets),
                 [&](auto idx) {
                   return load_beta_regression(p_out, graph, idx, control_type);
                 });

  std::vector<SBM_Graph::Edge_t> sizes;
  std::transform(datasets.begin(), datasets.end(), std::back_inserter(sizes),
                 [](auto &dataset) {
                   return SBM_Graph::Edge_t(dataset.first.rows(),
                                            dataset.second.cols());
                 });
  uint32_t const tot_rows =
      std::accumulate(sizes.begin(), sizes.end(), 0,
                      [](auto acc, auto &size) { return acc + size.from; });
  uint32_t const cols = sizes[0].to;
  Mat connection_infs_tot(tot_rows, cols);
  Mat F_beta_rs_mat(tot_rows, cols);
  uint32_t row_offset = 0;
  for (int i = 0; i < datasets.size(); i++) {
    auto &dataset = datasets[i];
    auto &size = sizes[i];
    connection_infs_tot(Eigen::seqN(row_offset, size.from), Eigen::all) =
        dataset.second;
    F_beta_rs_mat(Eigen::seqN(row_offset, size.from), Eigen::all) =
        dataset.first;
    row_offset += size.from;
  }
  assert(F_beta_rs_mat.array().sum() != 0);
  assert(connection_infs_tot.array().sum() != 0);

  return std::make_tuple(F_beta_rs_mat, connection_infs_tot);
}

auto compute_MSE(const Eigen::VectorXf &x, const Eigen::VectorXf &y) {
  return (x - y).squaredNorm() / x.rows();
}

auto compute_MAE(const Eigen::VectorXf &x, const Eigen::VectorXf &y) {
  return (x - y).cwiseAbs().sum() / x.rows();
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
           std::vector<float>>
beta_regression(const Eigen::MatrixXf &F_beta_rs_mat,
                const Eigen::MatrixXf &connection_infs, float tau) {

  uint32_t const N_connections = connection_infs.cols();
  std::vector<float> thetas_LS(N_connections);
  std::vector<float> thetas_QR(N_connections);
  std::vector<float> MSE(N_connections);
  std::vector<float> MAE(N_connections);
  for (int i = 0; i < thetas_QR.size(); i++) {
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

float alpha_regression(const Eigen::VectorXf &x, const Eigen::VectorXf &y) {
  return y.dot(x) / x.dot(x);
}
Eigen::MatrixXf filter_columns(const Eigen::MatrixXf &mat,
                               const std::vector<uint32_t> &indices) {
  Eigen::MatrixXf filtered_mat(mat.rows(), indices.size());
  for (int i = 0; i < indices.size(); i++) {
    filtered_mat.col(i) = mat.col(indices[i]);
  }
  return filtered_mat;
}

auto filter_zero_columns(const Eigen::MatrixXf &mat) {
  std::vector<uint32_t> non_zero_cols;
  for (int i = 0; i < mat.cols(); i++) {
    if (mat.col(i).array().sum() != 0) {
      non_zero_cols.push_back(i);
    }
  }
  return std::make_tuple(filter_columns(mat, non_zero_cols), non_zero_cols);
}
template <typename T>
std::vector<T> project(const std::vector<T> &data,
                       const std::vector<uint32_t> &indices, auto size) {
  std::vector<T> projected_data(size, T{});
  for (int i = 0; i < size; i++) {
    if (std::find(indices.begin(), indices.end(), i) != indices.end()) {
      projected_data[i] = data[i];
    }
  }
  return projected_data;
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
           std::vector<float>>
regression_on_datasets(const std::string &datapath, uint32_t N, float tau,
                       uint32_t offset) {
  auto [F_beta_rs_mat_raw, connection_infs_raw] =
      load_N_datasets(datapath, N, offset);
  auto N_connections = connection_infs_raw.cols();
  auto [connection_infs, non_zero_cols] =
      filter_zero_columns(connection_infs_raw);
  auto F_beta_rs_mat = filter_columns(F_beta_rs_mat_raw, non_zero_cols);

  auto [thetas_LS, thetas_QR, MSE, MAE] =
      beta_regression(F_beta_rs_mat, connection_infs, tau);
  auto LS_proj = project(thetas_LS, non_zero_cols, N_connections);
  auto QR_proj = project(thetas_QR, non_zero_cols, N_connections);

  auto MSE_proj = project(MSE, non_zero_cols, N_connections);
  auto MAE_proj = project(MAE, non_zero_cols, N_connections);

  return std::make_tuple(LS_proj, QR_proj, MSE_proj, MAE_proj);
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
           std::vector<float>>
regression_on_datasets(const std::vector<std::string> &datapaths, uint32_t N,
                       float tau, uint32_t offset) {
  using Mat = Eigen::MatrixXf;
  std::vector<std::tuple<Mat, Mat>> datasets(datapaths.size());
  std::transform(
      datapaths.begin(), datapaths.end(), datasets.begin(),
      [&](auto &datapath) { return load_N_datasets(datapath, N, offset); });

  Mat F_beta_rs_mat_tot;
  Mat connection_infs_tot;

  uint32_t row_offset = 0;
  for (auto &dataset : datasets) {
    auto &F_beta_rs_mat = std::get<0>(dataset);
    auto &connection_infs = std::get<1>(dataset);
    uint32_t const N_rows = F_beta_rs_mat.rows();

    F_beta_rs_mat_tot(Eigen::seqN(row_offset, N_rows), Eigen::all) =
        F_beta_rs_mat;
    connection_infs_tot(Eigen::seqN(row_offset, N_rows), Eigen::all) =
        connection_infs;
    row_offset += N_rows;
  }

  auto [thetas_LS, thetas_QR, MSE, MAE] =
      beta_regression(F_beta_rs_mat_tot, connection_infs_tot, tau);
  return std::make_tuple(thetas_LS, thetas_QR, MSE, MAE);
}

void regression_on_database(float tau, uint32_t tau_id, uint32_t p_out,
                            uint32_t graph, const QString &control_type) {
  using Mat = Eigen::MatrixXf;
  auto N_sims = SBM_Database::get_N_sims("community_state_excitation");
  std::vector<std::tuple<Mat, Mat>> datasets(N_sims);
  std::transform(datasets.begin(), datasets.end(), datasets.begin(),
                 [&](auto &dataset) {
                   return load_database_datasets(p_out, graph, control_type);
                 });

  Mat F_beta_rs_mat_tot;
  Mat connection_infs_tot;

  uint32_t row_offset = 0;
  for (auto &dataset : datasets) {
    auto &F_beta_rs_mat = std::get<0>(dataset);
    auto &connection_infs = std::get<1>(dataset);
    uint32_t const N_rows = F_beta_rs_mat.rows();

    F_beta_rs_mat_tot(Eigen::seqN(row_offset, N_rows), Eigen::all) =
        F_beta_rs_mat;
    connection_infs_tot(Eigen::seqN(row_offset, N_rows), Eigen::all) =
        connection_infs;
    row_offset += N_rows;
  }
  SBM_Database::drop_regression_tables();

  auto [thetas_LS, thetas_QR, MSE, MAE] =
      beta_regression(F_beta_rs_mat_tot, connection_infs_tot, tau);

  SBM_Database::regression_parameters_upsert(
      p_out, graph, tau, tau_id, control_type, "LS", thetas_LS, MSE);
  SBM_Database::regression_parameters_upsert(
      p_out, graph, tau, tau_id, control_type, "QR", thetas_QR, MAE);
}
void regression_on_database(float tau, uint32_t tau_id,
                            const QString &control_type) {
  using Mat = Eigen::MatrixXf;
  auto Np = SBM_Database::get_N_p_out();
  auto Ng = SBM_Database::get_N_graphs();
  for (int i = 0; i < Np; i++) {
    for (int j = 0; j < Ng; j++) {
      regression_on_database(tau, tau_id, i, j, control_type);
    }
  }
}

float quantile_regression(const Eigen::VectorXf &x, const Eigen::VectorXf &y,
                          float tau, float y_tol, float x_tol) {
  using Mat = Eigen::MatrixXf;
  using Vec = Eigen::VectorXf;
  using namespace operations_research;
  operations_research::MPSolver::OptimizationProblemType problem_type =
      operations_research::MPSolver::GLOP_LINEAR_PROGRAMMING;
  if (y.template lpNorm<Eigen::Infinity>() == 0 ||
      x.template lpNorm<Eigen::Infinity>() == 0) {
    return 0.0F;
  }

  if ((y.template lpNorm<Eigen::Infinity>() < y_tol) ||
      (x.template lpNorm<Eigen::Infinity>() < x_tol)) {
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
                [=](auto &u) { objective->SetCoefficient(u, tau); });
  std::for_each(u_neg.begin(), u_neg.end(),
                [=](auto &u) { objective->SetCoefficient(u, (1 - tau)); });

  for (int i = 0; i < N_rows; i++) {
    g[i] = solver->MakeRowConstraint(y(i), y(i));
    g[i]->SetCoefficient(theta, x(i));
    g[i]->SetCoefficient(u_pos[i], 1);
    g[i]->SetCoefficient(u_neg[i], -1);
  }
  const bool solver_status = solver->Solve() == MPSolver::OPTIMAL;

  float theta_sol = std::numeric_limits<float>::infinity();

  if (solver_status) {
    float const f = objective->Value();

    std::vector<float> u_neg_sol(N_rows);
    std::vector<float> u_pos_sol(N_rows);
    for (int i = 0; i < N_rows; i++) {
      u_neg_sol[i] = u_neg[i]->solution_value();
      u_pos_sol[i] = u_pos[i]->solution_value();
    }

    theta_sol = theta->solution_value();
  } else {

    std::cout << "[Quantile_Regressor] Warning: Quantile regression failed"
              << std::endl;
    std::for_each(g.begin(), g.end(), [](auto &gi) { gi->Clear(); });
  }

  return theta_sol;
}
} // namespace SBM_Regression