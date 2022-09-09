#include "Quantile_Regressor.hpp"
#include <Math.hpp>
#include <casadi/casadi.hpp>
#include <limits>

namespace FROLS::Regression {

Feature Quantile_Regressor::feature_select(const Mat &X, const Vec &y,
                                                    const iVec &used_features) const {
  // tau - Quantile
  size_t N_features = X.cols();
  size_t N_rows = X.rows();
  using namespace casadi;
  MX theta_pos = MX::sym("theta_pos");
  MX theta_neg = MX::sym("theta_neg");
  MX u_pos = MX::sym("u_pos", N_rows);
  MX u_neg = MX::sym("u_neg", N_rows);

  DM dm_y = DM(std::vector<double>(y.data(), y.data() + N_rows));
  Feature best_feature;
  best_feature.f_ERR = std::numeric_limits<double>::max();
  std::map<std::string, DM> res;
  std::map<std::string, DM> arg;
  arg["lbx"] = 0;
  arg["ubx"] = inf;
  arg["lbg"] = 0;
  arg["ubg"] = 0;
  arg["x0"] = 1;
  Dict opts;
  opts["ipopt.print_level"] = 0;
  opts["ipopt.linear_solver"] = "ma27";
  // opts["ipopt.output_file"] = (std::string(DATA_DIR) +
  // "/ipopt.out").c_str();
  opts["ipopt.file_print_level"] = 6;
  opts["print_time"] = 0;
  opts["ipopt.sb"] = "yes";
  for (int i = 0; i < N_features; i++) {
    // If the feature is already used, skip it
    if (!used_features.cwiseEqual(i).any()) {
      DM xi =
          DM(std::vector<double>(X.col(i).data(), X.col(i).data() + N_rows));
      // print xi
      // std::cout << "xi: " << xi << std::endl;
      MX g = xi * (theta_pos - theta_neg) + u_pos - u_neg - dm_y;
      MX f_obj = (tau * sum1(u_pos) + (1 - tau) * sum1(u_neg))/N_rows;
      MXDict nlp = {{"x", vertcat(theta_pos, theta_neg, u_pos, u_neg)},
                    {"f", f_obj},
                    {"g", g}};
      Function solver = nlpsol("solver", "ipopt", nlp, opts);
      res = solver(arg);
      // get element 0 from dm
      auto success = solver.stats()["success"];
      if (success) {
        double f = res["f"](0).scalar();
        if (f < best_feature.f_ERR) {
          double theta_val = res["x"](0).scalar() - res["x"](1).scalar();
          best_feature.f_ERR = f;
          best_feature.g = theta_val;
          best_feature.index = i;
        }
      }
    }
  }

  return best_feature;
}

bool Quantile_Regressor::tolerance_check(
    const Mat &Q, const Vec &y, const std::vector<Feature> &best_features) const{
  Vec y_pred(y.rows());
  y_pred.setZero();
  for (const auto &feature : best_features) {
    y_pred += Q.col(feature.index) * feature.g;
  }
  Vec diff = y - y_pred;
  size_t N_samples = y.rows();
  double err = (diff.array() > 0).select(tau * diff, -(1 - tau) * diff).sum()/N_samples;
  return err < tol;
}

} // namespace FROLS::Features
