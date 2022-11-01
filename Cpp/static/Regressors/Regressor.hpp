#ifndef FROLS_REGRESSOR_HPP
#define FROLS_REGRESSOR_HPP

#include <FROLS_Math.hpp>
#include <FROLS_Path_Config.hpp>
#include <Feature_Model.hpp>
#include <memory>
#include <vector>

namespace FROLS::Regression {
struct Regressor_Param {
  float tol = 1e-2;
  float theta_tol = 1e-4;
  uint32_t N_terms_max = 4;
};
struct Regression_Data {
  Regression_Data(uint32_t size): X(size), U(size), Y(size) {}
  void resize(uint32_t size) {
    X.resize(size);
    U.resize(size);
    Y.resize(size);
  }
  std::vector<Mat> X;
  std::vector<Mat> U;
  std::vector<Mat> Y;
};

struct Regressor {
  const float tol, theta_tol;
  const uint32_t N_terms_max;

  Regressor(const Regressor_Param &);

  // std::vector<Feature> fit(const std::vector<Mat> &X,
  //                          const std::vector<Vec> &y);
  // declare transform_fit functions with vector ys

  // std::vector<std::vector<Feature>>
  // transform_fit(const Regression_Data &rd, Features::Feature_Model &model);
  std::vector<std::vector<Feature>> transform_fit(const std::vector<Mat>& X_raw, const std::vector<Mat>& U_raw, const std::vector<Mat>& Y_list,
                                Features::Feature_Model &model);


std::vector<std::vector<Feature>> transform_fit(const Regression_Data &rd,
                         Features::Feature_Model &model);
  // std::vector<Feature> transform_fit(const std::vector<std::string> &filenames,
  //                                    const std::vector<std::string> &colnames_x,
  //                                    const std::vector<std::string> &colnames_u,
  //                                    const std::string &colname_y,
  //                                    Features::Feature_Model &model);
  // std::vector<Feature> transform_fit(const std::vector<Mat> &X_raw,
  //                                               const Mat &U_raw, const Vec &y,
  //                                               Features::Feature_Model &model);
  virtual bool objective_condition(float, float) const = 0;
  virtual void theta_solve(const Mat &A, const Vec &g,
                           std::vector<Feature> &features) const = 0;
  virtual ~Regressor() = default;

protected:
  Vec predict(const Mat &X, const std::vector<Feature> &features) const;

  std::vector<uint32_t>
  unused_feature_indices(const std::vector<Feature> &features,
                         uint32_t N_features) const;

private:
  //   std::vector<Feature> single_fit(const Mat &X, const Vec &y) const;

  std::vector<Feature> single_fit(const std::vector<Mat> &,
                                  const std::vector<Vec> &) const;

  virtual std::vector<std::vector<Feature>>
  candidate_regression(const std::vector<Mat> &X_list,
                       const std::vector<Mat> &Q_list_global,
                       const std::vector<Vec> &y_list,
                       const std::vector<Feature> &used_features) const = 0;
  virtual bool tolerance_check(const std::vector<Mat> &X_list,
                               const std::vector<Vec> &y_list,
                               const std::vector<Feature> &best_features,
                               uint32_t) const = 0;
  Feature feature_selection_criteria(
      const std::vector<std::vector<Feature>> &candidate_features) const;

  Feature best_feature_select(const std::vector<Mat> &X_list,
                              const std::vector<Mat> &Q_global,
                              const std::vector<Vec> &y_list,
                              const std::vector<Feature> &used_features) const;

  static int regressor_count;
};
} // namespace FROLS::Regression

#endif