#ifndef FROLS_POLYNOMIAL_HPP
#define FROLS_POLYNOMIAL_HPP
#include "Feature_Model.hpp"
#include <Regressor.hpp>
#include <Typedefs.hpp>
namespace FROLS::Features {

struct Polynomial_Model : public Feature_Model{
  const size_t N_output_features, d_max;
  const size_t Nx;
  const size_t Nu;

  Polynomial_Model(size_t Nx, size_t Nu, size_t N_output_features, size_t d_max)
      : N_output_features(N_output_features), d_max(d_max), Nx(Nx), Nu(Nu) {}


  // double transform(crVec &x_raw, size_t target_index) const;
  Vec _transform(crMat &X_raw, size_t target_index) const;
  Mat _transform(crMat &X_raw) const;
  const std::vector<std::vector<Feature>> get_features() const;

  void write_csv(const std::string &) const;
  void feature_summary() const;
  const std::string feature_name(size_t target_index, bool indent = true) const;
  const std::vector<std::string> feature_names() const;
  const std::string model_equation(size_t idx)const ;
  const std::string model_equations() const;
};

} // namespace FROLS::Features::Polynomial
#endif