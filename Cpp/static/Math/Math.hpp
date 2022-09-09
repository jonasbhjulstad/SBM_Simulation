#ifndef FROLS_MATH_HPP
#define FROLS_MATH_HPP
#include <Typedefs.hpp>

namespace FROLS {
double cov_normalize(const Vec &a, const Vec &b);

Vec vec_orthogonalize(const Vec &v, const Mat &Q);
std::vector<double> linspace(double min, double max, int N);

std::vector<double> arange(double min, double max, double step);

std::vector<size_t> range(size_t start, size_t end);

template <typename T>
inline Vec monomial_powers(const Mat& X, const T& powers)
{
  Vec result(X.rows());
  result.setConstant(1);
  for (int i = 0; i < powers.size(); i++)
  {
    result = result.array()*X.col(i).array().pow(powers[i]);
  }
  return result;
}

template <typename T>
inline double monomial_power(const Vec& x, const T& powers)
{
  double result = 1;
  for (int i = 0; i < powers.size(); i++)
  {
    result *= pow(x(i), powers[i]);
  }
  return result;
}

} // namespace FROLS

#endif