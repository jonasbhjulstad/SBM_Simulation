#ifndef FROLS_MATH_HPP
#define FROLS_MATH_HPP
#include <FROLS_Typedefs.hpp>

namespace FROLS {
double cov_normalize(const Vec &a, const Vec &b);

Vec vec_orthogonalize(const Vec &v, const Mat &Q);
std::vector<double> linspace(double min, double max, int N);

std::vector<double> arange(double min, double max, double step);

std::vector<size_t> range(size_t start, size_t end);

} // namespace FROLS

#endif