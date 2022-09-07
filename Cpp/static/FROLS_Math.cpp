#include "FROLS_Math.hpp"

namespace FROLS {
double cov_normalize(const Vec &a, const Vec &b) {
  return (a.transpose() * b).value() / (a.transpose() * a).value();
}

Vec vec_orthogonalize(const Vec &v, const Mat &Q) {
  Vec cov_remainder = v;
  for (int i = 0; i < Q.cols(); i++) {
    // if (Q.col(i).isApproxToConstant(0))
      // continue;
    cov_remainder -= cov_normalize(Q.col(i), cov_remainder) * Q.col(i);
  }
  return cov_remainder;
}
std::vector<double> linspace(double min, double max, int N) {
  std::vector<double> res(N);
  for (int i = 0; i < N; i++) {
    res[i] = min + (max - min) * i / (N - 1);
  }
  return res;
}

std::vector<double> arange(double min, double max, double step) {
  double s = min;
  std::vector<double> res;
  while (s <= max) {
    res.push_back(s);
    s += step;
  }
  return res;
}

std::vector<size_t> range(size_t start, size_t end) {
  std::vector<size_t> res(end - start);
  for (size_t i = start; i < end; i++) {
    res[i - start] = i;
  }
  return res;
}

} // namespace FROLS
