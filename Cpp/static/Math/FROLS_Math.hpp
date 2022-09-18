#ifndef FROLS_FROLS_MATH_HPP
#define FROLS_FROLS_MATH_HPP
#include <Typedefs.hpp>

namespace FROLS {
double cov_normalize(const Vec &a, const Vec &b);

size_t n_choose_k(size_t n, size_t k);

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

template <typename T>
inline std::vector<std::pair<T, T>> zip(const std::vector<T>& a, const std::vector<T>& b)
{
    std::vector<std::pair<T, T>> res(a.size());
    for (int i = 0; i < a.size(); i++)
    {
        res[i] = std::make_pair(a[i], b[i]);
    }
    return res;
}

    template <typename T>
    inline std::pair<std::vector<T>, std::vector<T>> unzip(const std::vector<std::pair<T, T>>& a)
    {
        std::pair<std::vector<T>, std::vector<T>> res;
        res.first.reserve(a.size());
        res.second.reserve(a.size());
        for (int i = 0; i < a.size(); i++)
        {
            res.first.push_back(a[i].first);
            res.second.push_back(a[i].second);
        }
        return res;
    }

    template <typename T>
    inline std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>>& vec)
    {
        std::vector<std::vector<T>> res(vec[0].size());
        for (int i = 0; i < vec.size(); i++)
        {
            for (int j = 0; j < vec[i].size(); j++)
            {
                res[j].push_back(vec[i][j]);
            }
        }
        return res;
    }


} // namespace FROLS

#endif