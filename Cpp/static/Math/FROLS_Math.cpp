#include "FROLS_Math.hpp"

namespace FROLS {
    double cov_normalize(const Vec &a, const Vec &b) {
        return ((a.transpose() * a).isZero()) ? 0 : (a.transpose() * b).value() / (a.transpose() * a);
    }

    uint16_t n_choose_k(uint16_t n, uint16_t k) {
        if (k > n) return 0;
        if (k * 2 > n) k = n - k;
        if (k == 0) return 1;

        int result = n;
        for (int i = 2; i <= k; ++i) {
            result *= (n - i + 1);
            result /= i;
        }
        return result;
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

    std::vector<uint16_t> range(uint16_t start, uint16_t end) {
        std::vector<uint16_t> res(end - start);
        for (uint16_t i = start; i < end; i++) {
            res[i - start] = i;
        }
        return res;
    }

    std::vector<uint16_t> filtered_range(const std::vector<uint16_t> &filter_idx, uint16_t min, uint16_t max) {
        auto full_range = range(min, max);
        std::vector<uint16_t> res;
        std::copy_if(full_range.begin(), full_range.end(), std::back_inserter(res),
                     [&](const uint16_t idx) {
                         return std::none_of(filter_idx.begin(), filter_idx.end(),
                                             [&](const uint16_t &f_idx) { return idx == f_idx; });
                     });
        return res;
    }

    Mat used_feature_orthogonalize(const Mat &X, const Mat &Q,
                                   const std::vector<Feature> &used_features) {
        uint16_t N_features = X.cols();
        Mat Q_current = Mat::Zero(X.rows(), X.cols());
        for (int k = 0; k < N_features; k++) {
            if (std::none_of(used_features.begin(), used_features.end(),
                             [&](const auto &feature) { return feature.index == k; })) {
                Q_current.col(k) = vec_orthogonalize(X.col(k), Q.leftCols(used_features.size()));
            }
        }
        return Q_current;
    }


} // namespace FROLS
