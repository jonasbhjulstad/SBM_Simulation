#ifndef FROLS_REGRESSOR_HPP
#define FROLS_REGRESSOR_HPP

#include <FROLS_Path_Config.hpp>
#include <Feature_Model.hpp>
#include <FROLS_Math.hpp>
#include <vector>
#include <memory>


namespace FROLS::Regression {
    struct Regressor {
        const double tol, theta_tol;
        const size_t regressor_id;
        const size_t N_terms_max;
        Regressor(double tol, double theta_tol, size_t N_terms_max = std::numeric_limits<size_t>::infinity());

        std::vector<std::vector<Feature>> fit(crMat &X, crMat &Y);

        void
        transform_fit(crMat &X, crMat &U, crMat &Y,
                      Features::Feature_Model &model);

    protected:
        Vec predict(crMat &X, const std::vector<Feature> &features) const;

        std::vector<size_t> unused_feature_indices(const std::vector<Feature> &features, size_t N_features) const;

    private:
        void theta_solve(crMat &A, crVec &g, std::vector<Feature> &features) const;

            std::vector<Feature> single_fit(const Mat &X, const Vec &y) const;
            Mat used_feature_orthogonalize(const Mat &X, const Mat &Q,
                                           const std::vector<Feature> &used_features) const;
            virtual std::vector<Feature> candidate_regression(crMat &X, crVec &y,
                                                              const std::vector<Feature> &used_features) const = 0;
            virtual bool
            tolerance_check(crMat &Q, crVec &y,
                            const std::vector<Feature> &best_features) const = 0;
            Feature best_feature_select(crMat &X, crVec &y, const std::vector<Feature> &used_features) const;
            static int regressor_count;
        };
    } // namespace FROLS::Regression

#endif