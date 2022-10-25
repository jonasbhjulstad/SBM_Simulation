#ifndef FROLS_REGRESSOR_HPP
#define FROLS_REGRESSOR_HPP

#include <FROLS_Path_Config.hpp>
#include <Feature_Model.hpp>
#include <FROLS_Math.hpp>
#include <vector>
#include <memory>

namespace FROLS::Regression
{
    struct Regressor_Param
    {
        float tol = 1e-2;
        float theta_tol = 1e-4;
        uint32_t N_terms_max = 4;
    };

    struct Regressor
    {
        const float tol, theta_tol;
        const uint32_t N_terms_max;

        Regressor(const Regressor_Param &);

        std::vector<std::vector<Feature>> fit(crMat &X, crMat &Y);

        void
        transform_fit(crMat &X, crMat &U, crMat &Y,
                      Features::Feature_Model &model);

        void transform_fit(const std::vector<std::string> &filenames, const std::vector<std::string> &colnames_x, const std::vector<std::string> &colnames_u, Features::Feature_Model &model);
        virtual void theta_solve(crMat &A, crVec &g, crMat& X, crVec& y, std::vector<Feature> &features) const = 0;

    protected:
        Vec predict(crMat &X, const std::vector<Feature> &features) const;

        std::vector<uint32_t> unused_feature_indices(const std::vector<Feature> &features, uint32_t N_features) const;

    private:

        std::vector<Feature> single_fit(const Mat &X, const Vec &y) const;

        std::vector<Feature> single_fit(const Mat &X, const Vec &y, std::vector<Feature> preselect_features) const;

        virtual std::vector<Feature> candidate_regression(crMat &X, crMat& Q_global, crVec &y,
                                                          const std::vector<Feature> &used_features) const = 0;

        virtual bool
        tolerance_check(crMat &Q, crVec &y,
                        const std::vector<Feature> &best_features) const = 0;

        virtual Feature feature_selection_criteria(const std::vector<Feature>& candidate_features) const = 0;

        Feature best_feature_select(crMat &X, crMat& Q_global, crVec &y, const std::vector<Feature> &used_features) const;

        static int regressor_count;

    };
} // namespace FROLS::Regression

#endif