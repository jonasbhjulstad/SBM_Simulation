#ifndef FROLS_FEATURE_MODEL_HPP
#define FROLS_FEATURE_MODEL_HPP

#include <Typedefs.hpp>
#include <FROLS_Path_Config.hpp>
#include <memory>

namespace FROLS::Features {
    struct Feature_Model {

        Feature_Model(const size_t N_output_features, const std::vector<size_t> ignore_idx = std::vector<size_t>(),
                      const std::vector<std::vector<Feature>> preselected_features = std::vector<std::vector<Feature>>());

        Vec step(crVec &x, crVec &u);

        Mat simulate(crVec &x0, crMat &U, size_t Nt);

        Vec transform(crMat &X_raw, size_t target_index, bool &index_failure);

        Mat transform(crMat &X_raw, const std::vector<Feature> preselected_features = {});


        virtual Vec _transform(crMat &X_raw, size_t target_index, bool &index_failure) = 0;

        virtual const std::vector<std::vector<Feature>> get_features() = 0;

        virtual void write_csv(const std::string &) = 0;

        virtual void feature_summary() = 0;

        virtual const std::string feature_name(size_t target_index,
                                               bool indent = true) = 0;

        virtual const std::vector<std::string> feature_names() = 0;

        virtual const std::string model_equation(size_t idx) = 0;

        virtual const std::string model_equations() = 0;

        const size_t N_output_features;
        std::vector<size_t> ignore_idx;
        const std::vector<std::vector<Feature>> preselected_features;
        std::vector<std::vector<Feature>> features;

        std::vector<size_t> get_candidate_feature_idx() { return candidate_feature_idx; }

        std::vector<size_t> get_preselect_feature_idx() { return preselect_feature_idx; }

    protected:
        std::vector<size_t> preselect_feature_idx;
        std::vector<size_t> candidate_feature_idx;
    };
} // namespace FROLS::Features

#endif // FEATURE_MODEL_HPP