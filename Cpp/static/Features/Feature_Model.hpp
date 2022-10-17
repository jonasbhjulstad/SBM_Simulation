#ifndef FROLS_FEATURE_MODEL_HPP
#define FROLS_FEATURE_MODEL_HPP

#include <Typedefs.hpp>
#include <FROLS_Path_Config.hpp>
#include <memory>

namespace FROLS::Features {
    struct Feature_Model {

        Feature_Model(const uint16_t N_output_features);

        Vec step(crVec &x, crVec &u);

        Mat simulate(crVec &x0, crMat &U, uint16_t Nt);

        Vec transform(crMat &X_raw, uint16_t target_index, bool &index_failure);

        Mat transform(crMat &X_raw, const std::vector<Feature> preselected_features = {});


        virtual Vec _transform(crMat &X_raw, uint16_t target_index, bool &index_failure) = 0;

        virtual const std::vector<std::vector<Feature>> get_features() = 0;

        virtual void write_csv(const std::string &) = 0;

        virtual void feature_summary() = 0;

        virtual const std::string feature_name(uint16_t target_index,
                                               bool indent = true) = 0;

        virtual const std::vector<std::string> feature_names() = 0;

        virtual const std::string model_equation(uint16_t idx) = 0;

        virtual const std::string model_equations() = 0;
        virtual uint16_t get_feature_index(const std::string&) = 0;
        void ignore(const std::string&);
        void ignore(uint16_t);
        void preselect(const std::string&, float, uint16_t, Feature_Tag);
        void preselect(uint16_t, float, uint16_t, Feature_Tag);

        const uint16_t N_output_features;
        std::vector<uint16_t> ignore_idx;
        std::vector<std::vector<Feature>> preselected_features;
        std::vector<std::vector<Feature>> features;

        std::vector<uint16_t> get_candidate_feature_idx() { return candidate_feature_idx; }

        std::vector<uint16_t> get_preselect_feature_idx() { return preselect_feature_idx; }

    protected:
        std::vector<uint16_t> preselect_feature_idx;
        std::vector<uint16_t> candidate_feature_idx;
    };
} // namespace FROLS::Features

#endif // FEATURE_MODEL_HPP