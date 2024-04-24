#ifndef FROLS_FEATURE_MODEL_HPP
#define FROLS_FEATURE_MODEL_HPP

#include <Typedefs.hpp>
#include <FROLS_Path_Config.hpp>
#include <memory>
#include <string>
#include <vector>

namespace FROLS::Features {
    struct Feature_Model {

        Feature_Model(const uint32_t N_output_features);

        Vec step(const Vec &x, const Vec &u, const std::vector<std::vector<Feature>>& features);

        Mat simulate(const Vec &x0, const Mat &U, uint32_t Nt, const std::vector<std::vector<Feature>>& features);

        Vec transform(const Mat &X_raw, uint32_t target_index);

        Mat transform(const Mat &X_raw);


        virtual Vec _transform(const Mat &X_raw, uint32_t target_index) = 0;


        virtual void write_csv(const std::string &, const std::vector<std::vector<Feature>>& features) = 0;

        virtual const std::vector<std::vector<Feature>> read_csv(const std::string &) = 0;

        virtual void feature_summary(const std::vector<std::vector<Feature>>& features) = 0;

        virtual const std::string feature_name(uint32_t target_index,
                                               bool indent = true) = 0;

        virtual const std::vector<std::string> feature_names() = 0;

        virtual const std::string model_equation(const std::vector<Feature>&) = 0;

        void write_latex(const std::vector<std::vector<Feature>>& features, const std::string &filename, const std::vector<std::string>& x_names, const std::vector<std::string>& u_names, const std::vector<std::string>& y_names, bool with_align = false, const std::string line_prefix = "&");

        virtual uint32_t get_feature_index(const std::string&) = 0;

        uint32_t N_output_features;

        virtual ~Feature_Model() = default;

    };
} // namespace FROLS::Features

#endif // FEATURE_MODEL_HPP