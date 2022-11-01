#ifndef FROLS_POLYNOMIAL_DISCRETE_HPP
#define FROLS_POLYNOMIAL_DISCRETE_HPP

#include "Feature_Model.hpp"
#include <Regressor.hpp>
#include <Typedefs.hpp>

namespace FROLS::Features {

    struct Polynomial_Model : public Feature_Model {
        const uint32_t d_max;
        const uint32_t Nx;
        const uint32_t Nu;

        Polynomial_Model(uint32_t Nx, uint32_t Nu, uint32_t d_max, uint32_t N_output_features = 20)
                : d_max(d_max), Nx(Nx), Nu(Nu), Feature_Model(std::min({N_output_features, get_N_features_max()})) 
                {
                    if (N_output_features > get_N_features_max()) {
                        std::cout << "Warning: N_output_features is greater than the number supported" << std::endl;
                        N_output_features = get_N_features_max();
                    }

                }

        // float transform(Vec &x_raw, uint32_t target_index) ;
        Vec _transform(const Mat &X_raw, uint32_t target_index);



        void write_csv(const std::string &, const std::vector<std::vector<Feature>>& features);
        const std::vector<std::vector<Feature>> read_csv(const std::string &);

        void feature_summary(const std::vector<std::vector<Feature>>& features);

        const std::string feature_name(uint32_t target_index, bool indent = true);

        const std::vector<std::string> feature_names();

        const std::string model_equation(const std::vector<Feature>& features);
        const std::string model_equations(const std::vector<std::vector<Feature>>& features);

        uint32_t get_feature_index(const std::string&);
        uint32_t get_N_features_max() const;

    private:
        bool index_warning_used = false;
    };

} // namespace FROLS::Features::Polynomial
#endif