#ifndef FROLS_POLYNOMIAL_DISCRETE_HPP
#define FROLS_POLYNOMIAL_DISCRETE_HPP

#include "Feature_Model.hpp"
#include <Regressor.hpp>
#include <Typedefs.hpp>

namespace FROLS::Features {

    struct Polynomial_Model : public Feature_Model {
        const uint16_t d_max;
        const uint16_t Nx;
        const uint16_t Nu;

        Polynomial_Model(uint16_t Nx, uint16_t Nu, uint16_t N_output_features, uint16_t d_max)
                : d_max(d_max), Nx(Nx), Nu(Nu), Feature_Model(N_output_features) {}

        // float transform(crVec &x_raw, uint16_t target_index) ;
        Vec _transform(crMat &X_raw, uint16_t target_index, bool& index_failure);


        const std::vector<std::vector<Feature>> get_features();

        void write_csv(const std::string &);

        void feature_summary();

        const std::string feature_name(uint16_t target_index, bool indent = true);

        const std::vector<std::string> feature_names();

        const std::string model_equation(uint16_t idx);

        const std::string model_equations();

        uint16_t get_feature_index(const std::string&);

    private:
        bool index_warning_used = false;
    };

} // namespace FROLS::Features::Polynomial
#endif