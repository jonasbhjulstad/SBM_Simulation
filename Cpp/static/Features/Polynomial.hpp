#ifndef FROLS_POLYNOMIAL_HPP
#define FROLS_POLYNOMIAL_HPP

#include "Feature_Model.hpp"
#include <Regressor.hpp>
#include <Typedefs.hpp>

namespace FROLS::Features {

    struct Polynomial_Model : public Feature_Model {
        const size_t d_max;
        const size_t Nx;
        const size_t Nu;

        Polynomial_Model(size_t Nx, size_t Nu, size_t N_output_features, size_t d_max,
                         const std::vector<size_t> ignore_idx = std::vector<size_t>())
                : d_max(d_max), Nx(Nx), Nu(Nu), Feature_Model(N_output_features, ignore_idx) {}


        // double transform(crVec &x_raw, size_t target_index) ;
        Vec _transform(crMat &X_raw, size_t target_index, bool& index_failure);


        const std::vector<std::vector<Feature>> get_features();

        void write_csv(const std::string &);

        void feature_summary();

        const std::string feature_name(size_t target_index, bool indent = true);

        const std::vector<std::string> feature_names();

        const std::string model_equation(size_t idx);

        const std::string model_equations();


    private:
        bool index_warning_used = false;
    };

} // namespace FROLS::Features::Polynomial
#endif