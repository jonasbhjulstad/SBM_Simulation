#include "Polynomial_Discrete.hpp"
#include <Eigen/src/Core/Array.h>
#include <FROLS_Math.hpp>
#include <iomanip>
#include <iostream>
#include <itertools.hpp>
namespace FROLS::Features {


    void Polynomial_Model::feature_summary() {
        std::cout << "y\tFeature\t\tg\t\tTheta\t\tf_ERR\n";
        std::cout << std::fixed << std::setprecision(4);
        for (int i = 0; i < features.size(); i++) {
            for (auto &feature: features[i]) {
                std::string name = (feature.f_ERR == -1) ? "-" : feature_name(candidate_feature_idx[feature.index], false);
                name = (name == "") ? "1" : name;
                // print features aligned with tabs
                std::cout << i << "\t" << name << "\t\t" << feature.g << "\t\t"
                          << feature.theta << "\t\t" << feature.f_ERR << std::endl;
            }
        }
    }

    void Polynomial_Model::write_csv(const std::string &filename) {
        std::ofstream f(filename);
        f << "Response,Feature,Index,g,ERR" << std::endl;
        for (int i = 0; i < features.size(); i++) {
            for (auto &feature: features[i]) {
                f << i << "," << feature_name(feature.index) << ","
                  << feature.index << "," << feature.g << "," << feature.f_ERR
                  << std::endl;
            }
        }
    }

    const std::vector<std::vector<Feature>> Polynomial_Model::get_features() {
        if (Nx == -1 || Nu == -1) {
            std::cout << "Model not yet trained" << std::endl;
            return {};
        }
        return features;
    }


    Vec Polynomial_Model::_transform(Mat &X_raw, uint32_t target_idx, bool& index_failure) {
        // get feature names for polynomial combinations with powers between d_min,
        // d_max of the original features
        uint32_t N_input_features = X_raw.cols();
        uint32_t N_rows = X_raw.rows();
        uint32_t feature_idx = 0;
        // for (int d = 1; d < d_max; d++) {
        for (auto &&comb: iter::combinations_with_replacement(range(0, d_max + 1),
                                                              N_input_features)) {
            // generate all combinations of powers of the original features up to d_max
            for (auto &&powers: iter::permutations(comb)) {
                if (feature_idx == target_idx) {
                    return monomial_powers(X_raw, powers);
                }
                feature_idx++;
            }
        }
        index_failure = true;
        return Vec::Zero(N_rows);
    }

    const std::string Polynomial_Model::feature_name(uint32_t target_idx,
                                                     bool indent) {

        std::string feature_name;
        uint32_t feature_idx = 0;
        uint32_t N_input_features = Nx + Nu;
        // for (int d = 1; d < d_max; d++) {
        for (auto &&comb: iter::combinations_with_replacement(range(0, d_max + 1),
                                                              N_input_features)) {
            for (auto &&powers: iter::permutations(comb)) {

                if (feature_idx == target_idx) {
                    uint32_t x_idx = 0;
                    for (auto &&pow: powers) {

                        std::string x_or_u = x_idx < Nx ? "x" : "u";
                        uint32_t idx_offset = x_idx < Nx ? 0 : Nx;
                        feature_name +=
                                (pow > 0) ? x_or_u + std::to_string(x_idx - idx_offset) : "";
                        feature_name += (pow > 1) ? "^" + std::to_string(powers[x_idx]) : "";
                        feature_name += ((pow > 0) && indent) ? " " : "";
                        x_idx++;
                    }
                    feature_name = (feature_name.empty()) ? "1" : feature_name;
                    return feature_name;
                }
                feature_idx++;
            }
        }
        if ((feature_idx < target_idx)&& !index_warning_used) {
            std::cout << "[Polynomial_Model] Warning: Target index is not contained in the permutation set\n";
            index_warning_used = true;
        }
        return "";
    }

    const std::vector<std::string>
    Polynomial_Model::feature_names() {
        // get feature names for polynomial combinations with powers between d_min,
        // d_max of the original features
        uint32_t N_input_features = Nx + Nu;
        std::vector<std::string> feature_names;
        for (auto &&comb: iter::combinations_with_replacement(range(0, d_max + 1),
                                                              N_input_features)) {
            for (auto &&powers: iter::permutations(comb)) {
                std::string feature_name = "";
                uint32_t x_idx = 0;
                for (auto &&pow: powers) {
                    feature_name += (pow > 0) ? "x" + std::to_string(x_idx) : "";
                    feature_name += (pow > 1) ? "^" + std::to_string(powers[x_idx]) : "";
                    feature_name += (pow > 0) ? " " : "";
                    x_idx++;
                }
                if (!feature_name.empty())
                    feature_names.push_back(feature_name);
            }
        }
        return feature_names;
    }

    const std::string Polynomial_Model::model_equation(uint32_t idx) {
        std::string model;
        const std::vector<Feature> &rd = features[idx];
        for (int i = 0; i < rd.size(); i++) {
            model += std::to_string(rd[i].theta);
            model += feature_name(rd[i].index);
            if (i != rd.size() - 1) {
                model += " + ";
            }
        }
        return model;
    }

    const std::vector<std::string> Polynomial_Model::model_equations() {
        std::vector<std::string> equations;
        for (int i = 0; i < features.size(); i++) {
            equations.push_back(model_equation(i));
        }
        return equations;
    }
} // namespace FROLS::Features::Polynomial