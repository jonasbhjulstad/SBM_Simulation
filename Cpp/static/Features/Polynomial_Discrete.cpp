#include "Polynomial_Discrete.hpp"
#include <Eigen/src/Core/Array.h>
#include <FROLS_Math.hpp>
#include <iomanip>
#include <iostream>
#include <itertools.hpp>
#include <fmt/format.h>

namespace FROLS::Features {

    void Polynomial_Model::feature_summary() {
        fmt::print("{:^15}{:^15}{:^15}{:^15}{:^15}{:^15}\n", "y", "Feature", "g", "Theta", "f_ERR", "Tag");
        for (int i = 0; i < features.size(); i++) {
            for (const auto &feature: features[i]) {
                uint32_t absolute_idx = (feature.tag == FEATURE_PRESELECTED) ? feature.index : candidate_feature_idx[feature.index];
                std::string name = feature_name(absolute_idx, false);
                name = (name == "") ? "1" : name;

                // print features aligned with tabs
                fmt::print("{:^15}{:^15}{:^15.3f}{:^15.3f}{:^15.3f}{:^15}{:^15}\n", i, name, feature.g, feature.theta,
                           feature.f_ERR, feature_tag_map.at(feature.tag), absolute_idx);
            }
        }
    }

    void Polynomial_Model::write_csv(const std::string &filename) {
        std::ofstream f(filename);
        f << "Response,Feature,Index,g, theta, ERR" << std::endl;
        for (int i = 0; i < features.size(); i++) {
            for (auto &feature: features[i]) {
                f << i << "," << feature_name(feature.index) << ","
                  << feature.index << "," << feature.g << ", " << feature.theta << "," << feature.f_ERR
                  << std::endl;
            }
        }
    }

    void Polynomial_Model::read_csv(const std::string & filename)
    {
        features = {};
        //read response, feature, index, g, theta, ERR
        std::ifstream f(filename);
        std::string line;
        std::getline(f, line); // skip header
        std::vector<Feature> response_features;
        size_t response_i = 0;
        while (std::getline(f, line)) {
            std::stringstream ss(line);
            std::string response, feature, index, g, theta, ERR;
            std::getline(ss, response, ',');
            if (std::stoi(response) != response_i)
            {
                response_i = std::stoi(response);
                features.push_back(response_features);
                response_features = {};
            }
            std::getline(ss, feature, ',');
            std::getline(ss, index, ',');
            std::getline(ss, g, ',');
            std::getline(ss, theta, ',');
            std::getline(ss, ERR, ',');
            uint32_t response_idx = std::stoi(response);
            uint32_t feature_idx = std::stoi(index);
            float g_val = std::stof(g);
            float theta_val = std::stof(theta);
            float ERR_val = std::stof(ERR);

            response_features.push_back(Feature{ERR_val, g_val, feature_idx, theta_val, FEATURE_PRESELECTED});
        }
        features.push_back(response_features);
    }


    const std::vector<std::vector<Feature>> Polynomial_Model::get_features() {
        if (Nx == std::numeric_limits<uint32_t>::max() || Nu == std::numeric_limits<uint32_t>::max()) {
            std::cout << "Model not yet trained" << std::endl;
            return {};
        }
        return features;
    }


    Vec Polynomial_Model::_transform(crMat &X_raw, uint32_t target_idx, bool &index_failure) {
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
        if ((feature_idx < target_idx) && !index_warning_used) {
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
            model += std::to_string(rd[i].theta) + " ";
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

    uint32_t Polynomial_Model::get_feature_index(const std::string& name)
    {
        uint32_t index = 0;
        std::string trial_name = ".";
        while ((trial_name != name))
        {
            trial_name = feature_name(index, false);
            if (trial_name == "")
            {
                fmt::print("[Polynomial Model] Warning: Unable to find index of feature {}", name);
                break;
            }
            index++;
        }
        return index-1;
    }

} // namespace FROLS::Features::Polynomial