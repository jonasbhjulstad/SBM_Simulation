#include "Feature_Model.hpp"
#include "Regressor.hpp"
#include <regex>
#include <fmt/format.h>
namespace FROLS::Features
{

    Feature_Model::Feature_Model(const uint32_t N_output_features) : N_output_features(
                                                                         N_output_features) {}

    Vec Feature_Model::transform(const Mat &X_raw, uint32_t target_index)
    {
        return _transform(X_raw, target_index);
    }

    Mat Feature_Model::transform(const Mat &X_raw)
    {
        Mat res(X_raw.rows(), N_output_features);
        for (int i= 0; i < N_output_features; i++) {
            res.col(i) = _transform(X_raw, i);
        }
        return res;
    }


    Vec Feature_Model::step(const Vec &x, const Vec &u, const std::vector<std::vector<Feature>> &features)
    {
        Vec x_next(x.rows());
        Mat X(1, x.rows() + u.rows());
        X << x.transpose(), u.transpose();
        bool index_failure = false;
        x_next.setZero();
        for (int i = 0; i < features.size(); i++)
        {
            for (int j = 0; j < features[i].size(); j++)
            {
                x_next(i) +=
                    features[i][j].theta *
                    _transform(X, features[i][j].index).value();
                if (index_failure)
                {
                    break;
                }
            }
        }

        return x_next;
    }

    Mat Feature_Model::simulate(const Vec &x0, const Mat &U, uint32_t Nt, const std::vector<std::vector<Feature>> &features)
    {
        Mat X(Nt + 1, x0.rows());
        X.setZero();
        X.row(0) = x0;
        for (int i = 0; i < Nt; i++)
        {
            X.row(i + 1) = step(X.row(i), U.row(i), features);
        }
        return X;
    }

    void Feature_Model::write_latex(const std::vector<std::vector<Feature>> &features, const std::string &filename, const std::vector<std::string> &x_names, const std::vector<std::string> &u_names, const std::vector<std::string> &y_names, bool with_align, const std::string line_prefix)
    {
        std::ofstream file(filename);

        file << (with_align ? "\\begin{align}" : "") << std::endl;
        std::for_each(features.begin(), features.end(), [&, n = 0](const auto &feature_set) mutable
                      {
            file << line_prefix;
            file << (with_align ? "" : "$") << y_names[n] << " " << (with_align ? "&" : "") << "= ";

            for (int i = 0; i < feature_set.size(); i++)
            {
                std::string name = feature_name(feature_set[i].index);
                //replace occurences of x0, x1, ... with x_names, do not use lambda
                for (int j = 0; j < x_names.size(); j++)
                {
                    std::string x_name = "x" + std::to_string(j);
                    std::regex re(x_name);
                    name = std::regex_replace(name, re, x_names[j]);
                }
                //do the same for u and y
                for (int j = 0; j < u_names.size(); j++)
                {
                    std::string u_name = "u" + std::to_string(j);
                    std::regex re(u_name);
                    name = std::regex_replace(name, re, u_names[j]);
                }
                // std::regex re("\\d");
                // name = std::regex_replace(name, re, "_$&");
                // Create string of feature_set[i].theta with 3 decimal places without using stringstream
                std::string theta_str = std::to_string(feature_set[i].theta);
                theta_str = theta_str.substr(0, theta_str.find('.') + 4);
                

                file << theta_str << name;
                if (i < (feature_set.size() - 1))
                {
                    file << " + ";
                }
            }
            n++;
            file << (with_align ? "" : "$") << "&" << fmt::format("{:.2e}", feature_set.back().f_ERR) << std::endl;
            file << ((n < features.size()) ? "\\\\" : "") << std::endl; });
        file << (with_align ? "\\end{align}" : "") << std::endl;
    }

} // namespace FROLS::Features