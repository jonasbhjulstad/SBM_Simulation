#include "FROLS_Typedefs.hpp"
#include <casadi/casadi.hpp>

namespace FROLS
{
    Vec run(const Mat& X, const Mat& Y, double ERR_tolerance = 1e-1)
    {
        using Eigen::seq;
        size_t N_samples = X.rows();
        size_t N_features = X.cols();
        size_t N_response = Y.cols();

        std::vector<Vec> best_coeffs;

        //Perform one FROLS-regression for each response variable
        for (int i = 0; i < N_response; i++)
        {
            Vec g = Vec::Zero(N_features);
            Mat A = Mat::Zero(N_features, N_features);
            iVec feature_inidices = iVec::Zero();
            Mat Q_global = x;
            Mat Q_current = x;
            
            Vec yi = Y.col(i);
            std::vector<Feature> best_features;

            for (int j = 0; j < N_features; j++)
            {
                for(int k = 0; k < N_features; k++)
                {
                    if (L.head(k).cwiseEqual(k).any())
                    {
                        Q_current.col(k) = vec_orthogonalize(x.col(k), Q.col(j));
                    }
                }
                best_features.push_back(feature_select(Q_current, yi, feature_indices));

                for (int m = 0; m < j; m++)
                {
                    A[m,j] = cov_normalize(Q_current.col(m), X.col(feature_indices[j]));
                }
                A[j, j] = 1.;
                Q.col(j) = Q_current.col(feature_indices[j]);

                double ERR_tot = 0;
                for (const auto& feature: best_features)
                {
                    ERR_tot += feature.ERR;
                }
                if ((1-ERR_tot) < ERR_tolerance)
                {
                    break;
                }

                Q_current = Mat::Zero(N_samples, N_features);                
            }
            best_coeffs.push_back(A.topLeft(j, j).inverse())


        }
    }
}