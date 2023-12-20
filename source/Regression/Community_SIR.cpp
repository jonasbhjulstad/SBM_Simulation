#include <SBM_Simulation/Regression/Community_SIR.hpp>
#include <SBM_Database/Simulation/Size_Queries.hpp>
#include <SBM_Database/Graph/Graph_Tables.hpp>
namespace SBM_Regression
{
    Vec compute_beta_rs_col(const Vec& X_from, const Vec& X_to, const Vec &p_I)
    {
        auto Nt = p_I.rows();
        Vec p_I_trunc = p_I(Eigen::seqN(0, Nt));
        Vec const S_r = X_from.col(0);
        Vec I_r = X_from.col(1);
        Vec const R_r = X_from.col(2);
        Vec S_s = X_to.col(0);
        Vec const I_s = X_to.col(1);
        Vec R_s = X_to.col(2);

        Vec denom = (S_s.array() + I_r.array() + R_s.array()).matrix();
        Vec nom = (S_s.array() * I_r.array()).matrix();
        return Vec(p_I_trunc.array() * nom.array() / denom.array());
    };

    Mat compute_beta_rs_mat(const SBM_Database::Sim_Dimensions& dims, const std::vector<SBM_Graph::Weighted_Edge_t>& ccm, const Mat& p_Is, const Mat& X_data)
    {
        Mat F_beta_rs_mat(dims.Nt, dims.N_connections);
        for (int i = 0; i < ccm.size(); i++)
        {
            F_beta_rs_mat.col(i) =
                compute_beta_rs_col(X_data.col(ccm[i].from),
                                    X_data.col(ccm[i].to), p_Is.col(i));
        }
        return F_beta_rs_mat;
    }
}