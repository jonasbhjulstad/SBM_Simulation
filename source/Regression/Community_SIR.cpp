#include <SBM_Simulation/Regression/Community_SIR.hpp>
#include <SBM_Database/Simulation/Size_Queries.hpp>
#include <SBM_Database/Graph/Graph_Tables.hpp>
namespace SBM_Regression
{
    Vec compute_beta_rs_col(uint32_t from_idx, uint32_t to_idx, const Vec &p_I)
    {
        Mat target_traj =
            community_traj(Eigen::seqN(0, Nt), Eigen::seqN(3 * to_idx, 3));
        Mat source_traj =
            community_traj(Eigen::seqN(0, Nt), Eigen::seqN(3 * from_idx, 3));
        Vec p_I_trunc = p_I(Eigen::seqN(0, Nt));
        Vec const S_r = source_traj.col(0);
        Vec I_r = source_traj.col(1);
        Vec const R_r = source_traj.col(2);
        Vec S_s = target_traj.col(0);
        Vec const I_s = target_traj.col(1);
        Vec R_s = target_traj.col(2);

        Vec denom = (S_s.array() + I_r.array() + R_s.array()).matrix();
        Vec nom = (S_s.array() * I_r.array()).matrix();
        return Vec(p_I_trunc.array() * nom.array() / denom.array());
    };

    Mat compute_beta_rs_mat(const Sim_Dimensions& dims)
    {
        Mat F_beta_rs_mat(Nt, N_connections);
        for (int i = 0; i < connection_community_map.rows(); i++)
        {
            F_beta_rs_mat.col(i) =
                compute_beta_rs_col(connection_community_map(i, 0),
                                    connection_community_map(i, 1), p_Is.col(i));
        }
        return F_beta_rs_mat;
    }
}