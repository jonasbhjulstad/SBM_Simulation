
#include <DataFrame.hpp>
#include <FROLS_Eigen.hpp>
#include <Features.hpp>
#include <Quantile_Regressor.hpp>
#include <ERR_Regressor.hpp>
#include <iostream>
#include <vector>

using namespace FROLS;
Vec linsys_step(const Mat &A, const Mat &b, const Vec &x, const Vec &u) {
    return A * x + b * u;
}

int main() {
    using namespace FROLS;
    using namespace FROLS::Regression;
    using namespace FROLS::Features;
    size_t Nx = 3;
    Vec x0(Nx);
    x0 << 10, 100, -100;

    Mat A(Nx, Nx);
    A << .9, 4., 0, 0., .5, 0, .1, 0, 0;
    size_t Nu = 1;
    Mat b(Nx, Nu);
    b.setConstant(1);

    size_t Nt = 100;
    Mat u(Nt, 1);
    size_t N_sims = 100;
    double omega = 4;
    u.col(0).setLinSpaced(Nt, 0, Nt * omega / (2 * M_PI));
    u = u.array().sin();
    Mat Y(Nt * N_sims, Nx);
    Mat X(Nt * N_sims, Nx);
    Mat U(N_sims * Nt, Nu);
    for (int j = 0; j < N_sims; j++) {
        Mat traj(Nt + 1, Nx);
        x0.setRandom();
        traj.row(0) = x0;
        u = u.array().setRandom().sin();
        for (int i = 0; i < Nt; i++) {
            traj.row(i + 1) = linsys_step(A, b, traj.row(i), u.row(i));
        }
        Y(Eigen::seqN(j * Nt, Nt), Eigen::all) = traj.bottomRows(Nt);
        X(Eigen::seqN(j * Nt, Nt), Eigen::all) = traj.topRows(Nt);
        U(Eigen::seqN(j * Nt, Nt), Eigen::all) = u;
    }


    size_t d_max = 1;
    size_t N_features = 16;
    double ERR_tol = 1e-5;
    double MAE_tol = 1;
    double tau = .95;
    double theta_tol = 10;
    Polynomial_Model model(Nx, Nu, N_features, d_max);
    ERR_Regressor er(ERR_tol, theta_tol);
    Quantile_Regressor qr(tau, MAE_tol, theta_tol);

    er.transform_fit(X, U, Y, model);
    model.feature_summary();
//    qr.transform_fit(X, U, Y, model);
//    model.feature_summary();



    return 0;
}