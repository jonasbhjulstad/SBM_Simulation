#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#include <DataFrame.hpp>
#include <FROLS_Eigen.hpp>
#include <Features.hpp>
#include <Quantile_Regressor.hpp>
#include <ERR_Regressor.hpp>
#include <matplotlibcpp.h>
#include <iostream>



Vec simulate(Vec x0, Mat A, Vec B, Vec u, size_t Nt)
{
  size_t Nx = x0.rows();
  Mat y(Nt+1, Nx);
  y.row(0) = x0;
  for (int i = 0; i < Nt; i++)
  {
    y.row(i+1) = A*y.row(i) + B*u;
  }
  return y;
}

void plot(Vec x, Vec y)
{
  std::vector<double> xv(x.data(), x.data() + x.rows());
  std::vector<double> yv(y.data(), y.data() + y.rows());

  namespace plt = matplotlibcpp;
  plt::figure();
  plt::plot(xv, yv, "r");
}


int main() {
  using namespace FROLS;

  double x0 = 0;
  Mat traj = simulate(x0, Nt)
  Mat X(N_rows, N_features);

  // create sine input u
  Vec u = Vec::LinSpaced(Nt, 0, 2*M_PI);
  u = u.array().sin();
  //create oscillator system
  Mat A = Mat::Zero(1,1);
  A(0,0) = -1;
  A(0,1) = 1;
  A(1,0) = -1;
  Vec B = Vec::Ones(1);
  Vec x0 = Vec::Zero(1);
  Mat traj = simulate(x0, A, B, u, Nt);
  size_t Nt = 100;
  plt::subplot(1,2,1);
  Vec t = Vec::LinSpaced(Nt, 0, 1);
  plot(t, traj.col(0));
  plt::subplot(1,2,2);
  plot(t, traj.col(1));


  // size_t d_max = 1;
  // using namespace FROLS::Features;

  // double ERR_tolerance = 1e-2;
  // FROLS::Features::Polynomial_Model model(Nx, Nu, N_output_features, d_max);
  // double MAE_tol = 1.0;
  // double tau = .95;
  // Regression::Quantile_Regressor qr(tau, MAE_tol);
  // qr.transform_fit(X, U, Y, model);
  // Regression::ERR_Regressor er(ERR_tolerance);
  // er.transform_fit(X, U, Y, model);

  // Vec x0 = X.row(0);
  // double u0 = U(0, 0);
  // size_t Nt = 30;
  // Vec u = Vec::Ones(Nt) * u0;
  // // print x0, u
  // std::cout << "x0 = " << x0.transpose() << std::endl;
  // std::cout << "u0 = " << u0 << std::endl;
  // model.feature_summary();
  // model.simulate(x0, u, Nt);

  return 0;
}