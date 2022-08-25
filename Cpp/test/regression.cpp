#include <FROLS.hpp>
#include <gtest/gtest.h>



TEST(Regression, single_response_regression)
{
    size_t N_features = 2;
    size_t N_samples = 10;
    Mat X = Mat::Zero(N_samples, N_features);
    //Linear feature
    X.col(0) = Vec::LinSpaced(N_samples, 0, 1);
    //Quandratic feature
    X.col(1) = X.col(0).cwiseProduct(X.col(0));

    Vec y = X.col(1);

    double ERR_tolerance = 1e-4;
    auto result = FROLS::Regression::single_response_batch(X, y, ERR_tolerance);

}

TEST(Regression, multiple_response_regression)
{

}