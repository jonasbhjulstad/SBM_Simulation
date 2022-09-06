#include <FROLS_Algorithm.hpp>
#include <gtest/gtest.h>


TEST(FeatureSelection, feature_select)
{
    size_t N_features = 2;
    size_t N_samples = 10;
    Mat X = Mat::Zero(N_samples, N_features);
    //Linear feature
    X.col(0) = Vec::LinSpaced(N_samples, 0, 1);
    //Quandratic feature
    X.col(1) = X.col(0).cwiseProduct(X.col(0));

    Vec y = X.col(1);

    iVec used_features = iVec::Constant(N_features, -1);

    FROLS::Feature best_feature = FROLS::feature_select(X, y, used_features);    

}

