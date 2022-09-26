#include <Typedefs.hpp>

#include <Polynomial_Discrete.hpp>
#include <gtest/gtest.h>

TEST(Feature_Transform, single_feature) {
    Mat X = Mat::Random(10, 10);
    Mat Q = Mat::Random(10, 10);
    using namespace FROLS;
    size_t target_idx = 5;
    Mat X_poly = Features::Polynomial::feature_transform(X, 1, 10);
    Vec X_vec = Features::Polynomial::single_feature_transform(X, 1, target_idx);

    std::cout << X_poly << std::endl;

    ASSERT_EQ(X_poly.col(target_idx), X_vec);

}