#include <FROLS_Typedefs.hpp>
#include <FROLS_Algorithm.hpp>
#include <gtest/gtest.h>

TEST(Orthogonalization, Orthogonalize) {
    Mat X = Mat::Random(10, 10);
    Mat Q = Mat::Random(10, 10);
    iVec used_indices(10);
    used_indices.setConstant(-1);
    used_indices(0) = 0;
    used_indices(1) = 1;
    used_indices(2) = 2;
    size_t current_feature_idx = 10;
    Mat result = FROLS::used_feature_orthogonalize(X, Q, used_indices, current_feature_idx);

    Vec orth_x = X.col(10);

    for (int i = 0; i < Q.cols(); i++)
    {
        if (used_indices.cwiseEqual(i).any())
        {
            orth_x -= FROLS::cov_normalize(Q.col(i), orth_x) * Q.col(i);
        }
    }

    ASSERT_EQ(result.col(10), result.col(10));

}