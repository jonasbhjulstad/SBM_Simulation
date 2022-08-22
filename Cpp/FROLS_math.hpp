#ifndef FROLS_MATH_HPP
#define FROLS_MATH_HPP
#include "FROLS_Typedefs.hpp"

namespace FROLS
{
    Mat cov_normalize(const Vec& a, const Vec& b)
    {
        return a.T * b / a.T * a;
    }

    Vec vec_orthogonalize(const Vec& v, const Mat& Q)
    {
        Vec cov_remainder = v;
        for (int i = 0; i < Q.cols(), i++)
        {
            cov_remainder -= cov_normalize(Q[:, i], cov_remainder)*Q[:,i];
        }
        return cov_remainder;
    }
    
    
}


#endif