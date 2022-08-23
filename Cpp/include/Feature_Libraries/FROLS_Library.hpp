#ifndef FROLS_LIBRARY_HPP
#define FROLS_LIBRARY_HPP
#include <FROLS_Typedefs.hpp>

namespace FROLS
{
    namespace Library
    {
        namespace Polynomial
        {
            Mat generate_features(const Mat& X_data, size_t N_features, size_t offset = 0)
            {
                Mat X(X_data.rows(), N_features);
                
            }
        }
    }
}

#endif