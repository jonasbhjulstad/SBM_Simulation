
#include <algorithm>
#include <stdint.h>
namespace FROLS
{

    template <typename M>
    struct Simplex
    {
        void pivot(uint32_t row, uint32_t col)
        {
            float p_val = tab(row, col);
            std::for_each(tab.row(row).begin(), tab.row(row).end(), [&p_val](auto& elem){elem /= p_val;});
            for (int i = 0; i < tab.rows(); i++)
            {
                float factor = tab(i, col);
                for (int j = 0; j < tab.cols(); j++)
                {
                    tab(i,j) -= factor * tab(row, j);
                }
            }
        }

        uint32_t find_pivot()
        {
            uint32_t j, piv_idx = 1;
            //find the most negative column in tab
            float lowest = tab(0, piv_idx);

        }

        M tab;
    };
}