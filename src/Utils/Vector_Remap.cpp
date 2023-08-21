#include <Sycl_Graph/Utils/Vector_Remap.hpp>
#include <stdexcept>
std::vector<std::vector<std::vector<State_t>>> vector_remap(std::vector<State_t> &input, size_t N0, size_t N1, size_t N2)
{
    size_t M = N0 * N1 * N2;
    if (input.size() != M)
    {
        throw std::runtime_error("Input vector size does not match 3D dimensions.");
    }

    std::vector<std::vector<std::vector<State_t>>> output(N0, std::vector<std::vector<State_t>>(N1, std::vector<State_t>(N2)));

    // linear position defined by i2 + i1*n2 + i0*n1*n2
    for (size_t i0 = 0; i0 < N0; i0++)
    {
        for (size_t i1 = 0; i1 < N1; i1++)
        {
            for (size_t i2 = 0; i2 < N2; i2++)
            {
                output[i0][i1][i2] = input[i2 + i1 * N2 + i0 * N1 * N2];
            }
        }
    }
    return output;
}

std::vector<std::vector<std::vector<uint32_t>>> vector_remap(std::vector<uint32_t> &input, size_t N0, size_t N1, size_t N2)
{
    size_t M = N0 * N1 * N2;
    if (input.size() != M)
    {
        throw std::runtime_error("Input vector size does not match 3D dimensions.");
    }

    std::vector<std::vector<std::vector<uint32_t>>> output(N0, std::vector<std::vector<uint32_t>>(N1, std::vector<uint32_t>(N2)));

    //linear position defined by i2 + i1*n2 + i0*n1*n2
    for (size_t i0 = 0; i0 < N0; i0++)
    {
        for (size_t i1 = 0; i1 < N1; i1++)
        {
            for (size_t i2 = 0; i2 < N2; i2++)
            {
                output[i0][i1][i2] = input[i2 + i1 * N2 + i0 * N1 * N2];
            }
        }
    }
    return output;
}
