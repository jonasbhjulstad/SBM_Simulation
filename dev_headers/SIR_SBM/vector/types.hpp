#pragma once
#hdr
#include <vector>
#include <cstdint>

namespace SIR_SBM
{
    template <typename T>
    using Vec2D = std::vector<std::vector<T>>;
    template <typename T>
    using Vec3D = std::vector<std::vector<std::vector<T>>>;
}
#end