#pragma once

#include <SIR_SBM/common.hpp>

#include <filesystem>
#include <fstream>

#include <type_traits>

namespace SIR_SBM {
Vec2D<uint32_t> read_csv(const std::filesystem::path &path, uint32_t N0,
                         uint32_t N1);

Vec3D<uint32_t> read_csv(const std::filesystem::path &file_prefix, int N0,
                         int N1, int N2);

void write_csv(const std::vector<uint32_t> &data,
               const std::filesystem::path &path, int N0, int N1);
} // namespace SIR_SBM