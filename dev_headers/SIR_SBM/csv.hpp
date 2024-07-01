#pragma once
#hdr
#include <SIR_SBM/common.hpp>
#include <SIR_SBM/vector.hpp>
#include <filesystem>
#include <fstream>
#end

#src
#include <type_traits>
#end

namespace SIR_SBM {
template <typename T = int>
Vec2D<T> read_csv(const std::filesystem::path &path, uint32_t N0,
                             uint32_t N1) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + path.string());
  }
  Vec2D<T> result(N0, std::vector<T>(N1));
  std::string line;
  uint32_t n0 = 0;
  uint32_t n1 = 0;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
      if constexpr (std::is_same_v<T, int> || std::is_same_v<T, uint32_t>)
        result[n0][n1] = std::stoi(cell);
      else if constexpr (std::is_same_v<T, double>)
        result[n0][n1] = std::stod(cell);
      n1++;
    }
    n0++;
    n1 = 0;
  }
  return result;
}

template <typename T = int>
Vec3D<T> read_csv(const std::filesystem::path &file_prefix,
                                    int N0, int N1, int N2) {
  std::ifstream f;
  // std::vector<int> result(N0 * N1 * N2);
  Vec3D<T> result(N0, N1, N2);

  for (int i = 0; i < N0; i++) {
    std::string filename = file_prefix.string() + std::to_string(i) + ".csv";
    f.open(filename);
    if (!f.is_open()) {
      throw std::runtime_error("Could not open file: " + filename);
    }
    std::string line;
    int n1 = 0;
    while (std::getline(f, line)) {
      std::stringstream ss(line);
      std::string cell;
      int n2 = 0;
      while (std::getline(ss, cell, ',')) {
        if constexpr (std::is_same_v<T, int> || std::is_same_v<T, uint32_t>)
          result(i, n1, n2) = std::stoi(cell);
        else if constexpr (std::is_same_v<T, double>)
          result(i, n1, n2) = std::stod(cell);
        n2++;
      }
      n1++;
    }
    f.close();
  }
  return result;
}

template <typename T>
void write_csv(const std::vector<T> &data, const std::filesystem::path &path,
               int N0, int N1) {
  std::ofstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + path.string());
  }
  for (int i = 0; i < N0; i++) {
    for (int j = 0; j < N1; j++) {
      file << data[i * N1 + j] << ",";
    }
    file << std::endl;
  }
  file.close();
}
} // namespace SIR_SBM