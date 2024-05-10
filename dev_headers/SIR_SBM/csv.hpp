#pragma once
#hdr
#include <SIR_SBM/common.hpp>
#include <filesystem>
#include <fstream>
#end

namespace SIR_SBM {
Vec2D<int> read_csv(const std::filesystem::path &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + path.string());
  }
  Vec2D<int> result;
  std::string line;
  while (std::getline(file, line)) {
    std::vector<int> row;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
      row.push_back(std::stoi(cell));
    }
    result.push_back(row);
  }
  return result;
}

std::vector<int> read_csv(const std::filesystem::path &file_prefix, int N0, int N1, int N2) {
  std::ifstream f;
  std::vector<int> result(N0 * N1 * N2);
  for(int n0 = 0; n0 < N0; n0++)
  {
    std::filesystem::path file = file_prefix;
    file += std::to_string(n0) + ".csv";
    f.open(file);
    if (!f.is_open()) {
      throw std::runtime_error("Could not open file: " + file_prefix.string());
    }
    std::string line;
    int n1 = 0;
    while (std::getline(f, line)) {
      std::stringstream ss(line);
      std::string cell;
      while (std::getline(ss, cell, ',')) {
        result[n0 * N1 * N2 + n1 * N2] = std::stoi(cell);
        n1++;
      }
    }
  }
  return result; 
}


template <typename T>
void write_csv(const std::vector<T> &data, const std::filesystem::path &path, int N0, int N1) {
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