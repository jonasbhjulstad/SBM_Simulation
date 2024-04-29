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
} // namespace SIR_SBM