// csv.hpp
//

#ifndef LZZ_SIR_SBM_LZZ_csv_hpp
#define LZZ_SIR_SBM_LZZ_csv_hpp
#line 3 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//csv.hpp"
#include <SIR_SBM/common.hpp>
#include <SIR_SBM/vector.hpp>
#include <filesystem>
#include <fstream>
#define LZZ_INLINE inline
#line 13 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//csv.hpp"
namespace SIR_SBM
{
#line 14 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//csv.hpp"
  template <typename T = int>
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//csv.hpp"
  Vec2D <T> read_csv (std::filesystem::path const & path, uint32_t N0, uint32_t N1);
}
#line 13 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//csv.hpp"
namespace SIR_SBM
{
#line 41 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//csv.hpp"
  template <typename T = int>
#line 42 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//csv.hpp"
  Vec3D <T> read_csv (std::filesystem::path const & file_prefix, int N0, int N1, int N2);
}
#line 13 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//csv.hpp"
namespace SIR_SBM
{
#line 74 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//csv.hpp"
  template <typename T>
#line 75 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//csv.hpp"
  void write_csv (std::vector <T> const & data, std::filesystem::path const & path, int N0, int N1);
}
#line 13 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//csv.hpp"
namespace SIR_SBM
{
#line 14 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//csv.hpp"
  template <typename T>
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//csv.hpp"
  Vec2D <T> read_csv (std::filesystem::path const & path, uint32_t N0, uint32_t N1)
#line 16 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//csv.hpp"
                                          {
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
}
#line 13 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//csv.hpp"
namespace SIR_SBM
{
#line 41 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//csv.hpp"
  template <typename T>
#line 42 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//csv.hpp"
  Vec3D <T> read_csv (std::filesystem::path const & file_prefix, int N0, int N1, int N2)
#line 43 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//csv.hpp"
                                                            {
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
}
#line 13 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//csv.hpp"
namespace SIR_SBM
{
#line 74 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//csv.hpp"
  template <typename T>
#line 75 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//csv.hpp"
  void write_csv (std::vector <T> const & data, std::filesystem::path const & path, int N0, int N1)
#line 76 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//csv.hpp"
                               {
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
}
#undef LZZ_INLINE
#endif
