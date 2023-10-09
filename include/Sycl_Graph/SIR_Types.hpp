#ifndef SIR_TYPES_HPP
#define SIR_TYPES_HPP
#include <array>
#include <vector>
#include <cstdint>
#include <string>
#include <fstream>
#include <sstream>

struct State_t: public std::array<uint32_t, 3>
{
  friend std::ofstream& operator<<(std::ofstream& os, const State_t& s)
  {
    os << s[0] << "," << s[1] << "," << s[2];
    return os;
  }

  //fstream
  friend std::fstream& operator<<(std::fstream& os, const State_t& s)
  {
    os << s[0] << "," << s[1] << "," << s[2];
    return os;
  }
  friend std::stringstream& operator<<(std::stringstream& os, const State_t& s)
  {
    os << s[0] << "," << s[1] << "," << s[2];
    return os;
  }
};

struct Inf_Sample_Data_t
{
    uint32_t community_idx;
    uint32_t N_infected;
    uint32_t seed;
    std::vector<uint32_t> events;
    std::vector<uint32_t> indices;
    std::vector<uint32_t> weights;
};
enum SIR_State : unsigned char {
  SIR_INDIVIDUAL_S = 0,
  SIR_INDIVIDUAL_I = 1,
  SIR_INDIVIDUAL_R = 2
};

#endif
