#ifndef SIR_TYPES_HPP
#define SIR_TYPES_HPP
#include <array>
#include <cstdint>
#include <fstream>
#include <soci/soci.h>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <SBM_Simulation/Graph/Graph_Types.hpp>
struct State_t : public std::array<uint32_t, 3>
{
  friend std::ofstream &operator<<(std::ofstream &os, const State_t &s)
  {
    os << State_t::to_string(s,true, true);
    return os;
  }

  // fstream
  friend std::fstream &operator<<(std::fstream &os, const State_t &s)
  {
    os << State_t::to_string(s, true, true);
    return os;
  }
  friend std::stringstream &operator<<(std::stringstream &os, const State_t &s)
  {
    os << State_t::to_string(s,true, true);
    return os;
  }

  //basic ostream
  friend std::ostream &operator<<(std::ostream &os, const State_t &s)
  {
    os << State_t::to_string(s,true, true);
    return os;
  }


  static State_t from_string(const std::string &str, bool brackets = true)
  {
    // comes on the form '{1,2,3}'
    std::string s = str;
    if (brackets)
    {
      s = str.substr(1, str.size() - 2);
    }
    std::stringstream ss(s);
    std::string token;
    State_t state;
    int i = 0;
    while (std::getline(ss, token, ','))
    {
      state[i] = std::stoi(token);
      i++;
    }
    return state;
  }

  static std::string to_string(const State_t &state, bool brackets = true, bool quotes = false)
  {
    std::stringstream ss;
    if(quotes){
      ss << "'";
    }
    if (brackets)
    {
      ss << "{";
    }
    ss << state[0] << "," << state[1] << "," << state[2];
    if (brackets)
    {
      ss << "}";
    }
    if(quotes){
      ss << "'";
    }
    return ss.str();
  }
};

namespace soci
{

  template <>
  struct type_conversion<State_t>
  {
    typedef std::string base_type;

    static void from_base(const std::string &s, soci::indicator ind, State_t &state);

    static void to_base(const State_t &state, std::string &s, soci::indicator &ind);
  };

  template <>
  struct type_conversion<Edge_t>
  {
    typedef std::string base_type;

    static void from_base(const std::string &s, soci::indicator ind, Edge_t &edge);

    static void to_base(const Edge_t &edge, std::string &s, soci::indicator &ind);
  };
} // namespace soci

struct Inf_Sample_Data_t
{
  uint32_t community_idx;
  uint32_t N_infected;
  uint32_t seed;
  std::vector<uint32_t> events;
  std::vector<uint32_t> indices;
  std::vector<uint32_t> weights;
};
enum SIR_State : unsigned char
{
  SIR_INDIVIDUAL_S = 0,
  SIR_INDIVIDUAL_I = 1,
  SIR_INDIVIDUAL_R = 2
};

#endif
