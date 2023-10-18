#ifndef SIR_TYPES_HPP
#define SIR_TYPES_HPP
#include <array>
#include <cstdint>
#include <fstream>
#include <soci/soci.h>
#include <sstream>
#include <string>
#include <vector>
struct State_t : public std::array<uint32_t, 3>
{
  friend std::ofstream &operator<<(std::ofstream &os, const State_t &s)
  {
    os << s[0] << "," << s[1] << "," << s[2];
    return os;
  }

  // fstream
  friend std::fstream &operator<<(std::fstream &os, const State_t &s)
  {
    os << s[0] << "," << s[1] << "," << s[2];
    return os;
  }
  friend std::stringstream &operator<<(std::stringstream &os, const State_t &s)
  {
    os << s[0] << "," << s[1] << "," << s[2];
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

  static std::string to_string(const State_t &state, bool brackets = true)
  {
    std::stringstream ss;
    if (brackets)
    {
      ss << "{";
    }
    ss << state[0] << "," << state[1] << "," << state[2];
    if (brackets)
    {
      ss << "}";
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

    static void from_base(const std::string &s, soci::indicator ind, State_t &state)
    {
      if (ind == i_null)
      {
        throw soci_error("Null value not allowed for this type");
      }
      state = State_t::from_string(s);
    }

    static void to_base(const State_t &state, std::string &s, soci::indicator &ind)
    {
      s = State_t::to_string(state);
      ind = i_ok;
    }
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
