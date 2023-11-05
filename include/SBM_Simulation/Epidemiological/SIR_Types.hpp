#ifndef SIR_TYPES_HPP
#define SIR_TYPES_HPP
#include <array>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
struct State_t : public std::array<uint32_t, 3>
{
  friend std::ofstream &operator<<(std::ofstream &os, const State_t &s)
  {
    os << State_t::to_string(s, true, true);
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
    os << State_t::to_string(s, true, true);
    return os;
  }

  // basic ostream
  friend std::ostream &operator<<(std::ostream &os, const State_t &s)
  {
    os << State_t::to_string(s, true, true);
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
    if (quotes)
    {
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
    if (quotes)
    {
      ss << "'";
    }
    return ss.str();
  }

  bool is_valid(uint32_t N_pop) const
  {
    return ((*this)[0] + (*this)[1] + (*this)[2]) <= N_pop;
  }
};

enum SIR_State : unsigned char
{
  SIR_INDIVIDUAL_S = 0,
  SIR_INDIVIDUAL_I = 1,
  SIR_INDIVIDUAL_R = 2
};

// std::ostream& operator<<(std::ostream & os, SIR_State & state)
// {
//   os << (int) state;
//   return os;
// }

#endif
