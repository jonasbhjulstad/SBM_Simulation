#pragma once
#include <array>

namespace SIR_SBM {
enum class SIR_State : char {
  Susceptible = 0,
  Infected = 1,
  Recovered = 2,
  Invalid = 3
};

struct Population_Count {
  int S, I, R;
  Population_Count();
  Population_Count(int S, int I, int R);
  Population_Count(const std::array<int, 3> &arr);
  Population_Count operator+(const Population_Count &other) const;
  bool is_zero() const;
  int &operator[](SIR_State s);
};
} // namespace SIR_SBM