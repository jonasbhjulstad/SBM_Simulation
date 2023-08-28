#ifndef SIR_TYPES_HPP
#define SIR_TYPES_HPP
#include <array>
#include <cstdint>
#include <string>
#include <vector>
typedef std::array<uint32_t, 3> State_t;

template <typename T>
struct Timeseries_t : public std::vector<std::vector<T>>
{
  Timeseries_t(uint32_t Nt, uint32_t N_columns)
      : std::vector<std::vector<T>>(Nt, std::vector<T>(N_columns, T{})) {}
};
template <typename T>
struct Simseries_t : public std::vector<Timeseries_t<T>>
{
  Simseries_t(uint32_t N_sims, uint32_t Nt, uint32_t N_columns)
      : std::vector<Timeseries_t<T>>(N_sims, Timeseries_t<T>(Nt, N_columns))
  {
  }
};
template <typename T>
struct Graphseries_t : public std::vector<Simseries_t<T>>
{
  Graphseries_t(uint32_t N_graphs, uint32_t N_sims, uint32_t Nt, uint32_t N_columns)
      : std::vector<Simseries_t<T>>(N_graphs, Simseries_t<T>(N_sims, Nt, N_columns))
  {
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
enum SIR_State : unsigned char
{
  SIR_INDIVIDUAL_S = 0,
  SIR_INDIVIDUAL_I = 1,
  SIR_INDIVIDUAL_R = 2
};

#endif
