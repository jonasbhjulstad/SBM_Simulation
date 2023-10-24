#ifndef SBM_SIMULATION_GRAPH_TYPES_HPP
#define SBM_SIMULATION_GRAPH_TYPES_HPP
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <vector>
#include <soci/soci.h>
struct Edge_t
{
  uint32_t from;
  uint32_t to;
  uint32_t weight;
  friend std::fstream &operator<<(std::fstream &os, const Edge_t &e);
  friend std::ofstream &operator<<(std::ofstream &os, const Edge_t &e);
  friend std::ostream &operator<<(std::ostream &os, const Edge_t &e);
  friend std::stringstream &operator<<(std::stringstream &os, const Edge_t &e);

  static std::vector<uint32_t> get_weights(const std::vector<Edge_t> &edges);
  static std::vector<uint32_t> get_from(const std::vector<Edge_t> &edges);
  static std::vector<uint32_t> get_to(const std::vector<Edge_t> &edges);
  static constexpr std::size_t N = 3;
  static std::array<std::string, N> data_names()
  {
    return {"from", "to", "weight"};
  }
  static std::array<std::string, N> data_types()
  {
    return {"INTEGER", "INTEGER", "INTEGER"};
  }

  typedef std::tuple<uint32_t, uint32_t, uint32_t> Serialized_Data_t;
  Serialized_Data_t serialize() const;

  private:
  std::string to_array_string() const;
  // static std::vector<uint32_t> get_from(const std::vector<Edge_t> &edges);
  // static std::vector<uint32_t> get_to(const std::vector<Edge_t> &edges);
};

namespace soci
{

  template <>
  struct type_conversion<Edge_t>
  {
    typedef std::string base_type;
    static void from_base(const std::string &s, soci::indicator ind, Edge_t &edge);

    static void to_base(const Edge_t &edge, std::string &s, soci::indicator &ind);

  };
}

#endif
