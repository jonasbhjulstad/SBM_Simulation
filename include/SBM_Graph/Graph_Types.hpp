#ifndef SBM_SIMULATION_GRAPH_TYPES_HPP
#define SBM_SIMULATION_GRAPH_TYPES_HPP
#include <SBM_Graph/Common.hpp>
template <typename T>
struct Edge;

struct Edge_t
{
  uint32_t from;
  uint32_t to;
  friend std::fstream &operator<<(std::fstream &os, const Edge_t &e);
  friend std::ofstream &operator<<(std::ofstream &os, const Edge_t &e);
  friend std::ostream &operator<<(std::ostream &os, const Edge_t &e);
  friend std::stringstream &operator<<(std::stringstream &os, const Edge_t &e);

  static std::vector<uint32_t> get_from(const std::vector<Edge_t> &edges);
  static std::vector<uint32_t> get_to(const std::vector<Edge_t> &edges);
  static constexpr std::size_t N = 3;

};


struct Weighted_Edge_t
{
  uint32_t from;
  uint32_t to;
  uint32_t weight;
  friend std::fstream &operator<<(std::fstream &os, const Edge_t &e);
  friend std::ofstream &operator<<(std::ofstream &os, const Edge_t &e);
  friend std::ostream &operator<<(std::ostream &os, const Edge_t &e);
  friend std::stringstream &operator<<(std::stringstream &os, const Edge_t &e);

  static std::vector<uint32_t> get_from(const std::vector<Weighted_Edge_t> &edges);

  static std::vector<uint32_t> get_to(const std::vector<Weighted_Edge_t> &edges);

  static std::vector<uint32_t> get_weights(const std::vector<Weighted_Edge_t> &edges);


  static constexpr std::size_t N = 3;
};


#endif
