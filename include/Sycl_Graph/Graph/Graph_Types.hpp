#ifndef SYCL_GRAPH_GRAPH_TYPES_HPP
#define SYCL_GRAPH_GRAPH_TYPES_HPP
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <vector>
struct Edge_t
{
  uint32_t from;
  uint32_t to;
  uint32_t weight;
  friend std::fstream &operator<<(std::fstream &os, const Edge_t &e);
  friend std::ofstream &operator<<(std::ofstream &os, const Edge_t &e);

  static std::vector<uint32_t> get_weights(const std::vector<Edge_t> &edges);
  static std::vector<uint32_t> get_from(const std::vector<Edge_t> &edges);
  static std::vector<uint32_t> get_to(const std::vector<Edge_t> &edges);
  static std::vector<uint32_t> get_from(const std::vector<std::pair<uint32_t, uint32_t>> &edges);
  static std::vector<uint32_t> get_to(const std::vector<std::pair<uint32_t, uint32_t>> &edges);
};

#endif
