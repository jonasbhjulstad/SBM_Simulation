#pragma once
#include <cstdint>
#include <filesystem>
#include <vector>
namespace SIR_SBM {
struct Connection {
  uint32_t to, from;
  Connection() = default;
  Connection(uint32_t to, uint32_t from): to(to), from(from) {}
};
struct Connection_Data : public std::vector<uint32_t> {
  uint32_t N_sims;
  uint32_t N_connections;
  uint32_t Nt;
  Connection_Data(uint32_t N_sims, uint32_t N_connections, uint32_t Nt);
  Connection_Data(const std::vector<uint32_t> &data, uint32_t N_sims,
                  uint32_t N_connections, uint32_t Nt);
  Connection operator()(uint32_t sim_idx, uint32_t connection_idx, uint32_t t) const;
  void write(const std::filesystem::path &fname) const;
  std::tuple<uint32_t&, uint32_t&> ref_connection(uint32_t sim_idx, uint32_t connection_idx, uint32_t t) const;
  uint32_t get_connection_idx(uint32_t sim_idx, uint32_t connection_idx,
                              uint32_t t, bool reverse = false) const;
};
} // namespace SIR_SBM