#include <SIR_SBM/graph/indices.hpp>
#include <SIR_SBM/simulation/connection_data.hpp>
#include <fstream>

namespace SIR_SBM {
Connection_Data::Connection_Data(uint32_t N_sims, uint32_t N_connections,
                                 uint32_t Nt)
    : std::vector<uint32_t>(N_sims * 2 * N_connections * Nt), N_sims(N_sims),
      N_connections(N_connections), Nt(Nt) {}
Connection_Data::Connection_Data(const std::vector<uint32_t> &data,
                                 uint32_t N_sims, uint32_t N_connections,
                                 uint32_t Nt)
    : std::vector<uint32_t>(data), N_sims(N_sims), N_connections(N_connections),
      Nt(Nt) {}
uint32_t Connection_Data::get_connection_idx(uint32_t sim_idx,
                                             uint32_t connection_idx,
                                             uint32_t t, bool reverse) const {
  if (reverse)
    return sim_idx * N_connections * Nt + (2 * connection_idx + 1) * Nt + t;
  else
    return sim_idx * N_connections * Nt + 2 * connection_idx * Nt + t;
}
void Connection_Data::plus(uint32_t sim_idx, uint32_t connection_idx,
                           uint32_t t,
                           const std::pair<uint32_t, uint32_t> elem) {
  (*this)[get_connection_idx(sim_idx, connection_idx, t)] += elem.first;
  (*this)[get_connection_idx(sim_idx, connection_idx, t, true)] += elem.second;
}
Connection Connection_Data::operator()(uint32_t sim_idx,
                                       uint32_t connection_idx,
                                       uint32_t t) const {
  return Connection(
      this->operator[](get_connection_idx(sim_idx, connection_idx, t)),
      this->operator[](get_connection_idx(sim_idx, connection_idx, t, true)));
}
void Connection_Data::write(const std::filesystem::path &fname) const {
  std::ofstream f;
  std::filesystem::create_directories(fname.parent_path());
  for (int sim_idx = 0; sim_idx < N_sims; sim_idx++) {
    std::filesystem::path p = fname;
    p += "_" + std::to_string(sim_idx) + ".csv";
    f.open(p);
    for (int t_idx = 0; t_idx < Nt; t_idx++) {
      for (int c_idx = 0; c_idx < N_connections; c_idx++) {
        Connection c = this->operator()(sim_idx, c_idx, t_idx);
        f << c.to << "," << c.from << ",";
      }
      f << std::endl;
    }
    f.close();
  }
}

std::vector<uint32_t> Connection_Data::get_connections_t(uint32_t sim_idx,
                                                         uint32_t t) const {
  std::vector<uint32_t> result(2 * N_connections);
  for (int c_idx = 0; c_idx < N_connections; c_idx++) {
    Connection c = contact_events(sim_idx, c_idx, t);
    result[2 * c_idx] = c.to;
    result[2 * c_idx + 1] = c.from;
  }
  return result;
}
} // namespace SIR_SBM