#include <SIR_SBM/epidemiological/infection_count.hpp>
#include <SIR_SBM/graph/indices.hpp>
#include <SIR_SBM/utils/csv.hpp>
#include <fstream>

namespace SIR_SBM {

std::vector<float> read_csv_flat(const std::filesystem::path &file_prefix,
                                 int N0, int N1, int N2) {
  std::ifstream f;
  // std::vector<int> result(N0 * N1 * N2);
  auto result = std::vector<float>(N0 * N1 * N2);

  for (int i = 0; i < N0; i++) {
    std::string filename = file_prefix.string() + std::to_string(i) + ".csv";
    f.open(filename);
    if (!f.is_open()) {
      throw std::runtime_error("Could not open file: " + filename);
    }
    std::string line;
    int n1 = 0;
    while (std::getline(f, line)) {
      std::stringstream ss(line);
      std::string cell;
      int n2 = 0;
      while (std::getline(ss, cell, ',')) {
        result[i * N1 * N2 + n1 * N2 + n2] = std::stof(cell);
        n2++;
      }
      n1++;
    }
    f.close();
  }
  return result;
}
void write_csv(const std::vector<uint32_t> &data,
               const std::filesystem::path &path, int N0, int N1) {
  std::ofstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + path.string());
  }
  for (int i = 0; i < N0; i++) {
    for (int j = 0; j < N1; j++) {
      file << data[i * N1 + j] << ",";
    }
    file << std::endl;
  }
  file.close();
}

void write_csv(const std::vector<uint32_t> &data,
               const std::filesystem::path &fname, int N0, int N1, int N2) {
  std::ofstream file;
  for (int n0 = 0; n0 < N0; n0++) {
    std::filesystem::path p = fname;
    p += "_" + std::to_string(n0) + ".csv";
    file.open(p);
    for (int n1 = 0; n1 < N1; n1++) {
      for (int n2 = 0; n2 < N2; n2++) {
        file << data[n0 * N1 * N2 + n1 * N2 + n2] << ",";
      }
      file << "\n";
    }
    file.close();
  }
}
void write_contact_events(const std::vector<uint32_t> &contact_events,
                          const std::filesystem::path &fname, uint32_t N_sims,
                          uint32_t N_connections, uint32_t Nt) {
  std::ofstream f;
  std::filesystem::create_directories(fname.parent_path());
  for (int sim_idx = 0; sim_idx < N_sims; sim_idx++) {
    std::filesystem::path p = fname;
    p += "_" + std::to_string(sim_idx) + ".csv";
    f.open(p);
    for (int t_idx = 0; t_idx < Nt; t_idx++) {
      for (int c_idx = 0; c_idx < N_connections; c_idx++) {
        f << contact_events[get_from_connection_idx(sim_idx, c_idx, t_idx,
                                                    N_connections, Nt)]
          << ","
          << contact_events[get_to_connection_idx(sim_idx, c_idx, t_idx,
                                                  N_connections, Nt)]
          << ",";
      }
      f << std::endl;
    }
    f.close();
  }
}

void write_population_count(
    const std::vector<Population_Count> &population_count,
    const std::filesystem::path &fname, uint32_t N_sims, uint32_t N_partitions,
    uint32_t Nt) {
  std::ofstream f;
  std::filesystem::create_directories(fname.parent_path());
  uint32_t idx;
  Population_Count pc;
  for (int sim_idx = 0; sim_idx < N_sims; sim_idx++) {

    std::filesystem::path p = fname;
    p += "_" + std::to_string(sim_idx) + ".csv";
    f.open(p);
    for (int t_idx = 0; t_idx < Nt; t_idx++) {
      for (int p_idx = 0; p_idx < N_partitions; p_idx++) {
        pc = population_count[get_partition_idx(sim_idx, p_idx, t_idx,
                                                N_partitions, Nt)];
        f << pc.S << "," << pc.I << "," << pc.R << ",";
      }
      f << std::endl;
    }
    f.close();
  }
}
void write_partition_infections(const std::vector<Population_Count> &population_count,
                          const std::filesystem::path &fname, uint32_t N_sims,
                          uint32_t N_partitions, uint32_t Nt) {
  std::ofstream f;
  std::filesystem::create_directories(fname.parent_path());
  uint32_t idx;
  for (int sim_idx = 0; sim_idx < N_sims; sim_idx++) {

    std::filesystem::path p = fname;
    p += "_" + std::to_string(sim_idx) + ".csv";
    f.open(p);
    for (int t_idx = 0; t_idx < Nt; t_idx++) {
      for (int p_idx = 0; p_idx < N_partitions; p_idx++) {
        f << get_new_infections(population_count, sim_idx, p_idx, N_partitions,
                                t_idx, Nt + 1);
        f << ",";
      }
      f << std::endl;
    }
    f.close();
  }
}


void write_partition_contacts(const std::vector<uint32_t> &contact_events,
                          const std::filesystem::path &fname, uint32_t N_sims,
                          uint32_t N_connections, uint32_t Nt) {
  std::ofstream f;
  std::filesystem::create_directories(fname.parent_path());
  uint32_t idx;
  for (int sim_idx = 0; sim_idx < N_sims; sim_idx++) {

    std::filesystem::path p = fname;
    p += "_" + std::to_string(sim_idx) + ".csv";
    f.open(p);
    for (int t_idx = 0; t_idx < Nt; t_idx++) {
      for (int c_idx = 0; c_idx < N_connections; c_idx++) {
        f << ",";
      }
      f << std::endl;
    }
    f.close();
  }
}


} // namespace SIR_SBM