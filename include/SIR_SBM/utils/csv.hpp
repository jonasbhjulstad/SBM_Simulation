#pragma once

#include <filesystem>
#include <fstream>
#include <vector>
#include <SIR_SBM/epidemiological/types.hpp>
namespace SIR_SBM {

std::vector<float> read_csv_flat(const std::filesystem::path &file_prefix,
                                 int N0, int N1, int N2);
void write_csv(const std::vector<uint32_t> &data,
               const std::filesystem::path &path, int N0, int N1);
void write_csv(const std::vector<uint32_t> &data,
               const std::filesystem::path &fname, int N0, int N1, int N2);
void write_contact_events(const std::vector<uint32_t> &contact_events,
                          const std::filesystem::path &dir, uint32_t N_sims,
                          uint32_t N_connections, uint32_t Nt);

void write_population_count(
    const std::vector<Population_Count> &population_count,
    const std::filesystem::path &dir, uint32_t N_sims, uint32_t N_partitions,
    uint32_t Nt);

} // namespace SIR_SBM