#ifndef SIM_WRITE_HPP
#define SIM_WRITE_HPP
#include <Sycl_Graph/Simulation/Sim_Write_impl.hpp>
#include <vector>
#include <cstdint>
#include <string>

extern template void write_timeseries(const std::vector<std::vector<uint32_t>>& data, std::ofstream& f);
extern template void write_timeseries(const std::vector<std::vector<float>>& data, std::ofstream& f);
extern template void write_timeseries(const std::vector<std::vector<State_t>>& data, std::ofstream& f);

extern template void write_timeseries(const std::vector<std::vector<std::vector<uint32_t>>>& data, const std::string& fname, bool append = false);
extern template void write_timeseries(const std::vector<std::vector<std::vector<float>>>& data, const std::string& fname, bool append = false);
extern template void write_timeseries(const std::vector<std::vector<std::vector<State_t>>>& data, const std::string& fname, bool append = false);

extern template void write_timeseries(const std::vector<std::vector<std::vector<std::vector<uint32_t>>>>& data, const std::string& base_dir, const std::string& fname, bool append = false);
extern template void write_timeseries(const std::vector<std::vector<std::vector<std::vector<float>>>>& data, const std::string& base_dir, const std::string& fname, bool append = false);
extern template void write_timeseries(const std::vector<std::vector<std::vector<std::vector<State_t>>>>& data, const std::string& base_dir, const std::string& fname, bool append = false);

#endif
