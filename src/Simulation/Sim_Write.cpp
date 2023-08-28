#include <Sycl_Graph/Simulation/Sim_Write.hpp>


template void write_timeseries(const std::vector<std::vector<uint32_t>>& data, std::ofstream& f);
template void write_timeseries(const std::vector<std::vector<float>>& data, std::ofstream& f);
// template void write_timeseries(const std::vector<std::vector<State_t>>& data, std::ofstream& f);

template void write_timeseries(const std::vector<std::vector<std::vector<uint32_t>>>& data, const std::string& fname, bool append = false);
template void write_timeseries(const std::vector<std::vector<std::vector<float>>>& data, const std::string& fname, bool append = false);
template void write_timeseries(const std::vector<std::vector<std::vector<State_t>>>& data, const std::string& fname, bool append = false);

template void write_timeseries(const std::vector<std::vector<std::vector<std::vector<uint32_t>>>>& data, const std::string& base_dir, const std::string& fname, bool append = false);
template void write_timeseries(const std::vector<std::vector<std::vector<std::vector<float>>>>& data, const std::string& base_dir, const std::string& fname, bool append = false);
template void write_timeseries(const std::vector<std::vector<std::vector<std::vector<State_t>>>>& data, const std::string& base_dir, const std::string& fname, bool append = false);
