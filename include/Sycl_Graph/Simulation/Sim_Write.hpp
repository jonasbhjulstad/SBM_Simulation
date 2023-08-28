#ifndef SIM_WRITE_HPP
#define SIM_WRITE_HPP
#include <Sycl_Graph/Simulation/Sim_Write_impl.hpp>
#include <vector>
#include <cstdint>
#include <string>

extern template void write_timeseries(const Timeseries_t<uint32_t>&& data, std::fstream& f);
extern template void write_timeseries(const Timeseries_t<float>&& data, std::fstream& f);
// extern template void write_timeseries(const Timeseries_t<State_t>& data, std::fstream& f);

extern template void write_timeseries(const Simseries_t<uint32_t>&& data, const std::string& fname, bool append = false);
extern template void write_timeseries(const Simseries_t<State_t>&& data, const std::string& fname, bool append = false);
extern template void write_timeseries(const Simseries_t<float>&& data, const std::string& fname, bool append = false);

extern template void write_timeseries(const Graphseries_t<uint32_t>&& data, const std::string& base_dir, const std::string& fname, bool append = false);
extern template void write_timeseries(const Graphseries_t<State_t>&& data, const std::string& base_dir, const std::string& fname, bool append = false);
extern template void write_timeseries(const Graphseries_t<float>&& data, const std::string& base_dir, const std::string& fname, bool append = false);

#endif
