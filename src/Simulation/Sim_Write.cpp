#include <Sycl_Graph/Simulation/Sim_Write.hpp>


template void write_timeseries(const Timeseries_t<uint32_t>&& data, std::fstream& f);
template void write_timeseries(const Timeseries_t<float>&& data, std::fstream& f);
// template void write_timeseries(const std::vector<std::vector<State_t>>& data, std::fstream& f);

template void write_timeseries(const Simseries_t<uint32_t>&& data, const std::string& fname, bool append = false);
template void write_timeseries(const Simseries_t<State_t>&& data, const std::string& fname, bool append = false);
template void write_timeseries(const Simseries_t<float>&& data, const std::string& fname, bool append = false);

template void write_timeseries(const Graphseries_t<uint32_t>&& data, const std::string& base_dir, const std::string& fname, bool append = false);
template void write_timeseries(const Graphseries_t<State_t>&& data, const std::string& base_dir, const std::string& fname, bool append = false);
template void write_timeseries(const Graphseries_t<float>&& data, const std::string& base_dir, const std::string& fname, bool append = false);
