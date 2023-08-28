#include <Sycl_Graph/Simulation/Sim_Write_Impl.hpp>
#include <algorithm>
#include <array>
#include <fstream>
template void write_timeseries(const Timeseries_t<uint32_t>&& ts, std::fstream& f);
template void write_timeseries(const Timeseries_t<float>&& ts, std::fstream& f);
// extern template void write_timeseries(const Timeseries_t<State_t>&& ts, std::fstream& f);

template void write_simseries(const Simseries_t<uint32_t>&& ss, const std::string& abs_fname, bool append);
template void write_simseries(const Simseries_t<float>&& ss, const std::string& abs_fname, bool append);
template void write_simseries(const Simseries_t<State_t>&& ss, const std::string& abs_fname, bool append);

template void write_graphseries(const Graphseries_t<uint32_t>&& gs, const std::string& base_dir, const std::string& base_fname, bool append);
template void write_graphseries(const Graphseries_t<float>&& gs, const std::string& base_dir, const std::string& base_fname, bool append);
template void write_graphseries(const Graphseries_t<State_t>&& gs, const std::string& base_dir, const std::string& base_fname, bool append);
