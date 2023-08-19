#ifndef SIM_WRITE_HPP
#define SIM_WRITE_HPP

#include <vector>
#include <cstdint>
#include <string>
auto merge(const std::vector<std::vector<std::vector<uint32_t>>>& t0, const std::vector<std::vector<std::vector<uint32_t>>>& t1);

void events_to_file(const std::vector<std::vector<std::vector<uint32_t>>>& e_to, const std::vector<std::vector<std::vector<uint32_t>>>& e_from, const std::string& abs_fname);

void timeseries_to_file(const std::vector<std::vector<std::vector<std::array<uint32_t, 3>>>>& ts, const std::string& abs_fname);
void timeseries_to_file(const std::vector<std::vector<std::vector<uint32_t>>>& ts, const std::string& abs_fname);


#endif
