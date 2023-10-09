#ifndef SIM_WRITE_HPP
#define SIM_WRITE_HPP

#include <Sycl_Graph/Utils/Common.hpp>

#include <Sycl_Graph/Simulation/Sim_Types.hpp>
#include <Sycl_Graph/Dataframe/Dataframe.hpp>

void write_to_file(const std::vector<std::pair<uint32_t, uint32_t>>& ccm, const std::string& fname);
void ccms_to_file(const std::vector<std::vector<std::pair<uint32_t, uint32_t>>>& ccms, const std::string& output_dir);


#endif
