#pragma once
#include <SBM_Simulation/Regression/Regression_Types.hpp>
#include <string>
namespace SBM_Regression
{

  Mat openData(const std::string &fileToOpen);
  void readRowMajor(const char *fileToOpen, Mat &res, uint32_t N_entries_per_row = 1);

  Mat read_community_X_data(uint32_t p_out_id, uint32_t graph, const char *control_type, const char *fname = "community_X_data.csv");
  Mat read_community_Y_data(uint32_t p_out_id, uint32_t graph, const char *control_type, const char *fname = "community_Y_data.csv");

  Mat read_community_U_data(uint32_t p_out_id, uint32_t graph, const char *control_type, const char *fname = "community_U_data.csv");
  Mat read_connection_infections(uint32_t p_out_id, uint32_t graph, const char *control_type, const char *fname = "connection_infections.csv");

}