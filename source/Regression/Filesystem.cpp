#include <SBM_Simulation/Regression/Filesystem.hpp>
#include <SBM_Database/Simulation/Size_Queries.hpp>
#include <fstream>
namespace SBM_Regression {


Mat openData(const std::string &fileToOpen) {
  using namespace Eigen;
  std::vector<float> matrixEntries;

  std::ifstream matrixDataFile(fileToOpen);

  std::string matrixRowString;

  std::string matrixEntry;

  int matrixRowNumber = 0;

  while (std::getline(matrixDataFile, matrixRowString)) {
    std::stringstream matrixRowStringStream(matrixRowString);

    while (std::getline(matrixRowStringStream, matrixEntry, ',')) {
      matrixEntries.push_back(stod(matrixEntry));
    }
    matrixRowNumber++;
  }

  return Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(
      matrixEntries.data(), matrixRowNumber,
      matrixEntries.size() / matrixRowNumber);
}

void readRowMajor(const char* fileToOpen, Mat& res, uint32_t N_entries_per_row) {
  using namespace Eigen;
  auto N_cols = res.cols()/N_entries_per_row;
  auto N_rows = res.rows();
  std::vector<float> matrixEntries;

  std::ifstream matrixDataFile(fileToOpen);

  std::string matrixRowString;

  std::string matrixEntry;

  int matrixRowNumber = 0;

  for(int row = 0; row < N_rows; row++) {
    std::getline(matrixDataFile, matrixRowString);
    std::stringstream matrixRowStringStream(matrixRowString);
    for(int col = 0; col < N_cols; col++) {
      for(int entry = 0; entry < N_entries_per_row; entry++) {
        std::getline(matrixRowStringStream, matrixEntry, ',');
        matrixEntries.push_back(stod(matrixEntry));
      }
    }
  }

}



Mat read_community_X_data(uint32_t p_out_id, uint32_t graph, const char* control_type, const char* fname)
{
  std::stringstream st; 
  st << "\\copy (select s as s_0, i as i_0, r as r_0 from community_state where (p_out, graph, \\\"Control_Type\\\") = (";
  st << p_out_id << ", " << graph << ", '" << control_type << "'";
  st <<  ") AND t < (select count(distinct(t)) from community_state)) order by simulation, community, t asc)";
  st << "TO '" << fname << "' CSV";
  std::system(("psql -d sbm_database -c \"" + st.str() + "\"").c_str());
  auto dims = SBM_Database::get_simulation_dimensions();
  Mat result(dims.N_sims*dims.Nt, dims.N_communities*3);
  readRowMajor(fname, result, 3);
  std::remove(fname);
  return result;
}
Mat read_community_Y_data(uint32_t p_out_id, uint32_t graph, const char* control_type, const char* fname)
{
  std::stringstream st; 
  st << "\\copy (select s as s_0, i as i_0, r as r_0 from community_state where (p_out, graph, \\\"Control_Type\\\") = (";
  st << p_out_id << ", " << graph << ", '" << control_type << "'";
  st <<  ") AND t > 0 order by simulation, t, community asc)";
  st << "TO '" << fname << "' CSV";
  auto dims = SBM_Database::get_simulation_dimensions();
  Mat result(dims.N_sims*dims.Nt, dims.N_communities*3);
  readRowMajor(fname, result, 3);
  std::remove(fname);
  return result;
}

Mat read_community_U_data(uint32_t p_out_id, uint32_t graph, const char* control_type, const char* fname)
{
  std::stringstream st; 
  st << "\\copy (select simulation, t, community, s as s_0, i as i_0, r as r_0 from \\\"p_Is\\\" where (p_out, graph, \\\"Control_Type\\\") = (";
  st << p_out_id << ", " << graph << ", '" << control_type << "'";
  st <<  ") order by simulation, t, connection asc)";
  st << "TO '" << fname << "' CSV";
  auto dims = SBM_Database::get_simulation_dimensions();
  Mat result(dims.N_sims*dims.Nt, dims.N_connections);
  readRowMajor(fname, result);
  std::remove(fname);
  return result;
}

Mat read_connection_infections(uint32_t p_out_id, uint32_t graph, const char* control_type, const char* fname)
{
  std::stringstream st; 
  st << "\\copy (select value from infection_events where (p_out, graph, \\\"Control_Type\\\") = (";
  st << p_out_id << ", " << graph << ", '" << control_type << "'";
  st <<  ") order by simulation, t, connection asc)";
  st << "TO '" << fname << "' CSV";
  auto dims = SBM_Database::get_simulation_dimensions();
  Mat result(dims.N_sims*dims.Nt, dims.N_connections);
  readRowMajor(fname, result);
  std::remove(fname);
  return result;
}
} // namespace SBM_Regression