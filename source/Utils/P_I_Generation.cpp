#include <Dataframe/Dataframe.hpp>
#include <SBM_Database/Simulation/Simulation_Tables.hpp>
#include <SBM_Database/Utils/Filesystem.hpp>
#include <SBM_Graph/Complete_Graph.hpp>
#include <SBM_Simulation/Utils/P_I_Generation.hpp>
#include <Sycl_Buffer_Routines/Database.hpp>
#include <Sycl_Buffer_Routines/Random.hpp>
#include <orm/db.hpp>
namespace SBM_Simulation {

sycl::buffer<float, 3>
generate_p_Is_excitation(sycl::queue &q, const SBM_Database::Sim_Param &p,
                         const QString &control_type) {
  QVector<Orm::WhereItem> const_indices = {{"p_out", p.p_out_id},
                                           {"graph", p.graph_id},
                                           {"Control_Type", control_type}};

  return generate_insert_p_Is(q, p, "p_Is_excitation", control_type);
}

sycl::buffer<float, 3>
generate_p_Is_validation(sycl::queue &q, const SBM_Database::Sim_Param &p,
                         const QString &control_type,
                         const QString &regression_type) {
  QVector<Orm::WhereItem> const_indices = {
      {"p_out", p.p_out_id},
      {"graph", p.graph_id},
      {"Control_Type", control_type},
      {"Regression_Type", regression_type}};
  std::cout << "Warning: Validation Buffers p_I read is not Implemented!\n";
  return generate_insert_p_Is(q, p, "p_Is_validation", control_type);
}

// std::vector<float> generate_floats(uint32_t N, float min, float max, uint32_t
// seed);

std::vector<float> generate_insert_p_Is(const SBM_Database::Sim_Param &p,
                                        const QString &table_name,
                                        const QString &control_type,
                                        const QString &tmp_dir) {
  auto N = p.N_connections * p.Nt * p.N_sims;
  auto p_Is_vec =
      Buffer_Routines::generate_floats(N, p.p_I_min, p.p_I_max, p.seed);
  sycl::event event;
  // auto p_Is_buf = sycl::buffer<float, 3>(p_Is_vec.data(),
  // sycl::range<3>(p.N_sims, p.Nt, N_connections));
  const std::vector<std::string> column_id_names = {"simulation", "t",
                                                    "connection"};
  // table.primary({"p_out", "graph", "simulation", "t", "connection",
  // "Control_Type"});
  Dataframe::Dataframe_t<float, 3> df(p_Is_vec,
                                      {p.N_sims, p.Nt, p.N_connections});
  // std string to qstring
  auto filename = (tmp_dir + "p_Is_excitation.csv");
  auto std_filename = (tmp_dir + "p_Is_excitation.csv").toStdString();
  SBM_Database::connection_write(filename.toStdString(), p.p_out_id, p.graph_id,
                                 df, control_type, "", 0, p.Nt);
  SBM_Database::copy_file_to_table(filename, "p_Is_excitation");
  return p_Is_vec;
}
sycl::buffer<float, 3> generate_insert_p_Is(sycl::queue &q,
                                            const SBM_Database::Sim_Param &p,
                                            const QString &table_name,
                                            const QString &control_type,
                                            const QString &tmp_dir) {
  auto p_Is_vec = generate_insert_p_Is(p, table_name, control_type, tmp_dir);
  auto p_Is_buf = Buffer_Routines::construct_device_buffer<float, 3>(
      q, p_Is_vec, sycl::range<3>(p.N_sims, p.Nt, p.N_connections));
  return p_Is_buf;
}
} // namespace SBM_Simulation