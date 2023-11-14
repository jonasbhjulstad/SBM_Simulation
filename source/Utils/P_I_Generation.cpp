#include <SBM_Simulation/Utils/P_I_Generation.hpp>
#include <orm/db.hpp>
#include <Sycl_Buffer_Routines/Random.hpp>
#include <SBM_Graph/Complete_Graph.hpp>
#include <Sycl_Buffer_Routines/Database.hpp>
namespace SBM_Simulation
{

sycl::buffer<float, 3> generate_p_Is_excitation(
    sycl::queue &q, const SBM_Database::Sim_Param &p,
    const QString &control_type) {
  QVector<Orm::WhereItem> const_indices = {
      {"p_out", p.p_out_id},
      {"graph", p.graph_id},
      {"Control_Type", control_type}};

    return generate_upsert_p_Is(q, p, "p_Is_excitation", control_type);
}

sycl::buffer<float, 3> generate_p_Is_validation(
    sycl::queue &q, const SBM_Database::Sim_Param &p,
    const QString &control_type, const QString& regression_type) {
  QVector<Orm::WhereItem> const_indices = {
      {"p_out", p.p_out_id},
      {"graph", p.graph_id},
      {"Control_Type", control_type},
      {"Regression_Type", regression_type}};
    std::cout << "Warning: Validation Buffers p_I read is not Implemented!\n";
    return generate_upsert_p_Is(q, p, "p_Is_validation", control_type);
}

    // std::vector<float> generate_floats(uint32_t N, float min, float max, uint32_t seed);

sycl::buffer<float, 3> generate_upsert_p_Is(sycl::queue& q, const SBM_Database::Sim_Param &p, const QString& table_name, const QString &control_type)
{
    size_t N_connections = 0;
    if (control_type == "Community")
            N_connections = p.N_connections;
    else if(control_type == "Uniform")
            N_connections = 1;
    else if(control_type == "Partition")
            N_connections = SBM_Graph::complete_graph_max_edges(p.N_communities);
    auto N = N_connections*p.Nt*p.N_sims;
    auto p_Is_vec = Buffer_Routines::generate_floats(N, p.p_I_min, p.p_I_max, p.seed);
    sycl::event event;
    auto p_Is_buf = sycl::buffer<float, 3>(p_Is_vec.data(), sycl::range<3>(p.N_sims, p.Nt, N_connections));
    const std::vector<std::string> column_id_names = {"simulation", "t", "connection"};
                    // table.primary({"p_out", "graph", "simulation", "t", "connection", "Control_Type"});

    const std::vector<std::string> const_id_names = {"p_out", "graph", "Control_Type"};
    // const QVector<QVariant>
    const std::vector<QVariant> const_id_values = {p.p_out_id, p.graph_id, control_type};
    Buffer_Routines::buffer_to_table<float>(q, p_Is_buf, table_name.toStdString(), column_id_names, "value", const_id_names, const_id_values);
    event.wait();
    return p_Is_buf;
}
}