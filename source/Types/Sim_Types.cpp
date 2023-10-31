#include <SBM_Simulation/Types/Sim_Types.hpp>
namespace SBM_Simulation {
QJsonObject Sim_Param::to_json() const {

  QJsonObject json({{"N_pop", static_cast<int>(N_pop)},
                    {"graph_id", static_cast<int>(graph_id)},
                    {"N_communities", static_cast<int>(N_communities)},
                    {"p_in", p_in},
                    {"p_out_id", static_cast<int>(p_out_id)},
                    {"p_out", p_out},
                    {"N_sims", static_cast<int>(N_sims)},
                    {"Nt", static_cast<int>(Nt)},
                    {"Nt_alloc", static_cast<int>(Nt_alloc)},
                    {"seed", static_cast<int>(seed)},
                    {"p_I_min", p_I_min},
                    {"p_I_max", p_I_max},
                    {"p_R", p_R},
                    {"p_I0", p_I0},
                    {"p_R0", p_R0}});

  return json;
}

Sim_Param Sim_Param::from_json(QJsonObject json) {
  auto to_std_vector = [](const auto &json_arr) {
    auto json_vec =
        json_arr["N_communities"].toArray().toVariantList().toVector();
    return std::vector<uint32_t>(json_vec.begin(), json_vec.end());
  };

  // uint32_t N_pop;
  // uint32_t p_out_id;
  // uint32_t graph_id;
  // uint32_t N_communities;
  // uint32_t N_connections;
  // float p_in;
  // float p_out;
  // uint32_t N_sims;
  // uint32_t Nt;
  // uint32_t Nt_alloc;
  // uint32_t seed;
  // float p_I_min;
  // float p_I_max;
  // float p_R = 0.1f;
  // float p_I0 = 0.1f;
  // float p_R0 = 0.0f;

  auto to_float = [](auto num) { return num.toVariant().toFloat(); };
  auto to_uint = [](auto num) { return num.toVariant().toUInt(); };

  Sim_Param result{to_uint(json["N_pop"]),
                   to_uint(json["p_out_id"]),
                   to_uint(json["graph_id"]),
                   to_uint(json["N_communities"]),
                   to_uint(json["N_connections"]),
                   to_uint(json["N_sims"]),
                   to_uint(json["Nt"]),
                   to_uint(json["Nt_alloc"]),
                   to_uint(json["seed"]),
                   to_float(json["p_in"]),
                   to_float(json["p_out"]),
                   to_float(json["p_I_min"]),
                   to_float(json["p_I_max"]),
                   to_float(json["p_R"]),
                   to_float(json["p_I0"]),
                   to_float(json["p_R0"])};
  return result;
}
} // namespace SBM_Simulation
