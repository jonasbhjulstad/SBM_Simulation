
// Dataframe_t<State_t, 2> read_timeseries(pqxx::connection &con, int p_out, int sim_id, int graph_id, uint32_t N_cols)
// {
//   auto work = pqxx::work(con);

//   auto result = work.exec("SELECT state FROM timeseries WHERE p_out = " + std::to_string(p_out) + " AND graph = " + std::to_string(graph_id) + " AND sim = " + std::to_string(sim_id));
//   Dataframe_t<State_t, 2> timeseries(std::array<uint32_t, 2>{static_cast<uint32_t>(result.size()), N_cols});

//   auto result_idx = result[0];
//   State_t tmp_state;
//   for (int i = 0; i < result.size(); i++)
//   {
//     auto res = result[i]["state"].as_array();
//     auto entry = res.get_next();
//     for (int j = 0; j < N_cols; j++)
//     {
//       entry = res.get_next();
//       if (!is_string_entry(entry))
//       {
//         throw std::runtime_error("Expected more columns database timeseries, got to index: " + std::to_string(j) + " of " + std::to_string(N_cols));
//       }
//       for (int k = 0; k < 3; k++)
//       {
//         tmp_state[k] = std::stoi(std::get<1>(entry));
//         entry = res.get_next();
//       }

//       timeseries(i, j) = tmp_state;
//     }
//   }
//   return timeseries;
// }
// Dataframe_t<State_t, 3> read_simseries(pqxx::connection &con, int p_out, int graph_id, const std::vector<uint32_t> &sim_ids, uint32_t N_cols)
// {
//   auto N_sims = sim_ids.size();
//   Dataframe_t<State_t, 3> simseries(N_sims);
//   for (int sim_idx = 0; sim_idx < N_sims; sim_idx++)
//   {
//     simseries[sim_idx] = read_timeseries(con, p_out, sim_ids[sim_idx], graph_id, N_cols);
//   }
//   return simseries;
// }
// Dataframe_t<State_t, 4> read_graphseries(pqxx::connection &con, int p_out, const std::vector<uint32_t> &graph_ids, uint32_t N_sims, uint32_t N_cols)
// {
//   auto N_graphs = graph_ids.size();
//   Dataframe_t<State_t, 4> graphseries(N_graphs);
//   auto sim_ids = make_iota(N_sims);
//   for (int graph_idx = 0; graph_idx < N_graphs; graph_idx++)
//   {
//     graphseries[graph_idx] = read_simseries(con, p_out, graph_ids[graph_idx], sim_ids, N_cols);
//   }
//   return graphseries;
// }

void write_ccm(pqxx::connection &con, uint32_t p_out, Dataframe_t<Edge_t, 2> &ccm)
{
  auto work = pqxx::work(con);
  auto Ng = ccm.size();
  std::string insert_str = "INSERT INTO connection_community_map (p_out, graph, edge, \"from\", \"to\", weight) VALUES (" + std::to_string(p_out) + ", ";
  auto stream = pqxx::stream_to::table(work, {"connection_community_map"}, {"p_out", "graph", "edge", "\"from\"", "\"to\"", "weight"});
  for (int i = 0; i < Ng; i++)
  {
    auto N_edges = ccm[i].size();
    for (int vertex_id = 0; vertex_id < N_edges; vertex_id++)
    {
      auto edge = ccm(i, vertex_id);
      stream << std::make_tuple(p_out, i, vertex_id, edge.from, edge.to, edge.weight);
    }
  }
  stream.complete();
  work.commit();

}

void write_vcm(pqxx::connection &con, uint32_t p_out, const Dataframe_t<uint32_t, 2> &vcm)
{
  auto work = pqxx::work(con);
  auto Ng = vcm.size();
  auto N_vertices = vcm[0].size();
  auto stream = pqxx::stream_to::table(work, {"vertex_community_map"}, {"p_out", "graph", "vertex", "community"});
  for (int i = 0; i < Ng; i++)
  {
    for (int vertex_id = 0; vertex_id < N_vertices; vertex_id++)
    {

      stream << std::make_tuple(p_out, i, vertex_id, vcm(i, vertex_id));
    }
  }
  stream.complete();
  work.commit();
}

void write_edgelist(pqxx::connection &con, uint32_t p_out, const Dataframe_t<std::pair<uint32_t, uint32_t>, 2> &edgelists)
{
  auto work = pqxx::work(con);
  auto Ng = edgelists.size();
  auto stream = pqxx::stream_to::table(work, {"edgelists"}, {"p_out", "graph", "edge", "from", "to"});

  for (int i = 0; i < Ng; i++)
  {
    auto N_edges = edgelists[i].size();
    for (int edge_id = 0; edge_id < N_edges; edge_id++)
    {
      stream << std::make_tuple(p_out, i, edge_id, edgelists(i, edge_id).first, edgelists(i, edge_id).second);
    }
  }
  stream.complete();
  work.commit();
}
