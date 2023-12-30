
#include <Dataframe/Dataframe.hpp>
#include <SBM_Database/Graph/Graph_Tables.hpp>
#include <SBM_Database/Simulation/Simulation_Tables.hpp>
#include <SBM_Database/Simulation/Size_Queries.hpp>
#include <SBM_Simulation/Simulation/Sim_Infection_Sampling.hpp>
#include <SBM_Simulation/Utils/Math.hpp>
#include <SBM_Simulation/Utils/Validation.hpp>
#include <Static_RNG/Static_RNG.hpp>
#include <Sycl_Buffer_Routines/Buffer_Utils.hpp>
#include <Sycl_Buffer_Routines/Random.hpp>
#include <iostream>
#include <orm/db.hpp>
#include <random>
#define INF_MAX_SAMPLE_LIMIT 10000

namespace SBM_Simulation
{

  const std::vector<uint32_t> get_delta_I(uint32_t p_out_id, uint32_t graph_id,
                                          uint32_t sim_id, uint32_t community, uint32_t Nt,
                                          const QString &control_type,
                                          const QString &simulation_type)
  {

    auto get_community_state_traj = [&]()
    {
      auto query = Orm::DB::table("community_state")->select({"t", "i", "r"}).where(
          {{"p_out", p_out_id},
           {"graph", graph_id},
           {"simulation", sim_id},
           {"community", community},
           {"Control_Type", control_type},
           {"Simulation_Type", simulation_type}}).get();
      std::vector<uint32_t> Is(Nt + 1, 0);
      std::vector<uint32_t> Rs(Nt + 1, 0);

      while(query.next())
      {
        auto t = query.value(0).toUInt();
        auto i = query.value(1).toUInt();
        auto r = query.value(2).toUInt();
        Is[t] = i;
        Rs[t] = r;
        assert(t <= Nt);
      }
      return std::make_pair(Is, Rs);
    };
    auto [Is, Rs] = get_community_state_traj();

    std::vector<uint32_t> delta_I(Nt, 0);
    std::vector<uint32_t> delta_R(Nt, 0);
    for (int t = 0; t < Nt; t++)
    {
      delta_R[t] = Rs[t + 1] - Rs[t];
      delta_I[t] = Is[t + 1] - Is[t] + delta_R[t];
    }
    return delta_I;
  }

  Dataframe::Dataframe_t<SBM_Graph::Edge_t, 2>
  get_connection_events(uint32_t p_out_id, uint32_t graph_id, uint32_t sim_id, const QString &control_type,
                        const QString &regression_type = "")
  {
    auto dims = SBM_Database::get_simulation_dimensions();

    auto query = Orm::DB::unprepared("SELECT t, value, connection, \"Direction\" FROM connection_events WHERE (p_out, graph, simulation) = (" + QString::number(p_out_id) + ", " + QString::number(graph_id) + ", " + QString::number(sim_id) + ")  ORDER BY t ASC");

    Dataframe::Dataframe_t<SBM_Graph::Edge_t, 2> events(
        std::array<uint32_t, 2>({(uint32_t)dims.Nt, (uint32_t)dims.N_connections}));
    while (query.next())
    {
      auto t = query.value(0).toUInt();
      auto value = query.value(1).toUInt();
      auto connection = query.value(2).toUInt();
      QString direction = query.value(3).toString();
      if (direction == "to")
      {
        events[t][connection].to = value;
      }
      else
      {
        events[t][connection].from = value;
      }
    }
    return events;
  }

  std::vector<SBM_Graph::Edge_t>
  get_related_connections(size_t c_idx,
                          const std::vector<SBM_Graph::Weighted_Edge_t> &ccm)
  {
    std::vector<SBM_Graph::Edge_t> connection_indices(ccm.size());
    std::transform(ccm.begin(), ccm.end(), connection_indices.begin(),
                   [c_idx](auto e)
                   {
                     return SBM_Graph::Edge_t{e.from == c_idx, e.to == c_idx};
                   });
    return connection_indices;
  }
  Dataframe::Dataframe_t<SBM_Graph::Edge_t, 1>
  get_related_events(size_t c_idx,
                     const std::vector<SBM_Graph::Weighted_Edge_t> &ccm,
                     const std::vector<SBM_Graph::Edge_t> &events)
  {

    auto r_con = get_related_connections(c_idx, ccm);
    std::vector<uint32_t> r_con_events(r_con.size(), 0);
    std::vector<SBM_Graph::Edge_t> related_events = events;
    for (int i = 0; i < related_events.size(); i++)
    {
      related_events[i] = SBM_Graph::Edge_t{(r_con[i].from) ? events[i].from : 0,
                                            (r_con[i].to) ? events[i].to : 0};
    }
    return related_events;
  }
  Dataframe::Dataframe_t<SBM_Graph::Edge_t, 2>
  get_related_events(size_t c_idx,
                     const std::vector<SBM_Graph::Weighted_Edge_t> &ccm,
                     const Dataframe::Dataframe_t<SBM_Graph::Edge_t, 2> &events)
  {

    auto Nt = events.size();
    Dataframe::Dataframe_t<SBM_Graph::Edge_t, 2> result(std::array<uint32_t, 2>({
        (uint32_t)Nt,
    }));
    for (int t = 0; t < Nt; t++)
    {
      result.data[t] = get_related_events(c_idx, ccm, events[t]);
    }
    return result;
  }

  std::vector<SBM_Graph::Edge_t>
  get_ccm_weights(const std::vector<SBM_Graph::Weighted_Edge_t> &ccm)
  {
    std::vector<SBM_Graph::Edge_t> weights(ccm.size());
    std::transform(ccm.begin(), ccm.end(), weights.begin(), [](auto e)
                   { return SBM_Graph::Edge_t{e.weight, e.weight}; });
    return weights;
  }

  Dataframe::Dataframe_t<SBM_Graph::Edge_t, 1>
  sample_community_dI(const auto &related_events, const auto &related_weights,
                      auto N_samples, uint32_t seed)
  {
    using namespace SBM_Graph;
    auto rv_flat = Edge_t::flatten(related_events);
    if (std::all_of(rv_flat.begin(), rv_flat.end(),
                    [](auto rv_elem)
                    { return rv_elem == 0; }))
    {
      return std::vector<SBM_Graph::Edge_t>(related_events.size(), Edge_t{0, 0});
    }

    auto weights_flat = Edge_t::flatten(related_weights);
    auto sample_counts =
        Static_RNG::constrained_weight_sample<std::mt19937_64, float, uint32_t>(
            N_samples, weights_flat, rv_flat, seed);

    return Edge_t::to_edges(sample_counts);
  }

  Dataframe::Dataframe_t<SBM_Graph::Edge_t, 2> sample_community_infections(
      uint32_t p_out_id, uint32_t graph_id, uint32_t sim_id, uint32_t community, const Dataframe::Dataframe_t<SBM_Graph::Edge_t, 2> &connection_events,
      const QString &control_type, const QString &regression_type,
      const std::vector<SBM_Graph::Weighted_Edge_t> &ccm, uint32_t seed)
  {

    auto dims = SBM_Database::get_simulation_dimensions();
    auto dIs = get_delta_I(p_out_id, graph_id, sim_id, community, dims.Nt, control_type,
                           regression_type);

    auto related_events = get_related_events(community, ccm, connection_events);
    auto seeds = Static_RNG::generate_seeds(dIs.size(), seed);
    Dataframe::Dataframe_t<SBM_Graph::Edge_t, 2> community_infections(
        std::array<uint32_t, 2>({(uint32_t)dims.Nt, (uint32_t)dims.N_connections}));
    auto ccm_weights = get_ccm_weights(ccm);

    for (int t = 0; t < dIs.size(); t++)
    {
      auto community_dI =
          sample_community_dI(related_events[t], ccm_weights, dIs[t], seeds[t]);
      for (int i = 0; i < community_dI.size(); i++)
      {
        community_infections[t][i] = community_dI[i];
      }
    }
    return community_infections;
  }

  Dataframe::Dataframe_t<SBM_Graph::Edge_t, 2> sample_simulation_infections(
      uint32_t p_out_id, uint32_t graph_id, uint32_t sim_id,
      const QString &control_type, const QString &regression_type,
      const std::vector<SBM_Graph::Weighted_Edge_t> &ccm, uint32_t seed)
  {
    auto dims = SBM_Database::get_simulation_dimensions();
    Dataframe::Dataframe_t<SBM_Graph::Edge_t, 2> simulation_infections(
        std::array<uint32_t, 2>({(uint32_t)dims.Nt, (uint32_t)dims.N_communities}));
    auto seeds = Static_RNG::generate_seeds(dims.N_communities, seed);
    Dataframe::Dataframe_t<SBM_Graph::Edge_t, 2> community_infections(
        std::array<uint32_t, 2>({(uint32_t)dims.Nt, (uint32_t)dims.N_communities}));
    // connection_events
    auto connection_events = get_connection_events(p_out_id, graph_id, sim_id, control_type, regression_type);
    for (int c_idx = 0; c_idx < dims.N_communities; c_idx++)
    {
      community_infections = sample_community_infections(
          p_out_id, graph_id, sim_id, c_idx, connection_events, control_type, regression_type, ccm,
          seeds[c_idx]);
      for (int t = 0; t < dims.Nt; t++)
      {
        for (int c_idx = 0; c_idx < dims.N_connections; c_idx++)
        {
          simulation_infections.data[t][c_idx].from +=
              community_infections.data[t][c_idx].from;
          simulation_infections.data[t][c_idx].to +=
              community_infections.data[t][c_idx].to;
        }
      }
    }
    return simulation_infections;
  }

  void sample_graph_infections(uint32_t p_out_id, uint32_t graph_id,
                               const QString &control_type,
                               const QString &regression_type, uint32_t seed)
  {
    auto ccm = SBM_Database::ccm_read(p_out_id, graph_id);

auto dims = SBM_Database::get_simulation_dimensions();
    auto seeds = Static_RNG::generate_seeds(dims.N_sims, seed);
    Dataframe::Dataframe_t<SBM_Graph::Edge_t, 3> graph_infections(
        std::array<uint32_t, 3>(
            {(uint32_t)dims.N_sims, (uint32_t)dims.Nt, (uint32_t)dims.N_communities}));

    for (int sim_id = 0; sim_id < dims.N_sims; sim_id++)
    {
      graph_infections.data[sim_id] =
          sample_simulation_infections(p_out_id, graph_id, sim_id, control_type,
                                       regression_type, ccm, seeds[sim_id]);
    }
    SBM_Database::edge_to_table("infection_events", p_out_id, graph_id,
                                graph_infections, control_type, regression_type,
                                0, graph_infections.data[0].size());
  }

  void sample_all_infections(const QString &control_type,
                             const QString &regression_type, uint32_t seed)
  {
    auto N_graphs = SBM_Database::get_N_graphs("connection_events");
    auto Np = SBM_Database::get_N_p_out("connection_events");

    auto seeds = Static_RNG::generate_seeds(N_graphs * Np, seed);
    for (int p_out_id = 0; p_out_id < Np; p_out_id++)
    {
      for (int graph_id = 0; graph_id < Np; graph_id++)
      {
        sample_graph_infections(p_out_id, graph_id, control_type, regression_type,
                                seeds[p_out_id * N_graphs + graph_id]);
      }
    }
  }

}