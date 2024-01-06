import sys
import os
SBM_Database_Path = "/home/deb/Documents/SBM_Simulation/submodules/SBM_Database/Python/Database/"
sys.path.append(SBM_Database_Path)
from Simulation_Tables import *
from Graph_Tables import *



def simulation_infection_sample(p_out, graph_id, sim_id):
    df = read_community_state(p_out, graph_id, sim_id)
    dR = df['R'].diff()
    dI = df['I'].diff() + dR


#   Dataframe::Dataframe_t<SBM_Graph::Edge_t, 2>
#   get_connection_events(uint32_t p_out_id, uint32_t graph_id, uint32_t sim_id, const QString &control_type,
#                         const QString &regression_type = "")
#   {
#     auto dims = SBM_Database::get_simulation_dimensions();

#     auto query = Orm::DB::unprepared("SELECT t, value, connection, \"Direction\" FROM connection_events WHERE (p_out, graph, simulation) = (" + QString::number(p_out_id) + ", " + QString::number(graph_id) + ", " + QString::number(sim_id) + ")  ORDER BY t ASC");

#     Dataframe::Dataframe_t<SBM_Graph::Edge_t, 2> events(
#         std::array<uint32_t, 2>({(uint32_t)dims.Nt, (uint32_t)dims.N_connections}));
#     while (query.next())
#     {
#       auto t = query.value(0).toUInt();
#       auto value = query.value(1).toUInt();
#       auto connection = query.value(2).toUInt();
#       QString direction = query.value(3).toString();
#       if (direction == "to")
#       {
#         events[t][connection].to = value;
#       }
#       else
#       {
#         events[t][connection].from = value;
#       }
#     }
#     return events;
#   }
def get_connection_events(p_out, graph_id, sim_id, control_type, regression_type):
    dims = get_simulation_dimensions()
    ccm_from, ccm_to = read_ccm(p_out, graph_id, sim_id)

    connections_from = pd.read_sql("SELECT t, value, connection FROM connection_events WHERE (p_out, graph, simulation, \"Direction\") = ({}, {}, {}, 'from')  ORDER BY t ASC".format(p_out, graph_id, sim_id), engine)
    connections_to = pd.read_sql("SELECT t, value, connection FROM connection_events WHERE (p_out, graph, simulation, \"Direction\") = ({}, {}, {}, 'to')  ORDER BY t ASC".format(p_out, graph_id, sim_id), engine)

    
    