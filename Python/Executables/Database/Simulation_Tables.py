import pandas as pd
def read_community_state_single(p_out_id, graph_id, sim_id, community, engine):
    return pd.read_sql("SELECT t, \"S\", \"I\", \"R\" FROM community_state_excitation WHERE (p_out, graph, simulation, community) = ({}, {}, {}, {})".format(p_out_id, graph_id, sim_id, community), engine)

def read_connection_single(table_name, p_out_id, graph_id, sim_id, connection, engine):
    return pd.read_sql("SELECT value FROM " + table_name + " WHERE (p_out, graph, simulation, connection) = ({}, {}, {}, {}) ORDER BY t asc".format(p_out_id, graph_id, sim_id, connection), engine)

def get_N_distinct(table, column, engine):
    with engine.connect() as con:
        rs = con.execute("SELECT COUNT(DISTINCT {}) FROM {}".format(column, table))
        return rs.fetchone()[0]

def read_community_state(p_out_id, graph_id, sim_id, engine):
    return pd.read_sql("SELECT t, community, \"S\", \"I\", \"R\" FROM community_state_excitation WHERE (p_out, graph, simulation) = ({}, {}, {}) ORDER BY t, community asc".format(p_out_id, graph_id, sim_id), engine)

def read_community_graph_state(p_out_id, graph_id, engine):
    return pd.read_sql("SELECT t, community, simulation, \"S\", \"I\", \"R\" FROM community_state_excitation WHERE (p_out, graph) = ({}, {}) ORDER BY t, community, simulation asc".format(p_out_id, graph_id), engine)

def read_community_p_state(p_out_id, engine):
    return pd.read_sql("SELECT t, community, simulation, graph, \"S\", \"I\", \"R\" FROM community_state_excitation WHERE (p_out) = ({}) ORDER BY t, community, simulation, graph asc".format(p_out_id), engine)

def read_total_state(p_out_id, graph_id, sim_id, engine):
    #sum over community
    return pd.read_sql("SELECT t, SUM(\"S\") AS S, SUM(\"I\") AS I, SUM(\"R\") AS R FROM community_state_excitation WHERE (p_out, graph, simulation) = ({},{},{}) GROUP BY t ORDER BY t asc".format(p_out_id, graph_id, sim_id), engine) 

def read_total_graph_state(p_out_id, graph_id, engine):
    return pd.read_sql("SELECT t, simulation, SUM(\"S\") AS S, SUM(\"I\") AS I, SUM(\"R\") AS R FROM community_state_excitation WHERE (p_out, graph) = ({},{}) GROUP BY t, simulation ORDER BY t, simulation asc".format(p_out_id, graph_id), engine)

def read_total_p_state(p_out_id, engine):
    return pd.read_sql("SELECT t, simulation, graph, SUM(\"S\") AS S, SUM(\"I\") AS I, SUM(\"R\") AS R FROM community_state_excitation WHERE (p_out) = ({}) GROUP BY t, simulation, graph ORDER BY t, simulation, graph asc".format(p_out_id), engine)

def read_connection(table_name, p_out_id, graph_id, sim_id, engine):
    return pd.read_sql("SELECT value FROM " + table_name + " WHERE (p_out, graph, simulation) = ({}, {}, {}) ORDER BY t asc".format(p_out_id, graph_id, sim_id), engine)

def read_connection_graph(table_name, p_out_id, graph_id, engine):
    return pd.read_sql("SELECT value, simulation FROM " + table_name + " WHERE (p_out, graph) = ({}, {}) ORDER BY t, simulation asc".format(p_out_id, graph_id), engine)

def read_connection_p(table_name, p_out_id, engine):
    return pd.read_sql("SELECT value, simulation, graph FROM " + table_name + " WHERE (p_out) = ({}) ORDER BY t, simulation, graph asc".format(p_out_id), engine)


        # uint32_t N_pop;
        # uint32_t p_out_id;
        # uint32_t graph_id;
        # uint32_t N_communities;
        # uint32_t N_connections;
        # uint32_t N_sims;
        # uint32_t Nt;
        # uint32_t Nt_alloc;
        # uint32_t seed;
        # float p_in;
        # float p_out;
        # float p_I_min;
        # float p_I_max;
        # float p_R = 0.1f;
        # float p_I0 = 0.1f;
        # float p_R0 = 0.0f;

def read_sim_param(p_out_id, graph_id, engine):
    return pd.read_sql("SELECT \"N_pop\", \"N_communities\", \"N_connections\", \"N_sims\", \"Nt\", seed, p_in, p_out, \"p_I_min\", \"p_I_max\", \"p_R\", \"p_I0\", \"p_R0\", p_out_id, graph_id FROM simulation_parameters WHERE (p_out_id, graph_id) = ({}, {})".format(p_out_id, graph_id), engine).iloc[0].to_dict()
