from Simulation_Tables import *
import pandas
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
engine = create_engine("postgresql://postgres:postgres@localhost:5432")

p_out_id, graph_id, sim_id = 0, 0, 0
community = 0
connection = 0

df = read_community_state_single(p_out_id, graph_id, sim_id, community, engine)
df = read_connection_single("connection_events_excitation", p_out_id, graph_id, sim_id, connection, engine)
df = read_community_state(p_out_id, graph_id, sim_id, engine)
df = read_community_graph_state(p_out_id, graph_id, engine)
df = read_community_p_state(p_out_id, engine)
df = read_total_state(p_out_id, graph_id, sim_id, engine)
df = read_total_graph_state(p_out_id, graph_id, engine)
df = read_total_p_state(p_out_id, engine)
