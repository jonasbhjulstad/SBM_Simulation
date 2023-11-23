import pandas
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

engine = create_engine("postgresql://postgres:postgres@localhost:5432")

def read_simulation(p_out_id, graph_id, sim_id):
    return pandas.read_sql("SELECT t, \"S\", \"I\", \"R\" FROM community_state_excitation WHERE (p_out, graph, simulation, community) = ({}, {}, {}, {})".format(p_out_id, graph_id, sim_id, 0), engine)


df_0 = read_simulation(0, 0, 0)
df_0.to_csv('test.csv')
plt.plot(df_0['t'], df_0['S'], label='S')
plt.plot(df_0['t'], df_0['I'], label='I')
plt.plot(df_0['t'], df_0['R'], label='R')
plt.show()