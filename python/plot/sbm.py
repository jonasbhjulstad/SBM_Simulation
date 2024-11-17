import graph_tool.all as gt
import numpy as np
b = []
N = 100
for i in range(10):
    b.extend([i]*N)

p_in = 1.0
p_out = 0.1
p = np.ones((10,10))*p_out*(0.25)*(N*N)
for i in range(10):
    p[i,i] = p_in*(0.25)*(N*N)

g = gt.generate_sbm(b, p)
partition_map_0 = g.new_vertex_property("int")
partition_map_1 = g.new_vertex_property("int")

for i in range(10):
    partition_map_1[g.vertex(i)] = i
for i, p in enumerate(b):
    partition_map_0[g.vertex(i)] = p

state = gt.BlockState(g, b=partition_map_0)
# pos = gt.sfdp_layout(g, C=1.0, groups=partition_map)
# state.draw(vertex_fill_color=state.get_state(), pos=pos)
# state.
nstate = gt.NestedBlockState(g, bs=[partition_map_0, partition_map_1])
# gt.draw_hierarchy(nstate)
gt.draw_hierarchy(nstate, layout="sfdp", vertex_fill_color=state.get_state(), )