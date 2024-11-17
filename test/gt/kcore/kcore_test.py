import graph_tool.all as gt
from kcore import kcore_decomposition

g = gt.collection.data["football"]

c = kcore_decomposition(g)

print(c.a)