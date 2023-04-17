import graph_tool.all as gt
from graph_tool.collection import data
g = data["football"]
state = gt.minimize_blockmodel_dl(g)

state.draw(pos=g.vp.pos, output="football-sbm-fit.png")