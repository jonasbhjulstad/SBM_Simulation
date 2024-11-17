import graph_tool.all as gt
import numpy as np
b = []
for i in range(10):
    b.extend([i]*100)
p = np.eye(10)*0.1
g = gt.generate_sbm(b, p)
state = gt.IsingGlauberState(g, beta=1.5/10)
g.vp.pos = gt.sfdp_layout(g)
win = None
for i in range(100):
    ret = state.iterate_sync(niter=100)
    win = gt.graph_draw(g, g.vp.pos, vertex_fill_color=state.get_state(),
                        window=win, return_window=True, main=False)
