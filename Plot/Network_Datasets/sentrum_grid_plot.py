from igraph import *
import matplotlib.pyplot as plt
import geopandas as gpd
data_filename = '/home/man/Documents/Sycl_Graph/data/sentrum_dynamic/data.gml'

df = gpd.read_file(data_filename)

#print column names
fig = df.plot()
plt.show()
# Plot the graph
a = 1