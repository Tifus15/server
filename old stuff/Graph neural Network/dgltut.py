import dgl
import numpy as numpy
import torch
import networkx as nx
import matplotlib.pyplot as plt
g = dgl.graph(([0,0,0,1,1,1,2,2,2],[0,1,2,0,1,2,0,1,2]),num_nodes=3)
print(g)

gx = dgl.to_networkx(g)
nx.draw_networkx(gx)
plt.show()

def calculateDiagonal(g):
    degs = g.in_degrees().float()
    norm = torch.pow(degs,-0.5)
    norm[torch.isinf(norm)] = 0
    return norm.unsqueeze(1)

g.ndata["norm"] = calculateDiagonal(g)
print(g.ndata["norm"])


