import dgl
import dgl.data
import networkx as nx
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = dgl.data.CoraFullDataset()
g = data[0]

X = torch.FloatTensor(g.ndata["feat"])
Y = torch.LongTensor(g.ndata["label"])
print(X.shape)
print(Y.shape)
print(g.ndata["norm"])