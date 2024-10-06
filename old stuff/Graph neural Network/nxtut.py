import networkx as nx
G = nx.Graph()
G.add_node(1)
print("G1: {}".format(G))
G.add_nodes_from([2, 3])
G.add_nodes_from([
    (4, {"color": "red"}),
    (5, {"color": "green"}),
])
print("G2: {}".format(G))

H = nx.path_graph(10)
print("H: {}".format(H))

G.add_node(H)
print("G3: {}".format(G))
G.add_edge(1, 2)
e = (2, 3)
G.add_edge(*e)
G.add_edges_from([(1, 2), (1, 3)])
G.add_edges_from(H.edges)
print("G4: {}".format(G))

G.clear()
print("G5: {}".format(G))


G.add_edges_from([(1, 2), (1, 3)])
print("G6: {}".format(G))
G.add_node(1)
G.add_edge(1, 2)
print("G7: {}".format(G))
G.add_node("spam")
print("G8: {}".format(G))
G.add_nodes_from("spam")
print("G9: {}".format(G))
G.add_edge(3, 'm')
print("G10: {}".format(G))
v=G.number_of_nodes()
e=G.number_of_edges()
print(v,e)

print(list(G.nodes))
print(list(G.edges))
print(list(G.adj[1]))
print(G.degree[1])
