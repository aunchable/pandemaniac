# deg = G.degree()
# to_remove = [n for n in deg if deg[n] <= 2]
# G.remove_nodes_from(to_remove)
# # print nx.connected_components(G)
# subgraphs = [G.subgraph(c) for c in sorted(nx.connected_components(G), key=len, reverse=True) if len(c) > len(G) / 10.0 ]
# print [len(n) for n in subgraphs]
# # for u,v,d in G.edges(data=True):
# #     d['capacity']=1
# i = 0
# while i < 5:
#     G1 = subgraphs[-1]

#     s = choice(G1.nodes())
#     possible_nodes = set(G1.nodes())
#     neighbours = G1.neighbors(s) + [s]
#     possible_nodes.difference_update(neighbours)    # remove the first node and all its neighbours from the candidates
#     t = choice(list(possible_nodes))
#     print s,t

#     cuts = nx.minimum_edge_cut(G1, s, t)
#     print cuts

#     for edge in cuts:
#         G1.remove_edge(*edge)
#     subgraphs1 = [G1.subgraph(c) for c in nx.connected_components(G1)]
#     subgraphs = subgraphs[:-1] + subgraphs1
#     subgraphs = sorted(subgraphs)
#     print [len(n) for n in subgraphs]
#     i += 1