import json
import networkx as nx
import numpy as np
import sim
from operator import itemgetter
import time
import random
import matplotlib.pyplot as plt

############################################
# Load graph and convert to networkx graph #
############################################
#with open('graphs/testgraph1.json') as data_file:
with open('/Users/anshulramachandran/Downloads/8.35.2.json') as data_file:
# with open('./testgraph1.json') as data_file:
    data = json.load(data_file)

num_nodes = len(data)

adj = np.zeros((num_nodes, num_nodes))

for i in range(num_nodes):
    for neighbor in data[str(i)]:
        adj[i][int(neighbor)] = 1

G = nx.from_numpy_matrix(adj)
# # G=nx.dodecahedral_graph()
# pos = nx.spring_layout(G)
# #nx.draw(G)
# #plt.show()
#
# G1 = G.copy()
# old_len = -1
# while len(G1.nodes()) != old_len:
#     old_len = len(G1.nodes())
#     deg = G1.degree()
#     to_remove = [n for n in deg if deg[n] <= 10]
#     G1.remove_nodes_from(to_remove)
#     print old_len
# #nx.draw_networkx_nodes(G1, pos)
# #nx.draw_networkx_edges(G1, pos)
# #plt.show()
#
# bw_centrality_dict = nx.betweenness_centrality(G1)
# # dg_centrality_dict = nx.degree_centrality(G1)
# alpha =1
# ranking = {}
# for key in bw_centrality_dict.keys():
#     ranking[key] = alpha * bw_centrality_dict[key]
# print ranking
# ranking = np.array(sorted(ranking.items(), key=itemgetter(1)))
# drop = np.ndarray.tolist(ranking[:, 0])
# while nx.is_connected(G1):
#     print(len(G1.nodes()))
#     if (drop[-1] in G1.nodes()):
#         G1.remove_node(drop[-1])
#     drop = drop[:-1]
#     deg = G1.degree()
#     to_remove = [n for n in deg if deg[n] <= 1]
#     G1.remove_nodes_from(to_remove)
# # G1.remove_nodes_from(drop)
# print drop

#nx.draw_networkx_nodes(G1, pos)
#nx.draw_networkx_edges(G1, pos)
#plt.show()
####################
# Strategy Section #
####################
def get_random(k):
    nodes = [str(i) for i in list(data.keys())]
    return np.ndarray.tolist(np.random.choice(nodes, k))

def get_top_cluster(k):
    clust_dict = nx.clustering(G)
    sorted_clust = np.array(sorted(clust_dict.items(), key=itemgetter(1)), dtype=str)
    return np.ndarray.tolist(sorted_clust[-k:,0])

def get_top_degree(k):
    degree_dict = G.degree(G)
    sorted_degree_dict = np.array( sorted(degree_dict.items(), key = itemgetter(1)  ), dtype = str)
    return np.ndarray.tolist(sorted_degree_dict[-k:, 0])

def get_top_katz(k):
    katz_dict = nx.katz_centrality(G, 0.02)
    sorted_katz_dict = np.array( sorted(katz_dict.items(), key = itemgetter(1)  ), dtype = str)
    return np.ndarray.tolist(sorted_katz_dict[-k:, 0])

def get_centrality_sum(k):
    dg_centrality_dict = nx.degree_centrality(G)
    cl_centrality_dict = nx.closeness_centrality(G)
    bw_centrality_dict = nx.betweenness_centrality(G)
    centrality_dict = {}
    for key in bw_centrality_dict.keys():
        centrality_dict[key] = dg_centrality_dict[key] + cl_centrality_dict[key] + bw_centrality_dict[key]
    sorted_centrality_dict = np.array( sorted(centrality_dict.items(), key = itemgetter(1)  ), dtype = str)
    return np.ndarray.tolist(sorted_centrality_dict[-k:, 0])

def get_centrality_sum_randomized_multiple(k, distr, num_iters):
    dg_centrality_dict = nx.degree_centrality(G)
    cl_centrality_dict = nx.closeness_centrality(G)
    bw_centrality_dict = nx.betweenness_centrality(G)
    centrality_dict = {}
    for key in bw_centrality_dict.keys():
        centrality_dict[key] = dg_centrality_dict[key] + cl_centrality_dict[key] + bw_centrality_dict[key]
    sorted_centrality_dict = np.array( sorted(centrality_dict.items(), key = itemgetter(1)  ), dtype = str)
    topk = [i[0] for i in sorted_centrality_dict[-k:]]
    middle2k = [i[0] for i in sorted_centrality_dict[-3*k: -k]]
    bottom3k = [i[0] for i in sorted_centrality_dict[-6*k: -3*k]]
    numtopk = int(float(k) * distr[0] / sum(distr))
    nummiddle2k = int(float(k) * distr[1] / sum(distr))
    numbottom3k = k - numtopk - nummiddle2k
    choices = []
    for i in range(num_iters):
        random.shuffle(topk)
        random.shuffle(middle2k)
        random.shuffle(bottom3k)
        choices.append(topk[0:numtopk] + middle2k[0:nummiddle2k] + bottom3k[0:numbottom3k])
    return choices


def run_strategy(k, strat_name):
    if strat_name == 'random':
        return get_random(k)
    elif strat_name == 'cluster':
        return get_top_cluster(k)
    elif strat_name == 'degree':
        return get_top_degree(k)
    elif strat_name == 'katz_central':
        return get_top_katz(k)
    elif strat_name == 'centrality':
        return get_centrality_sum(k)
    else:
        return []



#######################
# Simulation Handling #
#######################
def repeat_strategy(k, num_iters, strat_name):
    choices = []
    for i in range(num_iters):
        choices.append(run_strategy(k, strat_name))
    return choices

def repeat_same_strategy(k, num_iters, strat_name):
    choices = [run_strategy(k, strat_name)] * 50
    return choices

def run_simulation(k, strat_name):
    nodes = {}
    for name in list(strat_name.keys()):
        nodes[name] = run_strategy(k, strat_name[name])
    print(sim.run(data, nodes))

def run_multiple_simulations(num_simulations, k, strat_name):
    win_dict = {}

    times = {}
    nodes = {}
    for name in list(strat_name.keys()):
        times[name] = []

    quads = 0
    for i in range(num_simulations):
        while i >= (quads+1) * (num_simulations / 4.0):
            quads += 1
            print ("Done with ~{0}% of simulations ({1} / {2})...".format(quads*25, i, num_simulations))

        for name in list(strat_name.keys()):
            start = time.clock()
            nodes[name] = run_strategy(k, strat_name[name])
            times[name] = time.clock() - start

        sim_results = sim.run(data, nodes)
        max_key = max(sim_results.iteritems(), key=itemgetter(1))[0]

        if strat_name[max_key] not in win_dict:
            win_dict[strat_name[max_key]] = 1
        else:
            win_dict[strat_name[max_key]] += 1

    avg_times = {}
    for name in times:
        avg_times[name] = np.mean(times[name])

    return win_dict, avg_times

# print run_multiple_simulations(50, 5, {'strat1': 'degree', 'strat2': 'cluster', 'strat3': 'katz_central', 'strat4': 'centrality'})

def print_out(choices, outfile_path):
    f = open(outfile_path, 'w')
    for iter_choices in choices:
        for node in iter_choices:
            f.write(str(node) + '\n')

choices = repeat_strategy(35, 50, 'random')
print_out(choices, '/Users/anshulramachandran/Downloads/submission8.35.2.1.txt')
# choices = repeat_same_strategy(35, 50, 'centrality')
choices = get_centrality_sum_randomized_multiple(35, [7,14,14], 50)
#print_out(choices, './output1.txt')
print_out(choices, '/Users/anshulramachandran/Downloads/submission8.35.2.2.txt')
