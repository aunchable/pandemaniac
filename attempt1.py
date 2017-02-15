import json
import networkx as nx
import numpy as np
import sim
import operator
from operator import itemgetter

############################################
# Load graph and convert to networkx graph #
############################################
with open('./testgraph1.json') as data_file:
    data = json.load(data_file)

num_nodes = len(data)

adj = np.zeros((num_nodes, num_nodes))

for i in range(num_nodes):
    for neighbor in data[str(i)]:
        adj[i][int(neighbor)] = 1

G = nx.from_numpy_matrix(adj)

####################
# Strategy Section #
####################
def get_random(k):
    nodes = [str(i) for i in list(data.keys())]
    return np.ndarray.tolist(np.random.choice(nodes, k))

def get_top_cluster(k):
    clust_dict = nx.clustering(G)
    sorted_clust = np.array(sorted(clust_dict.items(), key=operator.itemgetter(1)), dtype=str)
    return np.ndarray.tolist(sorted_clust[-k:,0])

def get_top_degree(k):
    degree_dict = G.degree(G.nodes())
    sorted_degree_dict = np.array( sorted(degree_dict.items(), key = itemgetter(1)  ), dtype = str)
    return np.ndarray.tolist(sorted_degree_dict[-k:, 0])

def run_strategy(k, strat_name):
    if strat_name == 'random':
        return get_random(k)
    elif strat_name == 'cluster':
        return get_top_cluster(k)
    elif strat_name == 'degree':
        return get_top_degree(k)
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

def run_simulation(k, strat_name):
    nodes = {}
    for name in list(strat_name.keys()):
        nodes[name] = run_strategy(k, strat_name[name])
    print(sim.run(data, nodes))

run_simulation(5, {'strat1': 'degree', 'strat2': 'random'})

def print_out(choices, outfile_path):
    f = open(outfile_path, 'w')
    for iter_choices in choices:
        for node in iter_choices:
            f.write(str(node) + '\n')
