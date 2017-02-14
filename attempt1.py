import json
import networkx as nx
import numpy as np
import sim

with open('./testgraph1.json') as data_file:
    data = json.load(data_file)

num_nodes = len(data)

adj = np.zeros((num_nodes, num_nodes))

for i in range(num_nodes):
    for neighbor in data[str(i)]:
        adj[i][int(neighbor)] = 1

G = nx.from_numpy_matrix(adj)


def get_random(k):
    nodes = [str(i) for i in list(data.keys())]
    return np.ndarray.tolist(np.random.choice(nodes, k))


def repeat_strategy(k, num_iters, strat_name):
    choices = []
    if strat_name == 'random':
        for i in range(num_iters):
            choices.append(get_random(k))
    return choices


def run_simulation(k, name_strat):
    nodes = {}
    for name in list(name_strat.keys()):
        if name_strat[name] == 'random':
            nodes[name] = get_random(k)
    print(sim.run(data, nodes))

# run_simulation(5, {'strat1': 'random', 'strat2': 'random'})


def print_out(choices, outfile_path):
    f = open(outfile_path, 'w')
    for iter_choices in choices:
        for node in iter_choices:
            f.write(str(node) + '\n')

#choices = repeat_strategy(5, 2, 'random')
#print_out(choices, './outputtest.txt')
