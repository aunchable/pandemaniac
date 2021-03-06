import json
import numpy as np

def erdos_renyi(n,p):
    adj_mat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j and np.random.random() < p:
                adj_mat[i,j] = 1

    return adj_mat

def convert_to_json(adj_mat):
    adj_dic = {}
    for i in range(len(adj_mat)):
        neighbors = []
        for j in range(len(adj_mat[i])):
            if adj_mat[i,j] == 1:
                neighbors.append(str(j))
        adj_dic[str(i)] = neighbors
    return json.dumps(adj_dic)

# graph = erdos_renyi(10,0.5)
# print(graph)
# print(convert_to_json(graph))
