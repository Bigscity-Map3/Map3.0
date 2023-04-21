import numpy as np
inflow_adj = np.load("Data/in_flow_adj.npy")
outflow_adj = np.load("Data/out_flow_adj.npy")
poi_simi = np.load("Data/poi_simi.npy")
od_label = np.load("Data/od_label.npy")
num_nodes = inflow_adj.shape[0]
k = num_nodes//5
n = num_nodes
inflow_adj_sp = np.copy(inflow_adj)
for i in range(n):
    t = np.argsort(inflow_adj_sp[:, i])[:-k]
    inflow_adj_sp[np.argsort(inflow_adj_sp[:, i])[:-k], i] = 0
    inflow_adj_sp[i, np.argsort(inflow_adj_sp[i, :])[:-k]] = 0
outflow_adj_sp = np.copy(outflow_adj)
for i in range(n):
    outflow_adj_sp[np.argsort(outflow_adj_sp[:, i])[:-k], i] = 0
    outflow_adj_sp[i, np.argsort(outflow_adj_sp[i, :])[:-k]] = 0

k= num_nodes//5
poi_adj_sp = np.copy(poi_simi)
for i in range(n):
    poi_adj_sp[np.argsort(poi_adj_sp[:, i])[:-k], i] = 0
    poi_adj_sp[i, np.argsort(poi_adj_sp[i, :])[:-k]] = 0
feature = np.random.uniform(-1, 1, size=(num_nodes, 250))
feature = feature[np.newaxis]