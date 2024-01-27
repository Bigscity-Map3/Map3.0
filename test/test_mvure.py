import numpy as np
a = np.load("Data/ind_to_geo_MVURE.npy",allow_pickle=True)
inflow_adj = np.load("Data/in_flow_adj_MVURE.npy")
outflow_adj = np.load("Data/out_flow_adj_MVURE.npy")
poi_simi = np.load("Data/poi_simi_MVURE.npy")
print(np.count_nonzero(inflow_adj))
print(np.count_nonzero(outflow_adj))
print(np.count_nonzero(poi_simi))
od_label = np.load("Data/od_label_MVURE.npy")
print(np.count_nonzero(np.count_nonzero(od_label)))
print(od_label[659][618])
a = od_label[:][618]
print(sum(od_label[:][618]))
num_nodes = inflow_adj.shape[0]
ind_to_geo = np.load("../libcity/cache/dataset_cache/xa_dataset/ind_to_geo_od.npy")
mx = inflow_adj+outflow_adj
print(np.count_nonzero(mx))
for i in range(num_nodes):
    if sum(od_label[i]) == 0 and sum(od_label[:,i]) == 0:
        print(ind_to_geo[i])
print("no")
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