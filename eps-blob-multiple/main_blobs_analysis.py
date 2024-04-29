#%%
import os
import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.optim as optim
from torch_geometric.data import Data
from sklearn.datasets import make_moons, make_blobs
from torchmetrics import Accuracy
from tqdm.auto import tqdm
import json

np.random.seed(0)
torch.manual_seed(0)

n = 2000
m = 0
x, y = make_blobs(n_samples=2000, random_state=42, n_features=2,centers=8, cluster_std=0.8,
                  center_box=(-10.0,10.0))
n_train = int(n * 0.7)
train_ind = torch.randperm(n)[:n_train]
test_ind = torch.LongTensor(list(set(np.arange(n)) - set(train_ind.tolist())))
D = pairwise_distances(x)
MAX_DISTANCE = np.max(D)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def analyze_network_structure(edges):
    G = nx.Graph()
    G.add_edges_from(edges)
    avg_degree = np.mean([d for n, d in G.degree()])
    num_components = nx.number_connected_components(G)
    component_sizes = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    return avg_degree, num_components, component_sizes

def get_accs_and_network_analysis(function: str, distance):
    A_binary = np.where(D <= distance, 1, 0)
    row_indices, col_indices = np.where(A_binary == 1)
    edges = list(zip(row_indices, col_indices))
    
    avg_degree, num_components, component_sizes = analyze_network_structure(edges)
    
    # Example of using GCN
    from util import GCN, GIN, GAT
    # net = GCN(m).to(device)
    # net = GIN(m).to(device)
    # net = GCN(m).to(device)
    net = eval(function)(m).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    x_tensor = torch.tensor(x, dtype=torch.float).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)
    edges_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    data = Data(x=x_tensor, y=y_tensor, edge_index=edges_tensor)
    # print(data)
    net.train() 
    for epoch in range(10):
        optimizer.zero_grad()
        out = net(data)
        loss = torch.nn.functional.cross_entropy(out[train_ind], y_tensor[train_ind])
        loss.backward()
        optimizer.step()

    net.eval()
    with torch.no_grad():
        pred_labels = net(data).argmax(dim=1)
    accuracy = Accuracy('multiclass', num_classes=8).to(device)
    acc = accuracy(pred_labels[test_ind], y_tensor[test_ind]).item()

    return acc, avg_degree, num_components, component_sizes


if __name__ == "__main__":
    eps_range = np.arange(0.01, MAX_DISTANCE, 0.1)
    results, results_GIN, results_GAT = {}, {}, {}

    print("GCN:")
    for eps in tqdm(eps_range):
        acc, avg_degree, num_components, component_sizes = get_accs_and_network_analysis('GCN', eps)
        results[eps] = {
            'accuracy': acc,
            'average_degree': avg_degree,
            'num_components': num_components,
            'component_sizes': component_sizes
        }

    print("GIN:")
    for eps in tqdm(eps_range):
        acc, avg_degree, num_components, component_sizes = get_accs_and_network_analysis('GIN', eps)
        results_GIN[eps] = {
            'accuracy': acc,
            'average_degree': avg_degree,
            'num_components': num_components,
            'component_sizes': component_sizes
        }
    
    print("GAT:")
    for eps in tqdm(eps_range):
        acc, avg_degree, num_components, component_sizes = get_accs_and_network_analysis('GAT', eps)
        results[eps] = {
            'accuracy': acc,
            'average_degree': avg_degree,
            'num_components': num_components,
            'component_sizes': component_sizes
        }
    
    with open("network_analysis_results_GCN.json", 'w') as f:
        json.dump(results, f, indent=4)

    with open("network_analysis_results_GIN.json", 'w') as f:
        json.dump(results_GIN, f, indent=4)
    
    with open("network_analysis_results_GAT.json", 'w') as f:
        json.dump(results_GAT, f, indent=4)

# %%
