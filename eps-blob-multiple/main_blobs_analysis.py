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
    from util import GCN
    net = GCN(m).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    x_tensor = torch.tensor(x, dtype=torch.float).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)
    edges_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    data = Data(x=x_tensor, y=y_tensor, edge_index=edges_tensor)
    print(data)
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
    accuracy = Accuracy('binary').to(device)
    acc = accuracy(pred_labels[test_ind], y_tensor[test_ind]).item()

    return acc, avg_degree, num_components, component_sizes

eps_range = np.arange(0.01, MAX_DISTANCE, 0.1)
results = {}

for eps in tqdm(eps_range):
    acc, avg_degree, num_components, component_sizes = get_accs_and_network_analysis('GCN', eps)
    results[eps] = {
        'accuracy': acc,
        'average_degree': avg_degree,
        'num_components': num_components,
        'component_sizes': component_sizes
    }

# Save results to a file
with open('./analysis/network_analysis_results.json', 'w') as file:
    json.dump(results, file, indent=4)

plt.figure(figsize=(10, 5))
for eps, res in results.items():
    plt.scatter(eps, res['accuracy'], color='blue')
plt.xlabel('Epsilon')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epsilon')
plt.grid(True)
plt.savefig("./analysis/accuracy_vs_epsilon.png")
plt.show()
#%%
import json
from matplotlib import pyplot as plt
import numpy as np

# Load results from a JSON file
with open('./analysis/network_analysis_results.json', 'r') as file:
    results = json.load(file)

plt.figure(figsize=(10, 5))

# Prepare lists for plotting
epss = []
res_acc = []
res_degree = []
res_num_component = []
res_component_size1 = []# the max component size
res_component_size2 = []# the second_large component size

# Extract data from results
for eps, res in sorted(results.items(), key=lambda x: float(x[0])):  # Ensure the keys are sorted numerically
    if(len(res['component_sizes'])==1):
        break
    epss.append(float(eps))  # Convert eps to float for numerical plotting
    res_degree.append(res['average_degree'])
    res_num_component.append(res['num_components'])
    # Assuming res['component_sizes'] is a list, we could plot its average size
    res_component_size1.append(res['component_sizes'][0])  # Example: calculating the mean of component sizes
    res_component_size2.append(res['component_sizes'][1])  # Example: calculating the mean of component sizes

# Plotting
plt.plot(epss, res_degree, label='Average Degree')
plt.plot(epss, res_num_component, label='Number of Components')
plt.plot(epss, res_component_size1, label='Largest Component Size')  # Now plotting the average component size
plt.plot(epss, res_component_size2, label='Second-Largest Component Size')  # Now plotting the average component size

# Setup legend and labels
plt.legend(loc='upper right')
plt.xlabel('Epsilon')
plt.ylabel('Values')
# plt.xticks(np.arange(0, 3.6, 0.2))  # Ensure this range makes sense for your data

# Set grid visibility
plt.grid(True)

# Save the figure
plt.savefig("./analysis/accuracy_vs_epsilon.png")

# Display the plot
plt.show()
# %%
