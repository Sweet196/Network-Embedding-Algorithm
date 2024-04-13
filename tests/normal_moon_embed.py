#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

n = 2000
DISTANCE = 0.1

# Generate moon dataset
x, y = make_moons(n_samples=n, noise=0.1, random_state=0) 
# Convert to PyTorch tensors
x = torch.tensor(x, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)

# Calculate pairwise distances
D = np.sqrt(((x[:, None] - x[None, :]) ** 2).sum(-1))
# Create adjacency matrix
A_binary = torch.tensor(np.where(D <= DISTANCE, 1, 0), dtype=torch.float)

# Define GCN model
class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(2, 16)  # Input dimension is 2 (x, y coordinates)
        self.conv2 = GCNConv(16, 2)   # Output dimension is 2 for binary classification

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Convert adjacency matrix to edge list
edges = A_binary.nonzero(as_tuple=False).t().contiguous()
data = Data(x=x, y=y, edge_index=edges)

# Initialize model and optimizer
net = GCN()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# Training loop
net.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = net(data)
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Get the final node embeddings
with torch.no_grad():
    out = net(data)
    embeddings = out.cpu().numpy()

# Plot the embeddings
plt.figure(figsize=(8, 6))
colors = ['r' if y == 0 else 'b' for y in data.y]
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=colors)
plt.title('Node Embeddings')
plt.xlabel('Embedding Dimension 1')
plt.ylabel('Embedding Dimension 2')
plt.grid(True)
plt.show()
