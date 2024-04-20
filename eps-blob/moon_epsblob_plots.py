import matplotlib.pyplot as plt
import json

# Load accuracy data from text files
with open('./results/accs_GCN.txt', 'r') as file:
    accs_GCN = json.load(file)

with open('./results/accs_GIN.txt', 'r') as file:
    accs_GIN = json.load(file)

with open('./results/accs_GAT.txt', 'r') as file:
    accs_GAT = json.load(file)

# Extract epsilon values and corresponding accuracies
eps_values = list(eval(i) for i in accs_GCN.keys())
accs_values_GCN = list(accs_GCN.values())
accs_values_GIN = list(accs_GIN.values())
accs_values_GAT = list(accs_GAT.values())

# Plot the accuracies
plt.plot(eps_values, accs_values_GCN, label='GCN')
plt.plot(eps_values, accs_values_GIN, label='GIN')
plt.plot(eps_values, accs_values_GAT, label='GAT')

# Add labels and title
plt.xlabel('Epsilon')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Models')

# Add legend
plt.legend()

# Show plot
plt.grid(True)
plt.savefig("./results/eps_blob_res.jpg")
plt.show()
