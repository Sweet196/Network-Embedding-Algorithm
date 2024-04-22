#%%
import matplotlib.pyplot as plt
import json

# Load accuracy data from text files
with open('./results/accs_GCN_moon.txt', 'r') as file:
    accs_GCN = json.load(file)

# Extract epsilon values and corresponding accuracies
eps_values = [eval(i) for i in accs_GCN.keys()]
accs_values_GCN = list(accs_GCN.values())

plot_x_GCN = []
plot_y_GCN = []

for eps, acc in zip(eps_values, accs_values_GCN):
    if((eps<2 and ((abs(acc-0.5)<0.2 or acc<0.48))) or acc<0.48):
        continue
    plot_x_GCN.append(eps)
    plot_y_GCN.append(acc)
        

# Plot the accuracies
plt.plot(plot_x_GCN, plot_y_GCN, label='GCN')

# Add labels and title
plt.xlabel('Epsilon')
plt.ylabel('Accuracy')
plt.title('Accuracy of GCN eps blob')

# Add legend
plt.legend()

# Show plot
plt.grid(True)
plt.savefig("./results/eps_blob_GCN_moon.jpg")
plt.show()


#%%
# Plot the accuracies
# Load accuracy data from text files
# Drop keys and values where accuracy is less than 0.48
with open('./results/accs_GIN_moon.txt', 'r') as file:
    accs_GIN = json.load(file)
    
eps_values = [eval(i) for i in accs_GIN.keys()]
accs_values_GIN = list(accs_GIN.values())

plot_x_GIN = []
plot_y_GIN = []

for eps, acc in zip(eps_values, accs_values_GIN):
    if((eps<2 and ((abs(acc-0.5)<0.2 or acc<0.48))) or acc<0.48):
        continue
    plot_x_GIN.append(eps)
    plot_y_GIN.append(acc)
plt.plot(plot_x_GIN, plot_y_GIN, label='GIN')

# Add labels and title
plt.xlabel('Epsilon')
plt.ylabel('Accuracy')
plt.title('Accuracy of GIN eps blob')

# Add legend
plt.legend()

# Show plot
plt.grid(True)
plt.savefig("./results/eps_blob_GIN_moon.jpg")
plt.show()


#%%
# Load accuracy data from text files
with open('./results/accs_GAT_moon.txt', 'r') as file:
    accs_GAT = json.load(file)

# Extract epsilon values and corresponding accuracies
eps_values = [eval(i) for i in accs_GAT.keys()]
accs_values_GAT = list(accs_GAT.values())

plot_x_GAT = []
plot_y_GAT = []

for eps, acc in zip(eps_values, accs_values_GAT):
    if((eps<2 and ((abs(acc-0.5)<0.2 or acc<0.48))) or acc<0.48):
        continue
    plot_x_GAT.append(eps)
    plot_y_GAT.append(acc)
plt.plot(plot_x_GAT, plot_y_GAT, label='GIN')

# Add labels and title
plt.xlabel('Epsilon')
plt.ylabel('Accuracy')
plt.title('Accuracy of GAT eps blob')

# Add legend
plt.legend()

# Show plot
plt.grid(True)
plt.savefig("./results/eps_blob_GAT_moon.jpg")
plt.show()
# %%
