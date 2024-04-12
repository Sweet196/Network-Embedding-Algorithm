import networkx as nx
from node2vec import Node2Vec
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

def learn_embeddings_and_train(window_size, walk_length, num_walks):
    # Step 1: Load Cora dataset
    # Load Cora dataset into a networkx graph
    G = nx.read_edgelist('./cora/cora.cites')
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())

    # Step 2: Learn node embeddings using Node2Vec
    node2vec = Node2Vec(G, dimensions=128, walk_length=walk_length, num_walks=num_walks, workers=4)
    model = node2vec.fit(window=window_size)

    # Step 3: Prepare training data
    X = []
    labels = []
    # Assuming labels are available in a separate file
    with open('./cora/cora.content', 'r') as f:
        for line in f:
            parts = line.strip().split()
            node_id = parts[0]
            label = parts[-1]  # Assuming label is the last element in each line
            labels.append(label)

    for node in G.nodes():
        X.append(model.wv[node])

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Step 4: Train SVM classifier
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)

    # Step 5: Evaluate model
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

if __name__ == "__main__":
    learn_embeddings_and_train(window_size=5, walk_length=40, num_walks=10)
    learn_embeddings_and_train(window_size=10, walk_length=80, num_walks=10)
    learn_embeddings_and_train(window_size=10, walk_length=40, num_walks=80)
