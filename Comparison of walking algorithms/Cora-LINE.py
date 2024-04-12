import networkx as nx
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def learn_embeddings_and_train_line_1st(G):
    # Step 2: Learn node embeddings using LINE 1st
    def compute_transition_probs(graph, node, context_size):
        neighbors = list(graph.neighbors(node))
        probs = np.zeros(len(graph.nodes()))
        for neighbor in neighbors:
            probs[int(neighbor)] = 1.0 / len(neighbors)
        return probs

    def generate_random_walks(graph, walk_length, num_walks, context_size):
        walks = []
        for _ in range(num_walks):
            for node in graph.nodes():
                walk = [str(node)]
                current_node = node
                for _ in range(walk_length):
                    transition_probs = compute_transition_probs(graph, current_node, context_size)
                    next_node = np.random.choice(graph.nodes(), p=transition_probs)
                    walk.append(str(next_node))
                    current_node = next_node
                walks.append(walk)
        return walks

    walks = generate_random_walks(G, walk_length=80, num_walks=10, context_size=5)

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
        embedding = np.zeros(len(G.nodes()))
        for walk in walks:
            if str(node) in walk:
                embedding += np.array([1 if str(i) in walk else 0 for i in G.nodes()])
        X.append(embedding)

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
    print("LINE 1st Accuracy:", accuracy)

if __name__ == "__main__":
    # Step 1: Load Cora dataset
    # Load Cora dataset into a networkx graph
    G = nx.read_edgelist('./cora/cora.cites')
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())

    learn_embeddings_and_train_line_1st(G)
