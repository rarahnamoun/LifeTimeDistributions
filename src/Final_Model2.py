import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt


# Generate degree sequence for configuration model
def generate_degree_sequence(dist, size):
    """
    Generates a degree sequence from a given statistical distribution.

    Parameters:
        dist (str): Name of the distribution (e.g., 'exponential', 'normal').
        size (int): Number of nodes in the graph.

    Returns:
        list: Degree sequence generated from the specified distribution.
    """
    if dist == 'weibull':
        degrees = np.random.weibull(1, size=size).astype(int) + 1
    elif dist == 'exponential':
        degrees = np.random.exponential(2, size=size).astype(int) + 1
    elif dist == 'pareto':
        degrees = np.random.pareto(3, size=size).astype(int) + 1
    else:
        raise ValueError("Unsupported distribution.")

    degrees = np.maximum(degrees, 0)  # Ensure non-negative degrees
    while sum(degrees) % 2 != 0:
        # Adjust sum to be even by adding 1 to a random degree
        index = random.randint(0, size - 1)
        degrees[index] += 1

    return list(degrees)


# Generate configuration graph
def generate_configuration_graph(dist, num_nodes):
    degree_sequence = generate_degree_sequence(dist, num_nodes)
    return nx.configuration_model(degree_sequence)


# Define the GNN model
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Calculate lifetime based on provided formula
def calculate_lifetime(graph):
    avg_degree = np.mean([d for n, d in graph.degree()])
    return avg_degree


# Generate dataset
def generate_dataset(dist, num_nodes, num_graphs):
    graphs = []
    lifetimes = []
    for _ in range(num_graphs):
        graph = generate_configuration_graph(dist, num_nodes)
        lifetime = calculate_lifetime(graph)
        graphs.append(graph)
        lifetimes.append(lifetime)
    return graphs, lifetimes


# Train GNN
def train_gnn(model, optimizer, criterion, train_data, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for graph, lifetime in train_data:
            optimizer.zero_grad()
            x = torch.tensor([[np.mean([d for n, d in graph.degree()])]]).float()
            y = torch.tensor([[lifetime]]).float()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_data)}")


# Test GNN
def test_gnn(model, test_data):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for graph, lifetime in test_data:
            x = torch.tensor([[np.mean([d for n, d in graph.degree()])]]).float()
            y = torch.tensor([[lifetime]]).float()
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
    avg_loss = total_loss / len(test_data)
    print(f"Average Loss on Test Data: {avg_loss}")


# Define parameters
distribution = 'weibull'
num_nodes = 200
num_graphs = 10000
num_epochs = 10
input_dim = 1  # Number of input features (average degree)
hidden_dim = 64
output_dim = 1

# Generate dataset
graphs, lifetimes = generate_dataset(distribution, num_nodes, num_graphs)

# Split dataset into train and test sets
train_data = list(zip(graphs[:80], lifetimes[:80]))
test_data = list(zip(graphs[80:], lifetimes[80:]))

# Initialize GNN model, optimizer, and loss function
model = GNN(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train GNN
train_gnn(model, optimizer, criterion, train_data, num_epochs)

# Test GNN
test_gnn(model, test_data)
