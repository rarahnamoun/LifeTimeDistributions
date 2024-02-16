import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt


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


def configuration_model(dist, size):
    """
    Generates a graph using the Configuration Model with a degree sequence
    generated from the specified statistical distribution.

    Parameters:
        dist (str): Name of the distribution (e.g., 'exponential', 'normal').
        size (int): Number of nodes in the graph.

    Returns:
        nx.Graph: Graph generated using the Configuration Model.
    """
    degree_sequence = generate_degree_sequence(dist, size)
    return nx.configuration_model(degree_sequence)


def remove_random_edges(graph, attack_percentage):
    """
    Remove random edges from the graph to simulate attacks.

    Parameters:
        graph (nx.Graph): The graph from which edges will be removed.
        attack_percentage (float): Percentage of edges to be removed.

    Returns:
        nx.Graph: Graph after edge removal.
    """
    edges_to_remove = random.sample(list(graph.edges()), int(attack_percentage * graph.number_of_edges()))
    graph.remove_edges_from(edges_to_remove)
    return graph


def lifetime_probability(graph):
    """
    Calculate the lifetime probability of each node in the graph.

    Parameters:
        graph (nx.Graph): The graph.

    Returns:
        dict: Dictionary mapping each node to its lifetime probability.
    """
    lifetime_probabilities = {}
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        degree = len(neighbors)
        probability = (1 - (degree / (len(graph) - 1))) ** len(neighbors)
        lifetime_probabilities[node] = probability
    return lifetime_probabilities


def average_lifetime_probability(graph):
    """
    Calculate the average lifetime probability of nodes in the graph.

    Parameters:
        graph (nx.Graph): The graph.

    Returns:
        float: Average lifetime probability of nodes.
    """
    lifetime_probabilities = lifetime_probability(graph)
    return sum(lifetime_probabilities.values()) / len(graph)


# Example usage:
distribution_types = ['weibull', 'exponential', 'pareto']
num_nodes =100
attack_percentages = [0.1, 0.2, 0.3, 0.4, 0.5]  # Varying attack percentages

plt.figure(figsize=(10, 6))

for dist in distribution_types:
    avg_lifetime_probabilities = []
    for attack_percentage in attack_percentages:
        avg_lifetime_prob_sum = 0
        for _ in range(100):  # Perform attacks 100 times
            graph = configuration_model(dist, num_nodes)
            graph = remove_random_edges(graph, attack_percentage)
            avg_lifetime_prob_sum += average_lifetime_probability(graph)
        avg_lifetime_prob = avg_lifetime_prob_sum / 100  # Average over 100 runs
        avg_lifetime_probabilities.append(avg_lifetime_prob)

    # Plotting
    plt.plot(attack_percentages, avg_lifetime_probabilities, marker='o', label=dist)

plt.xlabel('Attack Percentage')
plt.ylabel('Average Lifetime Probability')
plt.title('Average Lifetime Probability vs Attack Percentage')
plt.legend()
plt.grid(True)
plt.show()
