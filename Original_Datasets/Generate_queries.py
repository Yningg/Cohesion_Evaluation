"""
This script is written to generate the queries for community search algos
1. Decide the scope of the nodes sampled from the network (e.g., the nodes with the higher degree)
2. Decide the number of nodes sampled from the network: 100
"""

import networkx as nx
import numpy as np
import pandas as pd
import os
import random

seed = 2024
random.seed(seed)
query_num = 100
degree_percentage = 0.1


# Load the network
def load_network(file_path, file_name):
    G = nx.read_edgelist(file_path + file_name, create_using=nx.MultiDiGraph(), nodetype=str, data=(('timestamp', str), ('sentiment', str), ), delimiter='\t') # type: ignore
    print(f"Graph info: nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}, density: {nx.density(G)}")
    return G


def generate_query_nodes(G, query_num, degree_percentage):
    degree_dict = dict(G.degree())
    sorted_degree = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)
    degree_threshold = sorted_degree[int(len(sorted_degree) * degree_percentage)][1]

    print(f"Max degree: {sorted_degree[0][1]}, Min degree: {sorted_degree[-1][1]}")
    print(f"Degree threshold: {degree_threshold}")

    candidate_nodes = [node for node, degree in sorted_degree if degree >= degree_threshold]
    print(f"Number of candidate nodes: {len(candidate_nodes)}")

    query_nodes = random.sample(candidate_nodes, query_num)
    return query_nodes


def node2txt(target_path, dataset_name, query_nodes):
    with open(network_target_path + dataset_name + "_query_node.txt", 'w') as f:
        for node in query_nodes:
            f.write(f"{node}\n")
        f.close()

    print(f"Query nodes are saved in {target_path + dataset_name + '_query_node.txt'}")


def mapped_query_nodes(target_path, node_mapping_path, dataset_name):
    query_nodes = []
    with open(target_path + dataset_name + "_query_node.txt", 'r') as f:
        for line in f:
            query_nodes.append(int(line.strip()))
        f.close()

    node_mapping = pd.read_csv(node_mapping_path + dataset_name + "_node_mapping.txt", sep='\t', header=None)
    mapping_dict = dict(zip(node_mapping[0], node_mapping[1]))

    mapped_query_nodes = [mapping_dict[node] for node in query_nodes]

    with open(target_path + dataset_name + "_mapped_query_node.txt", 'w') as f:
        for node in mapped_query_nodes:
            f.write(f"{node}\n")
        f.close()

    print(f"Mapped query nodes are saved in {target_path + dataset_name + '_mapped_query_node.txt'}")


if __name__ == "__main__":

    dataset_list = ["BTW17", "Chicago_COVID", "Crawled_Dataset144", "Crawled_Dataset26"]

    for dataset_name in dataset_list:
        network_file_path = 'D:/Cohesion_Evaluation/Original_Datasets/Preprocessed_Datasets/'
        network_target_path = 'D:/Cohesion_Evaluation/Original_Datasets/Query_nodes/'
        node_mapping_path = 'D:/Cohesion_Evaluation/Original_Datasets/Node_Mapping/'
        network_file_name = dataset_name + "_attributed.txt"

        # Load the network
        G = load_network(network_file_path, network_file_name)

        # Choose query node based on the degree of the nodes
        query_nodes = generate_query_nodes(G, query_num, degree_percentage)


        # Output the query nodes into a .txt file, each line is a query node
        node2txt(network_target_path, dataset_name, query_nodes)

        # Read generated query nodes, according to the node mapping, and generate the corresponding mapped query nodes, and save them into a .txt file
        mapped_query_nodes(network_target_path, node_mapping_path, dataset_name)