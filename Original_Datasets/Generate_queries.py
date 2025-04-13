"""
Generate the queries for community search algos
1. Decide the scope of the nodes sampled from the network (e.g., the nodes with the higher degree)
2. Decide the number of nodes sampled from the network: 100
"""

import pandas as pd
import random

import sys
target_path = "./"
sys.path.append(target_path)
import Cohesiveness_Calculation.Utils.Graph_utils as gu


seed = 2024
random.seed(seed)
query_num = 100
degree_percentage = 0.1


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
        dataset_path = 'D:/Cohesion_Evaluation/Original_Datasets/Preprocessed_Datasets/'
        query_path = 'D:/Cohesion_Evaluation/Original_Datasets/Query_nodes/'
        node_mapping_path = 'D:/Cohesion_Evaluation/Original_Datasets/Node_Mapping/'
        dataset = dataset_name + "_attributed.txt"

        # Load the network
        G = gu.graph_construction(dataset_path + dataset)

        # Choose query node based on the degree of the nodes
        query_nodes = generate_query_nodes(G, query_num, degree_percentage)

        # Output the query nodes into a .txt file, each line is a query node
        node2txt(query_path, dataset_name, query_nodes)

        # According to the node mapping, and generate the corresponding mapped query nodes, and save them into a .txt file
        mapped_query_nodes(query_path, node_mapping_path, dataset_name)