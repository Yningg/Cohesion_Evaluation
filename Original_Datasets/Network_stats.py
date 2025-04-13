"""
Calculate the network statistics of four datasets, including num_nodes, num_edges, num_timestamps, density, average_degree 
"""

import networkx as nx
import sys

target_path = "./"
sys.path.append(target_path)
import Cohesiveness_Calculation.Utils.Graph_utils as gu

dataset_list = ["BTW17", "Chicago_COVID", "Crawled_Dataset144", "Crawled_Dataset26"]
dataset_dir = "D:/Cohesion_Evaluation/Original_Datasets/Preprocessed_Datasets/"

for dataset in dataset_list:
    dataset_file = dataset + "_attributed.txt"
    network = gu.graph_construction(dataset_dir + dataset_file)
    num_nodes = network.number_of_nodes()
    num_edges = network.number_of_edges()
    num_timestamps = len(set([int(d['timestamp']) for u, v, d in network.edges(data=True)]))
    density = nx.density(network)
    average_degree = sum(dict(network.degree()).values()) / num_nodes
    print(f"{dataset} & {num_nodes:,} & {num_edges:,} & {num_timestamps:,} & {density:.4f} & {average_degree:.2f}\\\\")