"""
This script is writing to calculate the network statistics of four datasets
info: num_nodes, num_edges, num_timestamps, density, average_degree 
"""

import networkx as nx

dataset_list = ["BTW17", "Chicago_COVID","Crawled_Dataset26", "Crawled_Dataset144"]
dataset_dir = "D:/Cohesion_Evaluation/Original_Datasets/Preprocessed_Datasets/"

for dataset in dataset_list:
    dataset_file = dataset + "_attributed.txt"
    network = nx.read_edgelist(dataset_dir + dataset_file, create_using=nx.MultiDiGraph(), nodetype=str, data=(('timestamp', str), ('sentiment', str), ), delimiter='\t') # type: ignore
    num_nodes = network.number_of_nodes()
    num_edges = network.number_of_edges()
    num_timestamps = len(set([int(d['timestamp']) for u, v, d in network.edges(data=True)]))
    density = nx.density(network)
    average_degree = sum(dict(network.degree()).values()) / num_nodes
    print(f"{dataset} & {num_nodes:,} & {num_edges:,} & {num_timestamps:,} & {density:.4f} & {average_degree:.4f}\\\\")