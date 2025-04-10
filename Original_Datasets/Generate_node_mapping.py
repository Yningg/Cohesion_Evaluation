"""
This script is used to generate the node mapping for the original datasets.
"""


import os
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime

import sys
target_path = "./"
sys.path.append(target_path)
import Cohesiveness_Calculation.Utils.Graph_utils as gu


def generate_node_mapping(dataset_name, source_path, target_path):
    attribute_file = source_path + dataset_name + "_attributed.txt"
    G = gu.graph_construction(attribute_file)

    # Get the unique nodes and order them
    nodes = list(G.nodes())
    nodes = sorted(nodes)

    # Create the node mapping
    node_mapping = {}
    for i, node in enumerate(nodes):
        node_mapping[node] = i

    # Save the node mapping
    with open(target_path + dataset_name + "_node_mapping.txt", "w") as f:
        for node, node_id in node_mapping.items():
            f.write(f"{node}\t{node_id}\n")


if __name__ == "__main__":
    dataset_list = ["BTW17", "Chicago_COVID", "Crawled_Dataset144", "Crawled_Dataset26"]

    # Set for the source path
    source_path = "D:/Cohesion_Evaluation/Original_Datasets/Preprocessed_Datasets/"

     # Set for storing node mapping 
    target_path_node_mapping = "D:/Cohesion_Evaluation/Original_Datasets/Node_Mapping/"
    

    for dataset_name in dataset_list:
        # Given preprocessed dataset, generate its node mapping list first
        generate_node_mapping(dataset_name, source_path, target_path_node_mapping)