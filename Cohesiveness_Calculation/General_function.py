"""
This script stores the functions that are commonly used in the experiments.
"""

import networkx as nx
import ast
from collections import defaultdict
import time
import csv

# Get query nodes from the file
def get_query_nodes(query_node_path, dataset_name):
    query_node_file = query_node_path + dataset_name + "_query_node.txt"
    query_nodes = []
    
    with open(query_node_file, 'r') as f:
        for line in f:
            query_nodes.append(line.strip())
        f.close()
    
    return query_nodes



# Build the graph with the original nodes and edges attributes
def graph_construction(attribute_file):
    # Read the attribute file and add the attributes to the graph
    attributed_G = nx.read_edgelist(attribute_file, nodetype=str, data=(('timestamp', str), ('sentiment', str)), create_using=nx.MultiDiGraph()) # type: ignore
    
    print(f"Original graph info: {attributed_G.number_of_nodes()} nodes, {attributed_G.number_of_edges()} edges, density: {nx.density(attributed_G)}")

    return attributed_G


"""
Read the network and construct the edge stream and temporal adjacency list
--> edge stream: {timestamp: [(u, v, timestamp, sentiment)]}
--> temporal adjacency list: {u: [(u, v, timestamp, sentiment)]}
"""

def build_graph(attribute_file):

    edge_stream = defaultdict(list)
    tadj_list = defaultdict(list)

    print("Loading the graph...")
    starttime = time.time()

    # The loaded data is already sorted, since the data is sorted by the timestamp
    with open(attribute_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        # Get the initial timestamp by read the first line of the file
        flag_read = False

        for parts in reader:
            if not flag_read:
                initial_timestamp = int(parts[2])
                flag_read = True
            
            u, v, timestamp, sentiment = parts[0], parts[1], int(parts[2]), int(parts[3])
            timestamp = int(parts[2]) - initial_timestamp
            
            edge_stream[timestamp].append((u, v, timestamp, sentiment))
            tadj_list[u].append((u, v, timestamp, sentiment))
            tadj_list[v].append((u, v, timestamp, sentiment))


    # Sort the temporal adjacency list based on the new timestamp
    for u in tadj_list:
        tadj_list[u] = sorted(tadj_list[u], key=lambda x: x[2]) 

    endtime = time.time()
    print(f"Loding graph time(s): {endtime - starttime}")
    return edge_stream, tadj_list


# Given the graph's edge stream and temporal adjacency list, extract the subgraph based on the subgraph node list
def build_subgraph(edge_stream, tadj_list, subgraph_nodes):
    edge_substream = defaultdict(list)
    tadj_sublist = defaultdict(list)
    subgraph_nodes_set = set(subgraph_nodes)

    for timestamp, edges in edge_stream.items():
        for edge in edges:
            u, v = edge[0], edge[1]
            if u in subgraph_nodes_set and v in subgraph_nodes_set:
                edge_substream[timestamp].append(edge)
                if edge not in tadj_sublist[u]:
                    tadj_sublist[u].append(edge)
                if edge not in tadj_sublist[v]:
                    tadj_sublist[v].append(edge)

    return dict(edge_substream), dict(tadj_sublist)


# Read the node mapping file
def read_node_mapping(node_mapping_file):
    node_mapping = {}
    with open(node_mapping_file, 'r') as f:
        lines = f.readlines()
        # Reverse the mapping to map the new nodes back to the original nodes
        node_mapping = {int(line.split("\t")[1]): int(line.split("\t")[0]) for line in lines}
    
    return node_mapping


"""
Functions to read the results produced by the various algorithms
"""

def process_ALS_CRC_I2ACSM_results(file):
    results = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split("\t")
            node = int(parts[0])
            score = float(parts[1])
            params = ast.literal_eval(parts[2])
            community_node_list = ast.literal_eval(parts[3])
            community_node_list = [str(node) for node in community_node_list]
            results.append([node, score, params, community_node_list])
    return results



def process_CSD_STExa_Repeeling_results(file, node_mapping):
    results = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split("\t")
            node = int(parts[0])
            node = node_mapping[node]

            params = ast.literal_eval(parts[1])
            
            community_node_list = ast.literal_eval(parts[2])
            community_node_list = [str(node_mapping[node]) for node in community_node_list]

            results.append([node, params, community_node_list])
    return results



def process_TransZero_results(file, node_mapping):
    results = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split('\t')
            node = ast.literal_eval(parts[0])[0]
            node = node_mapping[node]
            
            community_node_list = ast.literal_eval(parts[1])
            community_node_list = [str(node_mapping[node]) for node in community_node_list]
            results.append([node, community_node_list])

    return results