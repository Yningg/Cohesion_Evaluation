"""
This script stores the functions that are commonly used in the experiments.
"""

import networkx as nx
import ast

# Get query nodes from the file
def get_query_nodes(query_node_path, dataset_name):
    query_node_file = query_node_path + dataset_name + "_query_node.txt"
    query_nodes = []
    
    with open(query_node_file, 'r') as f:
        for line in f:
            query_nodes.append(line.strip())
        f.close()
    
    return query_nodes


"""
Read the network and construct the edge stream and temporal adjacency list
--> edge stream: {timestamp: [(u, v, timestamp, sentiment)]}
--> temporal adjacency list: {u: [(u, v, timestamp, sentiment)]}
"""

def build_graph(G):
    tadj_list, edge_stream = {}, {}

    for u, v, d in G.edges(data=True):
        timestamp, sentiment = int(d['timestamp']), int(d['sentiment'])
        if timestamp in edge_stream:
            edge_stream[timestamp].append((u, v, timestamp, sentiment))
        else:
            edge_stream[timestamp] = [(u, v, timestamp, sentiment)]

        if u in tadj_list:
            tadj_list[u].append((u, v, timestamp, sentiment))
        else:
            tadj_list[u] = []
            tadj_list[u].append((u, v, timestamp, sentiment))
        
        if v in tadj_list:
            tadj_list[v].append((u, v, timestamp, sentiment))
        else:
            tadj_list[v] = []
            tadj_list[v].append((u, v, timestamp, sentiment))

    # Sort the edge stream and temporal adjacency list based on the timestamp
    edge_stream = dict(dict(sorted(edge_stream.items())))
    for u in tadj_list:
        tadj_list[u] = sorted(tadj_list[u], key=lambda x: x[2])

    return edge_stream, tadj_list


# Build the graph with the original nodes and edges attributes
def graph_construction(attribute_file):
    # Read the attribute file and add the attributes to the graph
    attributed_G = nx.read_edgelist(attribute_file, nodetype=str, data=(('timestamp', str), ('sentiment', str)), create_using=nx.MultiGraph()) # type: ignore
    
    print(f"Original graph info: {attributed_G.number_of_nodes()} nodes, {attributed_G.number_of_edges()} edges, density: {nx.density(attributed_G)}")

    return attributed_G


# Given the graph's edge stream and temporal adjacency list, extract the subgraph based on the subgraph node list
def build_subgraph(edge_stream, tadj_list, subgraph_nodes):
    edge_substream, tadj_sublist = {}, {}

    for timestamp in edge_stream.keys():
        filtered_edges = [edge for edge in edge_stream[timestamp] if edge[0] in subgraph_nodes and edge[1] in subgraph_nodes]
        if len(filtered_edges) > 0:
            edge_substream[timestamp] = filtered_edges

    for node in subgraph_nodes:
        tadj_sublist[node] = []
        for edge in tadj_list[node]:
            if edge[0] in subgraph_nodes and edge[1] in subgraph_nodes:
                tadj_sublist[node].append(edge)
    
    return edge_substream, tadj_sublist



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