import networkx as nx
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
    attributed_G = nx.read_edgelist(attribute_file, nodetype=str, data=(('timestamp', str), ('sentiment', str)), create_using=nx.MultiDiGraph()) # type: ignore
    print(f"Original graph info: {attributed_G.number_of_nodes()} nodes, {attributed_G.number_of_edges()} edges, density: {nx.density(attributed_G)}")

    return attributed_G


"""
Read the network and construct the temporal adjacency list
temporal adjacency list: {u: [(u, v, timestamp, sentiment)]}
"""
def build_tadj(attribute_file):

    tadj_list = defaultdict(list)

    print("Loading the graph...")
    starttime = time.time()

    # The loaded data is already sorted by the timestamp
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
            
            if u != v:
                tadj_list[v].append((u, v, timestamp, sentiment))
            tadj_list[u].append((u, v, timestamp, sentiment))
            
    latest_timestamp = timestamp # The latest timestamp is the last timestamp in the file

    # Sort the temporal adjacency list based on the new timestamp
    for u in tadj_list:
        tadj_list[u] = sorted(tadj_list[u], key=lambda x: x[2]) 

    endtime = time.time()
    print(f"Loading graph time(s): {endtime - starttime}")
    return tadj_list, latest_timestamp


# Given the graph's temporal adjacency list, extract the subgraph based on the subgraph node list
def build_subtadj(tadj_list, subgraph_nodes, t_cur):
    tadj_sublist = defaultdict(list)
    subgraph_nodes_set = set(subgraph_nodes)

    for u, edges in tadj_list.items():
        if u in subgraph_nodes_set:
            for edge in edges:
                if edge[0] in subgraph_nodes_set and edge[1] in subgraph_nodes_set and edge[2] <= t_cur:
                    tadj_sublist[u].append(edge)

    return dict(tadj_sublist)