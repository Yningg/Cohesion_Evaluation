"""
Transform the original preprocessed datasets into the format required by each algorithm
"""

import os
import networkx as nx
import sys
import tqdm

target_path = "./"
sys.path.append(target_path)
import Cohesiveness_Calculation.Utils.Graph_utils as gu


# Notice: this function is different from the one in Process_algo.py
def read_node_mapping(node_mapping_file):
    node_mapping = {}
    with open(node_mapping_file, "r") as f:
        lines = f.readlines()
        node_mapping = {int(line.split("\t")[0]): int(line.split("\t")[1]) for line in lines}
    
    return node_mapping


# ALS Dataset Format (Undirected Temporal): from_id \t to_id \t timestamp
# Note the the transformation from directed graph to undirected graph is handled within the algorithm, 
# so we do not need to convert the graph to undirected graph here
def get_ALS_dataset(G, dataset_name, target_path):
    with open(target_path + dataset_name + "_timestamp.txt", "w") as f:
        for u, v, d in G.edges(data=True):
            f.write(str(u) + "\t" + str(v) + "\t" + str(d["timestamp"]) + "\n")


"""
WCF-CRC Dataset Format (Dynamic):
1. Read the attributed graph dataset
2. Generate graph instances (Undirected with no self-loop): guarantee each graph instance contains meaningful k-core components.
    (1) Sort the edges by chronological order
    (2) Divide them into X partitions
    i.e., |E|/|T| edges, where |T| is the target number of graph instances. 
3. Calculate the edge weight from the interaction frequency and is normalized to [0, 1] by min-max normalization.
    Note that the frequency is calculated in each partition and the min-max normalization is performed on the entire dataset.
"""
def get_CRC_dataset(G, dataset_name, target_path, num_instances):
    print(f"**********{dataset_name}**********")
    
    # Remove self-loop edges
    G.remove_edges_from(nx.selfloop_edges(G))
    # Sort edges by timestamp and split into partitions
    edges = sorted(G.edges(data=True), key=lambda x: x[2]['timestamp'])
    num_edges = len(edges)
    num_edges_per_instance = num_edges // num_instances

    edge_partitions = []
    for i in range(num_instances):
        edge_partitions.append(edges[i * num_edges_per_instance: (i + 1) * num_edges_per_instance])

    # Add the remaining edges to the last partition
    if num_edges % num_instances != 0:
        edge_partitions[-1].extend(edges[num_instances * num_edges_per_instance:])

    print(f"Number of edges in each partition: {[len(partition) for partition in edge_partitions]}")

    # Build graph instances and check their maximum k-core components
    graph_instances = {}
    for i, edge_partition in enumerate(edge_partitions):
        G_instance = nx.MultiDiGraph()
        for edge in edge_partition:
            G_instance.add_edge(edge[0], edge[1], timestamp=edge[2]['timestamp'])
        
        print(f"Graph instance {i}: nodes: {G_instance.number_of_nodes()}, edges: {G_instance.number_of_edges()}, density: {nx.density(G_instance)}")

        # Tranfer the graph instance to undirected graph, use the edge number between two nodes as the edge weight
        G_instance_undirected = nx.Graph()
        for u, v in G_instance.edges():
            if not G_instance_undirected.has_edge(u, v):
                forward_count = len(G_instance[u][v])
                reverse_count = len(G_instance[v][u]) if G_instance.has_edge(v, u) else 0
                G_instance_undirected.add_edge(u, v, weight=forward_count + reverse_count)

        # Check the k-core components of the graph instance
        core_number = 2 
        k_core = nx.k_core(G_instance_undirected, k=core_number)
        print(f"{core_number}-core info: nodes: {k_core.number_of_nodes()}, edges: {k_core.number_of_edges()} \n")

        # Save the graph instance
        graph_instances[f"graph_instance_{i}"] = G_instance_undirected

    # Add edge weights to the graph instances according to the interaction frequency
    min_freq, max_freq = 1, 1

    # Iterate through all the graph instances to calculate the min and max frequency
    for i in range(num_instances):
        G_instance = graph_instances[f"graph_instance_{i}"]
        edge_weights = [G_instance[edge[0]][edge[1]]['weight'] for edge in G_instance.edges()]

        temp_min_freq = min(min_freq, min(edge_weights))
        temp_max_freq = max(max_freq, max(edge_weights))
        print(f"Partition {i}: min_weight: {min(edge_weights)}, max_weight: {max(edge_weights)}")

        if temp_max_freq > max_freq:
            max_freq = temp_max_freq

    print("\n After normalization:")
    # Normalize the edge weights to [0, 1] using min-max normalization
    for i in range(num_instances):
        G_instance = graph_instances[f"graph_instance_{i}"]
        for edge in G_instance.edges():
            G_instance[edge[0]][edge[1]]['weight'] = (G_instance[edge[0]][edge[1]]['weight'] - min_freq) / (max_freq - min_freq)

        print(f"Partition {i}: min_weight: {min([G_instance[edge[0]][edge[1]]['weight'] for edge in G_instance.edges()])}, "
            f"max_weight: {max([G_instance[edge[0]][edge[1]]['weight'] for edge in G_instance.edges()])}")

    # Ensure the target directory exists
    target_dir = target_path + f"{dataset_name}/" 
    os.makedirs(target_dir, exist_ok=True)

    # Save the graph instances as .gml files
    for i in range(num_instances):
        nx.write_gml(graph_instances[f"graph_instance_{i}"], target_dir + f"graph_instance_{i}.gml")
        print(f"Graph instance {i} saved as graph_instance_{i}.gml")


"""
CSD Dataset (Directed Multigraph with no self-loop):
1. Read the original attributed dataset and node mapping file
3. Generate two file for each dataset use node mapping:
    (1) File recording every node's in degree and out degree: DatasetName_Degree.dat
        Format: node_id in_degree out_degree
    (2) File recording the graph in a adjacent format: DatasetName_Graph.dat
        Format: node_id neighbor1 neighbor2 ...
"""
def get_CSD_dataset(G, dataset_name, node_mapping, target_path):

    target_dir = target_path + f"{dataset_name}/" 
    os.makedirs(target_dir, exist_ok=True)

    # Generate the degree file, order by the node id
    degree_list = []
    adj_list = {}

    for original_id, mapped_id in node_mapping.items():
        degree_list.append((mapped_id, G.in_degree(str(original_id)), G.out_degree(str(original_id))))
        adj_list[mapped_id] = []

        # Find all edges that contain original_id as the source node
        edges = G.out_edges(str(original_id), data=True)
        edges = sorted(edges, key=lambda x: x[2]['timestamp'])
        for u, v, d in edges:          
            adj_list[mapped_id].append(node_mapping[int(v)])

    with open(target_dir + 'Degree.dat', 'w') as f:
        for node, in_degree, out_degree in degree_list:
            f.write(f"{node} {in_degree} {out_degree}\n")

    with open(target_dir +  'Graph.dat', 'w') as f:
        #  Write the overall number of nodes first
        f.write(f"{len(G.nodes())}\n")

        for node, neighbors in adj_list.items():
            f.write(f"{node} {' '.join([str(neighbor) for neighbor in neighbors])}\n")


"""
STExa Dataset Format (Undirected): from_id to_id, the first line is the number of nodes and edges
1. Read the original attributed dataset and the node mapping file
2. Convert the network to an undirected graph and remove all self-loop edges.
3. For every edge (u, v), ensure the corresponding edge (v, u) is also included.
"""
def get_STExa_dataset(G, dataset_name, node_mapping, target_path):
    G_undir = nx.Graph(G)
    print(f"Graph info after converting to undirected simple graph: nodes: {G_undir.number_of_nodes()}, edges: {G_undir.number_of_edges()}, density: {nx.density(G_undir)}")
    G_undir.remove_edges_from(nx.selfloop_edges(G_undir))
    print(f"Graph info after removing self-loop edges: nodes: {G_undir.number_of_nodes()}, edges: {G_undir.number_of_edges()}, density: {nx.density(G_undir)}")

    target_dir = target_path + f"{dataset_name}/"
    os.makedirs(target_dir, exist_ok=True)

    edge_list = []
    for u, v in G_undir.edges():
        mapped_u, mapped_v = node_mapping[int(u)], node_mapping[int(v)]
        if mapped_u != mapped_v:
            edge_list.extend([(mapped_u, mapped_v), (mapped_v, mapped_u)])
    
    edge_list.sort()

    with open(target_dir + dataset_name + ".txt", 'w') as f:
        f.write(f"{G_undir.number_of_nodes()} {len(edge_list)}\n")
        for u, v in edge_list:
            f.write(f"{u} {v}\n")


"""
Repeeling Dataset Format (Streaming Directed Multigraph with no self-loop):
1. Read the original attributed dataset and the node mapping file
2. The output file format: from_id to_id timestamp
Note that Repeeling algorithm deals with self-loop edges, so we do not need to remove them here.
"""
def get_Repeeling_dataset(G, dataset_name, node_mapping, target_path):

    target_dir = target_path + f"{dataset_name}/"
    os.makedirs(target_dir, exist_ok=True)
    
    edge_list = []
    for u, v, d in G.edges(data=True):
        edge_list.append((node_mapping[int(u)], node_mapping[int(v)], d['timestamp']))

    edge_list = sorted(edge_list, key=lambda x: (x[0], x[1], x[2]))

    with open(target_dir + dataset_name + ".txt", 'w') as f:
        for u, v, timestamp in edge_list:
            f.write(f"{u} {v} {timestamp}\n")


# I2ACSM Dataset Format (Undirected): from_id \t to_id
def get_I2ACSM_dataset(G, dataset_name, target_path):
    G_undir = nx.Graph(G)
    # print(f"Graph info after converting to undirected simple graph: nodes: {G_undir.number_of_nodes()}, edges: {G_undir.number_of_edges()}, density: {nx.density(G_undir)}")

    # Remove self-loop edges
    G_undir.remove_edges_from(nx.selfloop_edges(G_undir))
    # print(f"Graph info after removing self-loop edges: nodes: {G_undir.number_of_nodes()}, edges: {G_undir.number_of_edges()}, density: {nx.density(G_undir)}")

    with open(target_path + dataset_name + "_non_attributed.txt", "w") as f:
        for u, v in G_undir.edges():
            f.write(f"{u}\t{v}\n")


"""
TransZero_LS_GS Dataset Format (Undirected): from_id \t to_id
1. Similar to ST-Exa dataset, but save the edge list as ".edges" file
2. Save query file as ".query" file
"""
def get_TransZero_dataset(G, dataset_name, query_node_path, node_mapping, target_path):
    G_undir = nx.Graph(G)
    print(f"Graph info after converting to undirected simple graph: nodes: {G_undir.number_of_nodes()}, edges: {G_undir.number_of_edges()}, density: {nx.density(G_undir)}")
    G_undir.remove_edges_from(nx.selfloop_edges(G_undir))
    print(f"Graph info after removing self-loop edges: nodes: {G_undir.number_of_nodes()}, edges: {G_undir.number_of_edges()}, density: {nx.density(G_undir)}")

    target_dir = target_path + f"{dataset_name}/"
    os.makedirs(target_dir, exist_ok=True)

    edge_list = []
    for u, v in G_undir.edges():
        mapped_u, mapped_v = node_mapping[int(u)], node_mapping[int(v)]
        if mapped_u != mapped_v:
            edge_list.extend([(mapped_u, mapped_v), (mapped_v, mapped_u)])
    
    edge_list.sort()

    with open(target_dir + dataset_name + ".edges", 'w') as f:
        for u, v in edge_list:
            f.write(f"{u} {v}\n")
    
    # Save the query file as .query file
    query_nodes_file = query_node_path + dataset_name + "_mapped_query_node.txt"
    query_nodes = []
    with open(query_nodes_file, 'r') as f:
        for line in f:
            query_nodes.append(int(line.strip()))
  
    with open(target_dir + dataset_name + ".query", 'w') as f:
        for node in query_nodes:
            f.write(f"{node}\n")



if __name__ == "__main__":
    algo_list =["ALS", "WCF-CRC", "CSD", "ST-Exa", "Repeeling", "I2ACSM", "TransZero_LS_GS"]
    dataset_list = ["BTW17", "Chicago_COVID", "Crawled_Dataset144", "Crawled_Dataset26"]

    source_path = "D:/NTU/Academic/5. Part 2 Experimental Analysis/Code and Datasets/Cohesion_Evaluation/Original_Datasets/Preprocessed_Datasets/"
    query_node_path = "D:/NTU/Academic/5. Part 2 Experimental Analysis/Code and Datasets/Cohesion_Evaluation/Original_Datasets/Query_Nodes/"
    target_path = "D:/NTU/Academic/5. Part 2 Experimental Analysis/Code and Datasets/Cohesion_Evaluation/Input_Datasets/"  # Path to save the transformed datasets
    node_mapping_path = "D:/NTU/Academic/5. Part 2 Experimental Analysis/Code and Datasets/Cohesion_Evaluation/Original_Datasets/Node_Mapping/"

    
    for dataset_name in tqdm.tqdm(dataset_list):
        attribute_file = source_path + dataset_name + "_attributed.txt"
        G = gu.graph_construction(attribute_file)

        for algorithm in tqdm.tqdm(algo_list):
            algo_target_path = target_path + algorithm + "_Dataset/"
            node_mapping = read_node_mapping(node_mapping_path + dataset_name + "_node_mapping.txt")
            
            if algorithm == "ALS":
                get_ALS_dataset(G, dataset_name, algo_target_path)

            elif algorithm == "WCF-CRC":
                if dataset_name == "BTW17":
                    get_CRC_dataset(G, dataset_name, algo_target_path, num_instances = 3)
                else:
                    get_CRC_dataset(G, dataset_name, algo_target_path, num_instances = 10)

            elif algorithm == "CSD":
                get_CSD_dataset(G, dataset_name, node_mapping, algo_target_path)

            elif algorithm == "ST-Exa":
                get_STExa_dataset(G, dataset_name, node_mapping, algo_target_path)

            elif algorithm == "Repeeling":
                 get_Repeeling_dataset(G, dataset_name, node_mapping, algo_target_path)

            elif algorithm == "I2ACSM":
                get_I2ACSM_dataset(G, dataset_name, algo_target_path)

            elif algorithm == "TransZero_LS_GS":
                get_TransZero_dataset(G, dataset_name, query_node_path, node_mapping, algo_target_path)
