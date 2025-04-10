"""
This script is used to transform the original preprocessed datasets into the format that required by each algorithms, where the node mapping is also included.
"""

import os
import networkx as nx
import sys

target_path = "./"
sys.path.append(target_path)
import Cohesiveness_Calculation.Utils.Graph_utils as gu


def read_node_mapping(node_mapping_file):
    node_mapping = {}
    with open(node_mapping_file, "r") as f:
        lines = f.readlines()
        node_mapping = {int(line.split("\t")[0]): int(line.split("\t")[1]) for line in lines}
    
    return node_mapping


# from_id \t to_id \t timestamp
def get_ALS_dataset(G, dataset_name, target_path):
    with open(target_path + dataset_name + "_timestamp.txt", "w") as f:
        for u, v, d in G.edges(data=True):
            f.write(str(u) + "\t" + str(v) + "\t" + str(d["timestamp"]) + "\n")


def get_I2ACSM_dataset(G, dataset_name, target_path):
    G_undir = nx.Graph(G)
    print(f"Graph info after converting to undirected simple graph: nodes: {G_undir.number_of_nodes()}, edges: {G_undir.number_of_edges()}, density: {nx.density(G_undir)}")

    # Remove self-loop edges
    G_undir.remove_edges_from(nx.selfloop_edges(G_undir))
    print(f"Graph info after removing self-loop edges: nodes: {G_undir.number_of_nodes()}, edges: {G_undir.number_of_edges()}, density: {nx.density(G_undir)}")

    with open(target_path + dataset_name + "_non_attributed.txt", "w") as f:
        for u, v in G_undir.edges():
            f.write(f"{u}\t{v}\n")

"""
This function is to generate the data needed for the CRC algorithm, following are the rules for generating the data:
1. Read the attributed graph dataset
2. Generate graph instances: 
    (1) First sort the edges by chronological order
    (2) Then divide them into X partitions
    i.e., |E|/|T| edges, where |T| is the target number of graph instances. 
    It guarantees each graph instance contains meaningful k-core components.
3. Calculate the edge weight:
    (1) The edge weight is calculated from the interaction frequency. 
    (2) The edge weight of all the datasets is normalized to [0, 1] by min-max normalization.

    Note that the frequency is calculated in each partition and the min-max normalization is performed on the entire dataset.
"""
def get_CRC_dataset(G, dataset_name, target_path, num_instances = 10):
    print(f"**********{dataset_name}**********")
    
    # Split the edges into X partitions
    edges = list(G.edges(data=True))
    edges = sorted(edges, key=lambda x: x[2]['timestamp'])

    num_edges = len(edges)
    num_edges_per_instance = num_edges // num_instances

    edge_partitions = []
    for i in range(num_instances):
        edge_partitions.append(edges[i * num_edges_per_instance: (i + 1) * num_edges_per_instance])

    # Add the remaining edges to the last partition
    if num_edges % num_instances != 0:
        edge_partitions[-1].extend(edges[num_instances * num_edges_per_instance:])

    print(f"Number of edges in each partition: {[len(partition) for partition in edge_partitions]}")

    # Build the graph instances and check the maximum k-core components manually
    graph_instances = {}
    for i, edge_partition in enumerate(edge_partitions):
        G_instance = nx.MultiDiGraph()
        for edge in edge_partition:
            G_instance.add_edge(edge[0], edge[1], timestamp=edge[2]['timestamp'])
        
        print(f"Graph instance {i}: nodes: {G_instance.number_of_nodes()}, edges: {G_instance.number_of_edges()}, density: {nx.density(G_instance)}")

        G_instance_undirected = nx.Graph()
        
        # Tranfer the graph instance to undirected graph but use the edge number between two nodes as the weight of undirected edge
        for edge in G_instance.edges():
            if not G_instance_undirected.has_edge(edge[0], edge[1]):
                freq = len(G_instance[edge[0]][edge[1]]) 

                if G_instance.has_edge(edge[1], edge[0]):
                    freq +=  len(G_instance[edge[1]][edge[0]])
                G_instance_undirected.add_edge(edge[0], edge[1], weight=freq)

        # Check the k-core components of the graph instance
        core_number = 2 

        # Remove self-loop edges
        G_instance_undirected.remove_edges_from(nx.selfloop_edges(G_instance_undirected)) # Our simplified dataset has self-loop edges, but CRC algorithm does not consider self-loop edges
        
        k_core = nx.k_core(G_instance_undirected, k=core_number)

        print(f"{core_number}-core info: nodes: {k_core.number_of_nodes()}, edges: {k_core.number_of_edges()} \n")

        # Save the graph instance
        graph_instances[f"graph_instance_{i}"] = G_instance_undirected

    # Add edge weights to the graph instances according to the interaction frequency
    min_freq, max_freq = 1, 1

    # Iterate through all the graph instances to calculate the min and max frequency
    for i in range(num_instances):
        G_instance = graph_instances[f"graph_instance_{i}"]
        
        # Get the weight of each edge as a list
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
Generate CSD dataset
1. Read the original attributed dataset from the source path, read the node mapping file
2. Transfer the network to a directed graph, since the paper requires the directed graph
3. Generate two file for each dataset use node mapping:
    (1) File recording every node's in degree and out degree: DatasetName_Degree.dat
        Format: node_id, in_degree, out_degree
    (2) File recording the graph in a adjacent format: DatasetName_Graph.dat
        Format: node1 node2 node3 ...
"""
def get_CSD_dataset(G, dataset_name, node_mapping_path, target_path):
    G_dir = nx.DiGraph(G)
    print(f"Graph info after converting to directed graph: nodes: {G_dir.number_of_nodes()}, edges: {G_dir.number_of_edges()}, density: {nx.density(G_dir)}")

    # Read the node mapping file
    node_mapping = read_node_mapping(node_mapping_path + dataset_name + "_node_mapping.txt")

    target_dir = target_path + f"{dataset_name}/" 
    os.makedirs(target_dir, exist_ok=True)

    # Generate the degree file, order by the node id
    degree_list = []
    for node in G_dir.nodes():
        mapped_node = node_mapping[node]
        degree_list.append((mapped_node, G_dir.in_degree(node), G_dir.out_degree(node)))
    
    degree_list = sorted(degree_list, key=lambda x: x[0])
    with open(target_dir + 'Degree.dat', 'w') as f:
        for node, in_degree, out_degree in degree_list:
            f.write(f"{node} {in_degree} {out_degree}\n")
    
    # Generate the graph file, each line only stores the node id followed by its neighbors
    adj_list = {}
    for (node, _, _) in degree_list:
        neighbors = list(G_dir.neighbors(node))
        neighbors = [node_mapping[neighbor] for neighbor in neighbors]
        adj_list[node] = neighbors
    
    with open(target_dir +  'Graph.dat', 'w') as f:
        #  Write the overall number of nodes first
        f.write(f"{len(G_dir.nodes())}\n")

        for node, neighbors in adj_list.items():
            f.write(f"{node} {' '.join([str(neighbor) for neighbor in neighbors])}\n")



"""
Generate STExa dataset
1. Read the original attributed dataset from the source path, read the node mapping file
2. Transfer the network to an undirected simple graph, since the paper requires the undirected simple graph
3. For each edge, if (u, v) is an edge, then (v, u) is also an edge
4. Remove the self-loop edges
5. The output file format: from_id to_id, the first line is the number of nodes and edges
"""
# The graph is undirected simple graph
def get_STExa_dataset(G, dataset_name, node_mapping_path, target_path):
    G_undir = nx.Graph(G)
    print(f"Graph info after converting to undirected simple graph: nodes: {G_undir.number_of_nodes()}, edges: {G_undir.number_of_edges()}, density: {nx.density(G_undir)}")

    # Remove self-loop edges
    G_undir.remove_edges_from(nx.selfloop_edges(G_undir))
    print(f"Graph info after removing self-loop edges: nodes: {G_undir.number_of_nodes()}, edges: {G_undir.number_of_edges()}, density: {nx.density(G_undir)}")

    # Read the node mapping file
    node_mapping = read_node_mapping(node_mapping_path + dataset_name + "_node_mapping.txt")
    
    target_dir = target_path + f"{dataset_name}/"
    os.makedirs(target_dir, exist_ok=True)

    edge_list = []
    for u, v in G_undir.edges():
        u = node_mapping[u]
        v = node_mapping[v]
        if u != v:
            edge_list.append((u, v))
            edge_list.append((v, u))
    
    edge_list = sorted(edge_list, key=lambda x: (x[0], x[1]))

    with open(target_dir + dataset_name + ".txt", 'w') as f:
        #first line: number of nodes and edges
        f.write(f"{G_undir.number_of_nodes()} {len(edge_list)}\n")
        # from_id to_id
        for u, v in edge_list:
            f.write(f"{u} {v}\n")



"""
Generate the Repeeling dataset
1. Read the attributed version of the dataset from the source path, and read the node mapping file from the node mapping directory
2. The output file format: from_id to_id timestamp
"""
def get_Repeeling_dataset(G, dataset_name, node_mapping_path, target_path):
    # Read the node mapping file
    node_mapping = read_node_mapping(node_mapping_path + dataset_name + "_node_mapping.txt")

    target_dir = target_path + f"{dataset_name}/"
    os.makedirs(target_dir, exist_ok=True)
    
    edge_list = []
    for u, v, d in G.edges(data=True):
        edge_list.append((node_mapping[u], node_mapping[v], d['timestamp']))

    edge_list = sorted(edge_list, key=lambda x: (x[0], x[1], x[2]))

    with open(target_dir + dataset_name + ".txt", 'w') as f:
        for u, v, timestamp in edge_list:
            f.write(f"{u} {v} {timestamp}\n")


"""
Get the dataset for the TransZero_LS_GS algorithm
1. Similar to ST-Exa dataset, but save the edge list as ".edges" file
2. Save query file as ".query" file
"""

def get_TransZero_dataset(G, dataset_name, query_node_path, node_mapping_path, target_path):
    G_dir = nx.Graph(G)
    print(f"Graph info after converting to undirected simple graph: nodes: {G_dir.number_of_nodes()}, edges: {G_dir.number_of_edges()}, density: {nx.density(G_dir)}")

    # Remove self-loop edges
    G_dir.remove_edges_from(nx.selfloop_edges(G_dir))
    print(f"Graph info after removing self-loop edges: nodes: {G_dir.number_of_nodes()}, edges: {G_dir.number_of_edges()}, density: {nx.density(G_dir)}")

    # Read the node mapping file
    node_mapping = read_node_mapping(node_mapping_path + dataset_name + "_node_mapping.txt")
    
    target_dir = target_path + f"{dataset_name}/"
    os.makedirs(target_dir, exist_ok=True)

    edge_list = []
    for u, v in G_dir.edges():
        u = node_mapping[u]
        v = node_mapping[v]
        if u != v:
            edge_list.append((u, v))
            edge_list.append((v, u))
    
    edge_list = sorted(edge_list, key=lambda x: (x[0], x[1]))

    with open(target_dir + dataset_name + ".edges", 'w') as f:
        # from_id to_id
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

    # Set for the source path
    source_path = "D:/Cohesion_Evaluation/Original_Datasets/Preprocessed_Datasets/"

    # Set for the query node path
    query_node_path = "D:/Cohesion_Evaluation/Original_Datasets/Query_Nodes/"

    # Set for saving the transformed datasets
    target_path = "D:/Cohesion_Evaluation/Input_Datasets/"

    # Path to access the node mapping file
    node_mapping_path = "D:/Cohesion_Evaluation/Original_Datasets/Node_Mapping/"

    
    for dataset_name in dataset_list:
        # Read the original attributed dataset
        attribute_file = source_path + dataset_name + "_attributed.txt"
        G = gu.graph_construction(attribute_file)

        for algorithm in algo_list:
            target_path = target_path + algorithm + "_Dataset/"
            
            if algorithm == "I2ACSM":
                get_I2ACSM_dataset(G, dataset_name, target_path)
            
            elif algorithm == "ALS":
                get_ALS_dataset(G, dataset_name, target_path)

            elif algorithm == "WCF-CRC":
                if dataset_name == "BTW17":
                    get_CRC_dataset(G, dataset_name, target_path, num_instances = 3)
                else:
                    get_CRC_dataset(G, dataset_name, target_path, num_instances = 10)
            
            elif algorithm == "CSD":
                get_CSD_dataset(G, dataset_name, node_mapping_path, target_path)

            elif algorithm == "ST-Exa":
                get_STExa_dataset(G, dataset_name, node_mapping_path, target_path)
            
            elif algorithm == "Repeeling":
                 get_Repeeling_dataset(G, dataset_name, node_mapping_path, target_path)

            elif algorithm == "TransZero_LS_GS":
                get_TransZero_dataset(G, dataset_name, query_node_path, node_mapping_path, target_path)
