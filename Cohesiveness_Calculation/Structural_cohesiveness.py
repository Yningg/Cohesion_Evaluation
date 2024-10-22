"""
This script is used to evaluate the diameter, size, and minimum degree of CS results.
"""
import ast
import os
import networkx as nx
import numpy as np
import tqdm
from joblib import Parallel, delayed


# Build the graph with the original nodes and edges attributes
def graph_construction(attribute_file):
    # Read the attribute file and add the attributes to the graph
    attributed_G = nx.read_edgelist(attribute_file, nodetype=str, data=(('timestamp', str), ('sentiment', str)), create_using=nx.MultiGraph()) # type: ignore
    
    print(f"Original graph info: {attributed_G.number_of_nodes()} nodes, {attributed_G.number_of_edges()} edges, density: {nx.density(attributed_G)}")

    return attributed_G


# Read the node mapping file
def read_node_mapping(node_mapping_file):
    node_mapping = {}
    with open(node_mapping_file, 'r') as f:
        lines = f.readlines()
        # Reverse the mapping to map the new nodes back to the original nodes
        node_mapping = {int(line.split("\t")[1]): int(line.split("\t")[0]) for line in lines}
    
    return node_mapping


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
            community_node_list = [node_mapping[node] for node in community_node_list]

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
            community_node_list = [node_mapping[node] for node in community_node_list]
            results.append([node, community_node_list])

    return results

# Group the results according to the query node: 100 query node --> 100 groups
def group_results(results):
    grouped_results = {}
    
    for result in results:
        node = result[0]
        if node in grouped_results:
            grouped_results[node].append(result)
        else:
            grouped_results[node] = [result]
    return grouped_results


def process_node(G, node, result_list, dim_index):
    network_stats = []
    valid_count = 0

    for result in result_list:
        # Only calculate the stats for the community that has more than 1 node
        if len(result[dim_index]) > 0:
            subgraph = G.subgraph(result[dim_index])
            if nx.is_connected(subgraph):
                network_stats.append([nx.diameter(subgraph), nx.number_of_nodes(subgraph), min(dict(subgraph.degree()).values())])
                valid_count += 1

    if valid_count == 0:
        return node, None
    else:
        network_stats = np.mean(network_stats, axis=0)
        return node, list(network_stats)

def get_network_results(G, grouped_results, dim_index, n_jobs=1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_node)(G, node, result_list, dim_index) for node, result_list in tqdm.tqdm(grouped_results.items())
    )

    condensed_results = {}
    empty_result_num = 0

    for node, stats in results: # type: ignore
        if stats is None:
            empty_result_num += 1
        else:
            condensed_results[node] = stats

    return condensed_results, empty_result_num


def output_network_stats(algorithm, results_dir, dataset_list, dim_index):
    global attribute_dir, node_mapping_dir

    # Extract files according to the dataset name
    file_list = {}
    for file in os.listdir(results_dir):
        for dataset_name in dataset_list:
            if dataset_name in file:
                file_list[dataset_name] = file
  
    # Read the results
    for dataset, file in file_list.items():
        attribute_file = attribute_dir + dataset + "_attributed.txt"
        node_mapping_file = node_mapping_dir + dataset + "_node_mapping.txt"

        # Build the graph with original nodes and edges attributes
        G = graph_construction(attribute_file)

        # Read the node mapping file
        node_mapping = read_node_mapping(node_mapping_file)

        # Read the results
        if algorithm in ["ALS", "WCF-CRC", "I2ACSM"]:
            results = process_ALS_CRC_I2ACSM_results(results_dir + file)
        elif algorithm in ["CSD", "ST-Exa", "Repeeling"]:
            results = process_CSD_STExa_Repeeling_results(results_dir + file, node_mapping)
        elif algorithm in ["TransZero_LS", "TransZero_GS"]:
            results = process_TransZero_results(results_dir + file, node_mapping)
        
        
        print(f"Number of results: {len(results)}")
        
        # Group the results refer to the query node
        grouped_results = group_results(results)
        print(f"Number of grouped results: {len(grouped_results)}")
    
        # Calculate the community stats for each query node
        condensed_results, empty_num = get_network_results(G, grouped_results, dim_index)
        print(f"Number of condensed results: {len(condensed_results)}, Number of empty results: {empty_num}")

        # Calculate the average query node stats
        if empty_num == 100:
            print("No valid results")
            continue
        else:
            average_results = np.mean(list(condensed_results.values()), axis=0)

            print(f"{dataset}, {algorithm}: {average_results[0]:.2f} & {average_results[1]:.2f} & {average_results[2]:.2f} & {len(condensed_results)}")
    
        
def process(algorithm, dataset_list, dim_index):
    global algo_results_dir
    algo_result_dir = algo_results_dir + algorithm + "_Results/"
    output_network_stats(algorithm, algo_result_dir, dataset_list, dim_index)


if __name__ == "__main__":
    attribute_dir = "D:/Cohesion_Evaluation/Original_Datasets/Preprocessed_Datasets/"
    algo_results_dir = "D:/Cohesion_Evaluation/Algorithm_Output/"
    node_mapping_dir = "D:/Cohesion_Evaluation/Original_Datasets/Node_Mapping/"

    algo_index ={"ALS": 3, "WCF-CRC": 3, "CSD": 2, "ST-Exa": 2, "Repeeling": 2, "I2ACSM": 3, "TransZero_LS": 1, "TransZero_GS": 1}

    for algorithm, dim_index in algo_index.items():
        if algorithm == "Repeeling":
            process(algorithm, ["BTW17", "Chicago_COVID"], dim_index)
        else:
            process(algorithm, ["BTW17", "Chicago_COVID", "Crawled_Dataset26", "Crawled_Dataset144"], dim_index)