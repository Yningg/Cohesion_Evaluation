"""
Calculate the structural cohesiveness of CS results, including the diameter, size, and minimum degree.
"""

import os
import networkx as nx
import numpy as np
import tqdm
from joblib import Parallel, delayed
import Cohesiveness_Calculation.Utils.Process_algo as pa


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
        # Only calculate for the community that has more than 1 node
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


def get_network_results(G, grouped_results, dim_index, n_jobs=-1):
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
  
  
    for dataset, file in file_list.items():
        attribute_file = attribute_dir + dataset + "_attributed.txt"
        node_mapping_file = node_mapping_dir + dataset + "_node_mapping.txt"
        node_mapping = pa.read_node_mapping(node_mapping_file)

        # Build the graph with original nodes and edges attributes
        G = nx.read_edgelist(attribute_file, nodetype=str, data=(('timestamp', str), ('sentiment', str)), create_using=nx.MultiGraph()) # type: ignore

        # Read the results
        if algorithm in ["ALS", "WCF-CRC", "I2ACSM"]:
            results = pa.process_ALS_CRC_I2ACSM_results(results_dir + file)
        elif algorithm in ["CSD", "ST-Exa", "Repeeling"]:
            results = pa.process_CSD_STExa_Repeeling_results(results_dir + file, node_mapping)
        elif algorithm in ["TransZero_LS", "TransZero_GS"]:
            results = pa.process_TransZero_results(results_dir + file, node_mapping)
        
        print(f"Number of results: {len(results)}")
        
        # Group the results according to the query node
        grouped_results = group_results(results)
        print(f"Number of grouped results: {len(grouped_results)}")
    
        # Calculate the structural cohesiveness
        condensed_results, empty_num = get_network_results(G, grouped_results, dim_index)
        print(f"Number of condensed results: {len(condensed_results)}, Number of empty results: {empty_num}")

        # Average the results for communities searched by the same query node
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