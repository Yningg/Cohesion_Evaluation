import ast
from joblib import Parallel, delayed
import tqdm


import sys
target_path = "./"
sys.path.append(target_path)
import Cohesiveness_Calculation.Utils.Graph_utils as gu
import Cohesiveness_Calculation.Utils.Cohesiveness_score as cs


# Read the node mapping file
def read_node_mapping(node_mapping_file):
    node_mapping = {}
    with open(node_mapping_file, 'r') as f:
        lines = f.readlines()
        # Reverse the mapping to map the new nodes back to the original nodes
        node_mapping = {int(line.split("\t")[1]): int(line.split("\t")[0]) for line in lines}
    
    return node_mapping


# Merge the dictionaries in the list into a single dictionary
def merge_dicts(dict_list):
    merged_dict = {}
    for d in dict_list:
        merged_dict.update(d)
    return merged_dict


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


"""
Functions to calculate the psychology-informed cohesiveness for each algorithm's results
"""
def cal_ALS_CRC_I2ACSM_item(node, score, parameter_list, community_node_list, tadj_list, latest_timestamp, value, decay_method, cohesiveness_dict):
    if len(community_node_list) == 0:
        cohesiveness = ['Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid']
    else:
        sorted_community = tuple(sorted(community_node_list))
        if sorted_community in cohesiveness_dict:
            cohesiveness = cohesiveness_dict[sorted_community]
        else:
            tadj_sublist = gu.build_subtadj(tadj_list, community_node_list, latest_timestamp)
            cohesiveness = cs.cohesiveness_dim(tadj_list, tadj_sublist, latest_timestamp, value, decay_method)
            cohesiveness_dict[sorted_community] = cohesiveness
    
    return f"{node}\t{score}\t{parameter_list}\t{community_node_list}\t{cohesiveness}\n", cohesiveness_dict


def cal_CSD_STExa_Repeeling_item(node, parameter_list, community_node_list, tadj_list, latest_timestamp, value, decay_method, cohesiveness_dict):
    if len(community_node_list) == 0:
        cohesiveness = ['Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid']
    else:
        sorted_community = tuple(sorted(community_node_list))
        if sorted_community in cohesiveness_dict:
            cohesiveness = cohesiveness_dict[sorted_community]
        else:
            tadj_sublist = gu.build_subtadj(tadj_list, community_node_list, latest_timestamp)
            cohesiveness = cs.cohesiveness_dim(tadj_list, tadj_sublist, latest_timestamp, value, decay_method)
            cohesiveness_dict[sorted_community] = cohesiveness
    
    return f"{node}\t{parameter_list}\t{community_node_list}\t{cohesiveness}\n", cohesiveness_dict


def cal_TransZero_item(node, community_node_list, tadj_list, latest_timestamp, value, decay_method, cohesiveness_dict):
    if len(community_node_list) == 0:
        cohesiveness = ['Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid']
    else:
        sorted_community = tuple(sorted(community_node_list))
        if sorted_community in cohesiveness_dict:
            cohesiveness = cohesiveness_dict[sorted_community]
        else:
            tadj_sublist = gu.build_subtadj(tadj_list, community_node_list, latest_timestamp)
            cohesiveness = cs.cohesiveness_dim(tadj_list, tadj_sublist, latest_timestamp, value, decay_method)
            cohesiveness_dict[sorted_community] = cohesiveness
    
    return f"{node}\t{community_node_list}\t{cohesiveness}\n", cohesiveness_dict



"""
Calculate the psychology-informed cohesiveness for each algorithm's results
1. Read datasets, node mapping and algorithm results from the corresponding directories
2. Calculate the cohesiveness scores for algorithm results and save them to the output directory
"""
def cal_results(algorithm, decay_method, value, attribute_file, node_mapping_file, result_file, output_file, n_jobs):
  
    cohesiveness_dict = {}

    # Build the graph with original nodes and edges attributes
    tadj_list, latest_timestamp = gu.build_tadj(attribute_file)
    node_mapping = read_node_mapping(node_mapping_file)

    # Read the results of the algorithm
    if algorithm in ["ALS", "WCF-CRC", "I2ACSM"]:
        results = process_ALS_CRC_I2ACSM_results(result_file)
    elif algorithm in ["CSD", "ST-Exa", "Repeeling"]:
        results = process_CSD_STExa_Repeeling_results(result_file, node_mapping)
    elif algorithm in ["TransZero_LS", "TransZero_GS"]:
        results = process_TransZero_results(result_file, node_mapping)

    # Calculate the cohesiveness for each community
    if algorithm in ["ALS", "WCF-CRC", "I2ACSM"]:
        results_with_dicts = Parallel(n_jobs=n_jobs)(
            delayed(cal_ALS_CRC_I2ACSM_item)(node, score, parameter_list, community_node_list, tadj_list, latest_timestamp, value, decay_method, cohesiveness_dict)
            for node, score, parameter_list, community_node_list in tqdm.tqdm(results)
        )
    elif algorithm in ["CSD", "ST-Exa", "Repeeling"]:
        results_with_dicts = Parallel(n_jobs=n_jobs)(
            delayed(cal_CSD_STExa_Repeeling_item)(node, parameter_list, community_node_list, tadj_list, latest_timestamp, value, decay_method, cohesiveness_dict)
            for node, parameter_list, community_node_list in tqdm.tqdm(results)
        )
    elif algorithm == "TransZero_LS":
        results_with_dicts = Parallel(n_jobs=n_jobs)(
            delayed(cal_TransZero_item)(node, community_node_list, tadj_list, latest_timestamp, value, decay_method, cohesiveness_dict)
        for node, community_node_list in tqdm.tqdm(results)
        )
    
    cohesiveness_results, updated_dicts = zip(*results_with_dicts)
    cohesiveness_dict = merge_dicts(updated_dicts)
    
    with open(output_file, 'a') as f:
        f.writelines(cohesiveness_results)
    print(f"Successfully write {output_file}!")