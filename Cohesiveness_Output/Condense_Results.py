"""
This script is to condense the results of the all algorithms into a single file for easy comparison.
Output: each query node and its cohesiveness score
Format: {query node: cohesiveness score}

Steps:
1. Read the results of each algorithm
2. Group the results refer to the query node
3. If there are multiple communities for a query node, take the average of cohesiveness value
4. Save the results in a single txt file
"""



import os
import ast
import numpy as np


def parse_cohesiveness_dim(cohesiveness_str):
    if "Invalid" in cohesiveness_str:
        return ast.literal_eval(cohesiveness_str)
    else:
        # Remove the brackets and split the string by commas
        cohesiveness_str = cohesiveness_str.strip('[]')
        parts = cohesiveness_str.split(',')
        
        # Convert each part to a float, handling 'nan' values
        cohesiveness_dim = []
        for part in parts:
            part = part.strip()
            if part in ['nan', 'inf', '-inf']:
                cohesiveness_dim.append(0)
            else:
                cohesiveness_dim.append(float(part))
        
        return cohesiveness_dim

def group_results(results):
    grouped_results = {}
    for result in results:
        node = result[0]
        if node in grouped_results:
            grouped_results[node].append(result)
        else:
            grouped_results[node] = [result]
    return grouped_results


def get_condensed_results(grouped_results, dim_index):
    condensed_results = {}
    for node, result_list in grouped_results.items():
        cohesiveness_score = []
        valid_count = 0

        for result in result_list:
            if len(result[dim_index]) > 0 and result[dim_index][0] != 'Invalid':
                cohesiveness_score.append(result[dim_index])
                valid_count += 1
        
        if valid_count == 0:
            cohesiveness_score = ['Invalid'] * 5
        else:
            cohesiveness_avg = np.mean(cohesiveness_score, axis=0)
            cohesiveness_std = np.std(cohesiveness_score, axis=0)
            condensed_results[node] = {"avg": list(cohesiveness_avg), "std": list(cohesiveness_std)}

    return condensed_results

    

def process_algo_results(algo_result_dir, algorithm, file):
    results = []
    with open(algo_result_dir + file, 'r') as f:
        lines = f.readlines()
        if algorithm in ["ALS", "WCF-CRC", "I2ACSM"]:
            for line in lines:
                parts = line.strip().split("\t")
                node = int(parts[0])
                value = float(parts[1])
                params = ast.literal_eval(parts[2])
                community_node_list = list(ast.literal_eval(parts[3]))
                community_node_list = [str(node) for node in community_node_list]
                cohesiveness_dim = parse_cohesiveness_dim(parts[4])
                results.append([node, value, params, community_node_list, cohesiveness_dim])
        
        elif algorithm in ["ST-Exa", "CSD", "Repeeling"]:
            for line in lines:
                parts = line.strip().split("\t")
                node = int(parts[0])
                params = ast.literal_eval(parts[1])
                community_node_list = ast.literal_eval(parts[2])
                cohesiveness_dim = parse_cohesiveness_dim(parts[3])
                results.append([node, params, community_node_list, cohesiveness_dim])

        elif algorithm in ["TransZero_LS", "TransZero_GS"]:
            for line in lines:
                parts = line.strip().split("\t")
                node = int(parts[0])
                community_node_list = ast.literal_eval(parts[1].strip())
                cohesiveness_dim = parse_cohesiveness_dim(parts[2])
                results.append([node, community_node_list, cohesiveness_dim])
    return results


def process_dataset(algorithm, cohesiveness_index, dataset_list):
    algo_result_dir = result_dir + algorithm + "_results/"
    files = [file for file in os.listdir(algo_result_dir) if any(name in file for name in dataset_list)]

    # For four files in the directory
    for file in files:
        print(f"Processing {file}...")

        # Read the results
        dataset_name = [name for name in dataset_list if name in file][0]
        results = process_algo_results(algo_result_dir, algorithm, file)
        print(f"Number of results: {len(results)}")

        # Group the results refer to the query node
        grouped_results = group_results(results)
        print(f"Number of grouped results: {len(grouped_results)}")
    
        # Calculate the cohesiveness score for each query node
        condensed_results = get_condensed_results(grouped_results, cohesiveness_index)
    
        # Save the results in a single txt file
        output_file = output_dir + dataset_name + "/" + file.split(".txt")[0] + "_condensed.txt"
        with open(output_file, 'w') as f:
            for node, cohesiveness_score in condensed_results.items():
                f.write(f"{node}\t{cohesiveness_score['avg']}\t{cohesiveness_score['std']}\n")


# Output directory for the condensed results
output_dir = "D:/Cohesion_Evaluation/Cohesiveness_Output/"
result_dir = "D:/Cohesion_Evaluation/Cohesiveness_Output/"

dataset_list = ["BTW17", "Chicago_COVID", "Crawled_Dataset26", "Crawled_Dataset144"]
dataset_sublist = ["BTW17", "Chicago_COVID"]

algorithm_dict = {
    "ALS": {"index":4, "dataset_list": dataset_list},
    "WCF-CRC": {"index":4, "dataset_list": dataset_list},
    "CSD": {"index":3, "dataset_list": dataset_list},
    "ST-Exa": {"index":3, "dataset_list": dataset_list},
    "Repeeling": {"index":3, "dataset_list": dataset_sublist},
    "I2ACSM": {"index":4, "dataset_list": dataset_list},
    "TransZero_LS": {"index":2, "dataset_list": dataset_list},
    "TransZero_GS": {"index":2, "dataset_list": dataset_list}
}

# Deal with results
for algorithm, content in algorithm_dict.items():
    process_dataset(algorithm, content["index"], content["dataset_list"])
