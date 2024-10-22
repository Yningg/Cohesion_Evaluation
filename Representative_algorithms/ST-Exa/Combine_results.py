"""
Combine the results of STExa on different datasets into seperate single files
"""

import os
import ast
import tqdm


def get_community_list(line):
    parts = line.strip().split(" ", 1)
    community_node_list = parts[1][1:-1].split(" ")[:-1]
    community_node_list = [int(node) for node in community_node_list]
    return community_node_list

def read_results(result_dir, files):
    results = []
    for file in tqdm.tqdm(files):
        # Parse the file name to extract the query node, lower bound and upper bound
        query_node, lower_bound, upper_bound = file[:-4].split("_")

        with open(result_dir + file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                community_node_list = get_community_list(line)
                results.append((int(query_node), [int(lower_bound), int(upper_bound)], community_node_list))
    return results


if __name__ == "__main__":
    # Path to the directory containing the results
    result_dir = "D:/Cohesion_Evaluation/Algorithm_Output/ST-Exa_Results/"
    dataset_list = ["BTW17", "Chicago_COVID", "Crawled_Dataset144", "Crawled_Dataset26"]

    # For each dataset, combine the results into a single file
    for dataset in dataset_list:
        print(f"Processing the dataset: {dataset}")

        dataset_result_dir = result_dir + dataset + "/"
        result_file = result_dir + "ST-Exa_results_" + dataset + ".txt"

        # Read all files in the directory
        files = os.listdir(dataset_result_dir)
        results = read_results(dataset_result_dir, files)
        
        # Save the results into result file
        with open(result_file, 'w') as f:
            for result in results:
                query_node, bounds, community_node_list = result
                f.write(f"{query_node}\t{bounds}\t{community_node_list}\n")