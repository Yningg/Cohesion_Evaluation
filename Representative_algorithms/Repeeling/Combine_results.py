"""
Combine the results of STExa on different datasets into separate single files
"""

import os
import tqdm


def get_community_list(file):
    community_node = set()
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            source, target = line.strip().split(" ")
            community_node.add(int(source))
            community_node.add(int(target))
    community_node_list = list(community_node)
    return community_node_list

def read_results(result_dir, files):
    results = []
    for file in tqdm.tqdm(files):
        # Parse the file name to extract the query node, lower bound and upper bound
        query_node, window_size, slide_size, k_c, k_f = file[:-4].split("_")
        community_node_list = get_community_list(result_dir + file)
        results.append((int(query_node), [int(window_size), int(slide_size), int(k_c), int(k_f)], community_node_list))
    return results


if __name__ == "__main__":
    # Path to the directory containing the results
    result_dir = "D:/Algorithm_Output/Repeeling_Results/"
    dataset_list = ["BTW17", "Chicago_COVID"]

    # For each dataset, combine the results into a single file
    for dataset in dataset_list:
        print(f"Processing the dataset: {dataset}")

        dataset_result_dir = result_dir + dataset + "/"
        result_file = result_dir + "Repeeling_results_" + dataset + ".txt"

        # Read all files in the directory
        files = os.listdir(dataset_result_dir)
        results = read_results(dataset_result_dir, files)
        
        # Save the results into result file
        with open(result_file, 'w') as f:
            for result in results:
                query_node, params, community_node_list = result
                f.write(f"{query_node}\t{params}\t{community_node_list}\n")