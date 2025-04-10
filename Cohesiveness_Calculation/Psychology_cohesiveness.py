"""
This script is used to calculate the psychology-informed cohesiveness for each community
1. Consider the varying time decay speed
2. Consider different time decay functions
"""

import os
from joblib import Parallel, delayed
import tqdm
import General_function as gf


"""
For each dataset,
1. Read the results of the algorithm from its results directory
2. Read the attributed file to get the graph
3. For each file, read the line, split by "\t" and extract the data element of each line
4. Based on the community nodes, calculate the cohesiveness score of the community nodes and save the results
"""
def process_results(algorithm, dataset, results_dir, output_dir, decay_method, value, njobs):
    global attribute_dir, node_mapping_dir

    attribute_file = attribute_dir + dataset + "_attributed.txt"
    node_mapping_file = node_mapping_dir + dataset + "_node_mapping.txt"
    result_file = results_dir + algorithm + "_results_" + dataset + ".txt"
    
    # Dictionary to store the cohesiveness results for each community, in case of duplicate calculation
    cohesiveness_dict = {}

    # Build the graph with original nodes and edges attributes
    tadj_list, latest_timestamp = gf.build_graph(attribute_file)

    # Read the node mapping file
    node_mapping = gf.read_node_mapping(node_mapping_file)

    # Read the results of the algorithm
    if algorithm in ["ALS", "WCF-CRC", "I2ACSM"]:
        results = gf.process_ALS_CRC_I2ACSM_results(result_file)
    elif algorithm in ["CSD", "ST-Exa", "Repeeling"]:
        results = gf.process_CSD_STExa_Repeeling_results(result_file, node_mapping)
    elif algorithm in ["TransZero_LS", "TransZero_GS"]:
        results = gf.process_TransZero_results(result_file, node_mapping)
    print("Sucessfully load the algorithm results!")
    
    # Compose the output file name
    output_file = output_dir + algorithm + "_results_" + dataset + "_" + decay_method + "_" + str(value) + ".txt"

    # Calculate the cohesiveness for each community
    if algorithm in ["ALS", "WCF-CRC", "I2ACSM"]:
        results_with_dicts = Parallel(n_jobs=njobs)(
            delayed(gf.process_ALS_CRC_I2ACSM_item)(node, score, parameter_list, community_node_list, tadj_list, latest_timestamp, value, decay_method, cohesiveness_dict)
            for node, score, parameter_list, community_node_list in tqdm.tqdm(results)
        )
    elif algorithm in ["CSD", "ST-Exa", "Repeeling"]:
        results_with_dicts = Parallel(n_jobs=njobs)(
            delayed(gf.process_CSD_STExa_Repeeling_item)(node, parameter_list, community_node_list, tadj_list, latest_timestamp, value, decay_method, cohesiveness_dict)
            for node, parameter_list, community_node_list in tqdm.tqdm(results)
        )
    elif algorithm in ["TransZero_LS", "TransZero_GS"]:
        results_with_dicts = Parallel(n_jobs=njobs)(
            delayed(gf.process_TransZero_item)(node, community_node_list, tadj_list, latest_timestamp, value, decay_method, cohesiveness_dict)
        for node, community_node_list in tqdm.tqdm(results)
        )
    
    cohesiveness_results, updated_dicts = zip(*results_with_dicts)
    cohesiveness_dict = gf.merge_dicts(updated_dicts)
    
    with open(output_file, 'a') as f:
        f.writelines(cohesiveness_results)
    print(f"Sucessfully write {output_file}!")



"""
Calculate the psychology-informed cohesiveness for each algorithm's results
"""
def cohesiveness_calculation(algorithm, dataset_list, njobs):
    global algo_results_dir, cohesiveness_dir

    # Directory to access the algorithm results
    algo_result_dir = algo_results_dir + algorithm + "_Results/"
    # Directory to store the psychology-informed cohesiveness results
    algo_cohesiveness_dir = cohesiveness_dir + algorithm + "_Results/"
    if not os.path.exists(algo_cohesiveness_dir):
        os.makedirs(algo_cohesiveness_dir)

    # Define the tasks to be executed in parallel
    tasks = []
    for dataset_name in dataset_list:
        for decay_method in ['exp', 'poly']:
            if decay_method == 'exp':
                lambda_value_list = [0.0001, 0.0005, 0.001, 0.005, 0.01]
                for lambda_value in lambda_value_list:
                    tasks.append((algorithm, dataset_name, algo_result_dir, algo_cohesiveness_dir, decay_method, lambda_value, njobs))
            elif decay_method == 'poly':
                mu_value_list = [0.5, 1, 1.5, 2]
                for mu_value in mu_value_list:
                    tasks.append((algorithm, dataset_name, algo_result_dir, algo_cohesiveness_dir, decay_method, mu_value, njobs))

    # Execute the tasks in parallel
    Parallel(n_jobs=njobs)(delayed(process_results)(*task) for task in tasks)

if __name__ == "__main__":
    attribute_dir = "D:/Cohesion_Evaluation/Original_Datasets/Preprocessed_Datasets/"
    node_mapping_dir = "D:/Cohesion_Evaluation/Original_Datasets/Node_Mapping/"
    algo_results_dir = "D:/Cohesion_Evaluation/Algorithm_Output/"
    cohesiveness_dir = "D:/Cohesion_Evaluation/Cohesiveness_Output/"

    algo_list =["ALS", "WCF-CRC", "CSD", "ST-Exa", "Repeeling", "I2ACSM", "TransZero_LS", "TransZero_GS"]
    njobs = 1

    for algorithm in algo_list:
        if algorithm == "Repeeling":
            cohesiveness_calculation(algorithm, ["BTW17", "Chicago_COVID"], njobs)
        else:
            cohesiveness_calculation(algorithm, ["BTW17", "Chicago_COVID", "Crawled_Dataset26", "Crawled_Dataset144"], njobs)