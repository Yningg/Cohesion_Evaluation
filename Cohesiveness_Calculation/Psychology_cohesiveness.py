"""
This script is used to calculate the psychology-informed cohesiveness for each community
1. Consider the varying time decay speed
2. Consider different time decay functions
"""

import os
from joblib import Parallel, delayed
import General_function as gf


"""
Calculate the psychology-informed cohesiveness for each algorithm's results
"""
def cohesiveness_calculation(algorithm, dataset_list, njobs):
    global algo_results_dir, algo_cohesiveness_dir

    # Directory to access the algorithm results
    result_dir = algo_results_dir + algorithm + "_Results/"
    # Directory to store the psychology-informed cohesiveness results
    cohesiveness_dir = algo_cohesiveness_dir + algorithm + "_Results/"
    if not os.path.exists(cohesiveness_dir):
        os.makedirs(cohesiveness_dir)

    # Define the tasks to be executed in parallel
    tasks = []
    for dataset_name in dataset_list:
        for decay_method in ['exp', 'poly']:
            attribute_file = attribute_dir + dataset_name + "_attributed.txt"
            node_mapping_file = node_mapping_dir + dataset_name + "_node_mapping.txt"
            result_file = result_dir + algorithm + "_results_" + dataset_name + ".txt"

            if decay_method == 'exp':
                lambda_value_list = [0.0001, 0.0005, 0.001, 0.005, 0.01]
                for lambda_value in lambda_value_list:
                    output_file = cohesiveness_dir + algorithm + "_results_" + dataset_name + "_" + decay_method + "_" + str(lambda_value) + ".txt"
                    tasks.append((algorithm, dataset_name, decay_method, lambda_value, attribute_file, node_mapping_file, result_file, output_file, njobs))
            elif decay_method == 'poly':
                mu_value_list = [0.5, 1, 1.5, 2]
                for mu_value in mu_value_list:
                    output_file = cohesiveness_dir + algorithm + "_results_" + dataset_name + "_" + decay_method + "_" + str(mu_value) + ".txt"
                    tasks.append((algorithm, dataset_name, decay_method, mu_value, attribute_file, node_mapping_file, result_file, output_file, njobs))

    # Execute the tasks in parallel
    Parallel(n_jobs=njobs)(delayed(gf.cal_results)(*task) for task in tasks)

if __name__ == "__main__":
    attribute_dir = "D:/Cohesion_Evaluation/Original_Datasets/Preprocessed_Datasets/"
    node_mapping_dir = "D:/Cohesion_Evaluation/Original_Datasets/Node_Mapping/"
    algo_results_dir = "D:/Cohesion_Evaluation/Algorithm_Output/"
    algo_cohesiveness_dir = "D:/Cohesion_Evaluation/Cohesiveness_Output/"

    algo_list =["ALS", "WCF-CRC", "CSD", "ST-Exa", "Repeeling", "I2ACSM", "TransZero_LS", "TransZero_GS"]
    njobs = 1

    for algorithm in algo_list:
        if algorithm == "Repeeling":
            cohesiveness_calculation(algorithm, ["BTW17", "Chicago_COVID"], njobs)
        else:
            cohesiveness_calculation(algorithm, ["BTW17", "Chicago_COVID", "Crawled_Dataset26", "Crawled_Dataset144"], njobs)