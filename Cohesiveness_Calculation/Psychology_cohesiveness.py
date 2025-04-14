"""
Calculate the psychology-informed cohesiveness of CS results (using varying decay methods and decay rates).
"""

import os
from joblib import Parallel, delayed
import Utils.Process_algo as pa


def cohesiveness_calculation(algorithm, dataset_list, njobs):
    global algo_results_dir, algo_cohesiveness_dir

    result_dir = algo_results_dir + algorithm + "_Results/"
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
                    tasks.append((algorithm, decay_method, lambda_value, attribute_file, node_mapping_file, result_file, output_file, njobs))
            
            elif decay_method == 'poly':
                mu_value_list = [0.5, 1, 1.5, 2]
                for mu_value in mu_value_list:
                    output_file = cohesiveness_dir + algorithm + "_results_" + dataset_name + "_" + decay_method + "_" + str(mu_value) + ".txt"
                    tasks.append((algorithm, decay_method, mu_value, attribute_file, node_mapping_file, result_file, output_file, njobs))

    Parallel(n_jobs=njobs)(delayed(pa.cal_results)(*task) for task in tasks)

if __name__ == "__main__":
    attribute_dir = "D:/Cohesion_Evaluation/Original_Datasets/Preprocessed_Datasets/"
    node_mapping_dir = "D:/Cohesion_Evaluation/Original_Datasets/Node_Mapping/"
    algo_results_dir = "D:/Cohesion_Evaluation/Algorithm_Output/"
    algo_cohesiveness_dir = "D:/Cohesion_Evaluation/Cohesiveness_Output/"

    algo_list =["ALS", "WCF-CRC", "CSD", "ST-Exa", "Repeeling", "I2ACSM", "TransZero_LS", "TransZero_GS"]
    njobs = -1

    for algorithm in algo_list:
        if algorithm == "Repeeling":
            cohesiveness_calculation(algorithm, ["BTW17", "Chicago_COVID"], njobs)
        else:
            cohesiveness_calculation(algorithm, ["BTW17", "Chicago_COVID", "Crawled_Dataset144", "Crawled_Dataset26"], njobs)