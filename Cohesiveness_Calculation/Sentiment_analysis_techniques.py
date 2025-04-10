"""
This script is used to calculate the psychology-informed cohesiveness for each community
1. Consider the varying time decay speed
2. Consider different time decay functions
"""

import os
from joblib import Parallel, delayed
import Cohesiveness_Calculation.Utils.Process_algo as pa


"""
Calculate the psychology-informed cohesiveness for each algorithm's results
"""
def cohesiveness_calculation(algo_list, njobs):
    global algo_results_dir, algo_cohesiveness_dir, decay_method, value
    tasks = []

    # Prepare tasks for all algorithms and datasets
    for algorithm in algo_list:

        result_dir = algo_results_dir + algorithm + "_Results/"  
        cohesiveness_dir = algo_cohesiveness_dir + algorithm + "_Results/"
        
        if not os.path.exists(cohesiveness_dir):
            os.makedirs(cohesiveness_dir)

        if algorithm == "Repeeling":
            dataset_list = ["BTW17", "Chicago_COVID"]
        else:
            dataset_list = ["BTW17", "Chicago_COVID", "Crawled_Dataset26", "Crawled_Dataset144"]

        for dataset_name in dataset_list:
            attribute_file = attribute_dir + dataset_name + "_vader_attributed.txt"  # Use the vader attributed file
            node_mapping_file = node_mapping_dir + dataset_name + "_node_mapping.txt"
            result_file = result_dir + algorithm + "_results_" + dataset_name + ".txt"
            output_file = cohesiveness_dir + algorithm + "_results_" + dataset_name + "_vader.txt"

            tasks.append((algorithm, decay_method, value, attribute_file, node_mapping_file, result_file, output_file, njobs))
            
    # Execute the tasks in parallel
    Parallel(n_jobs=njobs)(delayed(pa.cal_results)(*task) for task in tasks)


if __name__ == "__main__":
    attribute_dir = "D:/Cohesion_Evaluation/Original_Datasets/Preprocessed_Datasets/"
    node_mapping_dir = "D:/Cohesion_Evaluation/Original_Datasets/Node_Mapping/"
    algo_results_dir = "D:/Cohesion_Evaluation/Algorithm_Output/"
    algo_cohesiveness_dir = "D:/Cohesion_Evaluation/Cohesiveness_Output/"

    algo_list =["ALS", "WCF-CRC", "CSD", "ST-Exa", "Repeeling", "I2ACSM", "TransZero_LS"]

    # Parameters for the sensitivity analysis (sentiment analysis methods)
    decay_method = 'exp'
    value = 0.0001
    njobs = -1

    cohesiveness_calculation(algo_list, njobs)