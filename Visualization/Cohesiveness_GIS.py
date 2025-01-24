"""
This script is used to extract the cohesiveness data in each dataset from each algo and plot the results
"""

import ast
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


"""
Organize dataset results in a dictionary
1. Each dataset contains a dictionary
2. For each dataset dict, the key is (algo name, lambda/mu, and lambda/mu value), and the value being a list of two lists
"""


def load_lambda_results(dataset, condense_result_dir, lambda_list, algo_list):
    dataset_dir = condense_result_dir + dataset + "/"

    # In each dataset, extract the condensed results for each algo + lambda, and store the results in a list
    dataset_results = {}
    for lambda_value in lambda_list:
        for algo in algo_list:
            algo_lambda_file = dataset_dir + algo + "_results_" + dataset + "_exp_" + str(lambda_value) + "_condensed.txt"
            with open(algo_lambda_file, 'r') as f:
                lines = f.readlines()

            algo_avg_results = []
            algo_std_results = []
            for line in lines:
                parts = line.strip().split("\t")
                algo_avg_results.append(ast.literal_eval(parts[1]))
                algo_std_results.append(ast.literal_eval(parts[2]))
            
            print(f"Number of results: {len(algo_avg_results)}")
            dataset_results[(algo, lambda_value)] = [np.mean(algo_avg_results, axis=0), np.mean(algo_std_results, axis=0)]
    
    print("Finished processing lambda results")
    
    return dataset_results


if __name__ == "__main__":

    condense_result_dir = "D:/Cohesion_Evaluation/Cohesiveness_Output/"
    save_path = "D:/Cohesion_Evaluation/Figures/Cohesiveness_GIS/"
    
    measure_label = {"GIP": 3, "GID": 4}
    lambda_list = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    mu_list = [0.5, 1, 1.5, 2]
    dataset_label_list = ["BTW", "CC", "C144", "C26"]

    # For BTW17 and Chicago_COVID datasets
    algo_list = ["ALS", "WCF-CRC", "CSD", "ST-Exa", "Repeeling", "I2ACSM", "TransZero_LS"]
    algo_label_list = ["ALS", "WCF-CRC", "CSD", "ST-Exa", "Repeeling+", "I2ACSM", "TransZero_LS"]
    color_list = [(53, 78, 151), (112, 163, 196), (199, 229, 236), (245, 180, 111), (223, 91, 63), (251, 236, 171), (175, 175, 175)] 
    color_list = [(r/255, g/255, b/255) for r, g, b in color_list]

    # For Crawled_Dataset144 and Crawled_Dataset26
    algo_sublist = ["ALS", "WCF-CRC", "CSD", "ST-Exa", "I2ACSM", "TransZero_LS"]
    algo_label_sublist = ["ALS", "WCF-CRC", "CSD", "ST-Exa", "I2ACSM", "TransZero_LS"]
    color_sublist = [(53, 78, 151), (112, 163, 196), (199, 229, 236), (245, 180, 111), (251, 236, 171), (175, 175, 175)] 
    color_sublist = [(r/255, g/255, b/255) for r, g, b in color_sublist]

    hatch_list = ['/', '\\', '|', '-', '+', 'x','.']
    hatch_sublist = ['/', '\\', '|', '-', 'x','.']

    
    font_size = 25
    lambda_value = 0.0001
    # Set the font family to Times New Roman
    plt.rcParams['font.family'] = 'arial'

    dataset_list = ["BTW17", "Chicago_COVID", "Crawled_Dataset144", "Crawled_Dataset26"]
    dataset_label_list = ["BTW", "CC", "C144", "C26"]
    
    total_dataset_results = {}
    for dataset in dataset_list:
        if dataset in ["BTW17", "Chicago_COVID"]:
            results = load_lambda_results(dataset, condense_result_dir, lambda_list, algo_list)
        elif dataset in ["Crawled_Dataset144", "Crawled_Dataset26"]:
            results = load_lambda_results(dataset, condense_result_dir, lambda_list, algo_sublist)

        total_dataset_results[dataset] = results

    """
    Draw two graphs for two measures, each graph containing four groups of bars for four datasets, each group containing bars for all algos
    """
    for measure, measure_idx in measure_label.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        
        bar_width = 0.7
        x_offset = 0
        x_ticks_positions = []

        for dataset in dataset_list:
            dataset_results = total_dataset_results[dataset]
            if dataset in ["BTW17", "Chicago_COVID"]:
                current_algo_list = algo_list
                current_algo_label_list = algo_label_list
                current_color_list = color_list
                current_hatch_list = hatch_list
            elif dataset in ["Crawled_Dataset144", "Crawled_Dataset26"]:
                current_algo_list = algo_sublist
                current_algo_label_list = algo_label_sublist
                current_color_list = color_sublist
                current_hatch_list = hatch_sublist

            avg_data = []
            std_data = []
            for algo in current_algo_list:
                avg_data.append(dataset_results[(algo, lambda_value)][0][measure_idx])
                std_data.append(dataset_results[(algo, lambda_value)][1][measure_idx])
            
            avg_data = np.array(avg_data)
            std_data = np.array(std_data)
            
            x = np.arange(len(current_algo_list)) + x_offset
            x_ticks_positions.append(x.mean())  # Store the center position for the current group
            
            for i, algo in enumerate(current_algo_label_list):
                if dataset == "BTW17":
                    ax.bar(x[i], avg_data[i], bar_width, label=algo, color=current_color_list[i], hatch=current_hatch_list[i])
                else:
                    ax.bar(x[i], avg_data[i], bar_width, color=current_color_list[i], hatch=current_hatch_list[i])
            
            x_offset += len(current_algo_list) + 1  # Add space between groups

        ax.set_xticks(x_ticks_positions)
        ax.set_xticklabels(dataset_label_list, fontsize=font_size)
        
        
        if measure == "GIP":
            y_ticks = np.linspace(0, 1, num=6)
            ax.set_yticklabels([f"{tick:.1f}" for tick in y_ticks], fontsize=font_size)
    
        elif measure == "GID":
            y_ticks = np.linspace(0, 7, num=5)
            ax.set_yticklabels([f"{tick:.1f}" for tick in y_ticks], fontsize=font_size)

        ax.set_yticks(y_ticks)
        ax.set_ylabel(measure, fontsize=font_size)

        # Add legend on the upper center, two rows
        rcParams['legend.borderaxespad'] = 0.3
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.31), ncol=4, fontsize=19, handlelength=0.9, handletextpad=0.6, columnspacing=0.5)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 1]) # type: ignore
        
        # Save the figure
        plt.savefig(f"{save_path}All_datasets_{measure}_lambda.png", dpi=300, bbox_inches='tight')

        # Show the figure
        # plt.show()
