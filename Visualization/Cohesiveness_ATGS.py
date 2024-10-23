"""
This script is used to extract the coheisveness data in each dataset from each algo and plot the results
"""

import ast
import os
from matplotlib import hatch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


"""
Organize dataset results in a dictionary
1. Each dataset contains a dictionary
2. For each daatset dict, the key is (algo name, lambda/mu, and lambda/mu value), and the value being a list of two lists
"""


def load_lambda_results(dataset_list, condense_result_dir, lambda_list, algo_list):
    total_results = {}
    for dataset in dataset_list:
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
        
        total_results[dataset] = dataset_results
    
    print("Finished processing lambda results")
    
    return total_results


def load_mu_results(dataset_list, condense_result_dir, mu_list, algo_list):
    total_results = {}
    for dataset in dataset_list:
        dataset_dir = condense_result_dir + dataset + "/"

        # In each dataset, extract the condensed results for each algo + lambda, and store the results in a list
        dataset_results = {}
        for mu in mu_list:
            for algo in algo_list:
                algo_mu_file = dataset_dir + algo + "_results_" + dataset + "_poly_" + str(mu) + "_condensed.txt"
                with open(algo_mu_file, 'r') as f:
                    lines = f.readlines()

                algo_avg_results = []
                algo_std_results = []
                for line in lines:
                    parts = line.strip().split("\t")
                    algo_avg_results.append(ast.literal_eval(parts[1]))
                    algo_std_results.append(ast.literal_eval(parts[2]))
                
                print(f"Number of results: {len(algo_avg_results)}")
                dataset_results[(algo, mu)] = [np.mean(algo_avg_results, axis=0), np.mean(algo_std_results, axis=0)]
        
        total_results[dataset] = dataset_results
    
    print("Finished processing mu results")
    
    return total_results


def draw_graphs(total_results, dataset_list, params_list, algo_list, algo_label_list, color_list, hatch_list, font_size, x_label, threshold, CED_y_min, CED_y_max, CED_y_num, GID_y_min, GID_y_max, GID_y_num, save_label):
    for measure, measure_idx in measure_label.items():
        fig, ax = plt.subplots(figsize=(10, 5))

        for dataset in dataset_list:
            dataset_results = total_results[dataset]

            # Prepare data for each measure
            avg_data = []
            std_data = []
            for param in params_list:
                avg_data.append([dataset_results[(algo, param)][0][measure_idx] for algo in algo_list])
                std_data.append([dataset_results[(algo, param)][1][measure_idx] for algo in algo_list])
            
            avg_data = np.array(avg_data)
            std_data = np.array(std_data)
            
            # Plot data
            bar_width = 0.1
            x = np.arange(len(params_list))
            
            for i, algo in enumerate(algo_label_list):
                # Error bar plot
                # ax.bar(x + i * bar_width, avg_data[:, i], bar_width, yerr=std_data[:, i], label=algo, color=color_code[i])

                # No error bar plot
                ax.bar(x + i * bar_width, avg_data[:, i], bar_width, label=algo, color=color_list[i], hatch=hatch_list[i])
            
            ax.set_xticks(x + bar_width * (len(algo_list) - 1) / 2)
            ax.set_xticklabels(params_list, fontsize=font_size)
            # ax.set_xlabel(x_label, fontsize=font_size)
            
            # Find the maximum value of the y data and set the y limit
            max_y = np.max(avg_data + std_data)
            min_y = np.min(avg_data - std_data)
            
            if measure in ["EI", "SIT"]:
                # Set y scale to ensure negative values and positive values are shown and in log scale
                ax.set_ylim(min_y * 1.5, max_y * 1.5) # type: ignore
                
                ax.set_yscale('symlog', linthresh=threshold)
                y_ticks = np.concatenate([
                    -np.logspace(np.log10(-min_y * 1.5), np.log10(pow(10, -10)), num=3, endpoint=False),
                    [0],
                    np.logspace(np.log10(pow(10, -8)), np.log10(max_y * 1.5), num=3, endpoint=True)
                ])
                
                def format_func(value, tick_number):
                    if value == 0:
                        return "0"
                    exponent = int(np.log10(abs(value)))
                    return f"$10^{{{exponent}}}$"

                ax.yaxis.set_major_formatter(FuncFormatter(format_func))
                ax.set_yticks(y_ticks)
                ax.tick_params(axis='y', labelsize=font_size)

                                
            elif measure == "CED":
                y_ticks = np.linspace(CED_y_min, CED_y_max, num=CED_y_num)
                ax.set_yticklabels([f"{tick:.2f}" for tick in y_ticks], fontsize=font_size)

            ax.set_yticks(y_ticks)
            ax.set_ylabel(measure, fontsize=font_size)

            # Add legend on the upper center, two rows
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.23), ncol=4, fontsize=16, columnspacing=0.5)

            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 1]) # type: ignore

            # Save the figure
            save_path = "D:/Cohesion_Evaluation/Figures/Cohesiveness_ATGS/"
            plt.savefig(f"{save_path}{dataset}_{measure}_{save_label}.png")           

            # Show the figure
            plt.show()


if __name__ == "__main__":

    condense_result_dir = "D:/Cohesion_Evaluation/Cohesiveness_Output/"

    measure_label = {"EI": 0, "SIT": 1, "CED": 2}
    lambda_list = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    mu_list = [0.5, 1, 1.5, 2]
    dataset_label_list = ["BTW", "CC", "C144", "C26"]

    # For BTW17 and Chicago_COVID datasets
    algo_list = ["ALS", "WCF-CRC", "CSD", "ST-Exa", "Repeeling", "I2ACSM", "TransZero_LS"]
    algo_label_list = ["ALS", "WCF-CRC", "CSD", "ST-Exa", "Repeeling+", "I2ACSM", "TransZero_LS"]
    color_list = [(53, 78, 151), (112, 163, 196), (199, 229, 236), (245, 180, 111), (223, 91, 63), (251, 236, 171), (175, 175, 175)] 
    color_list = [(r/255, g/255, b/255) for r, g, b in color_list]
    hatch_list = ['/', '\\', '|', '-', '+', 'x','.']

    # For Crawled_Dataset144 and Crawled_Dataset26
    algo_sublist = ["ALS", "CRC", "CSD", "STExa", "IACSM", "TransZero_localsearch"]
    algo_label_sublist = ["ALS", "WCF-CRC", "CSD", "ST-Exa", "I2ACSM", "TransZero_LS"]
    color_sublist = [(53, 78, 151), (112, 163, 196), (199, 229, 236), (245, 180, 111), (251, 236, 171), (175, 175, 175)] 
    color_sublist = [(r/255, g/255, b/255) for r, g, b in color_sublist]
    hatch_sublist = ['/', '\\', '|', '-', 'x','.']

 
    dataset_list = ["BTW17", "Chicago_COVID", "Crawled_Dataset144", "Crawled_Dataset26"]

    total_lambda_results = load_lambda_results(dataset_list, condense_result_dir, lambda_list, algo_list)
    total_mu_results = load_mu_results(dataset_list, condense_result_dir, mu_list, algo_list)


    font_size = 19
    lambda_threshold = 1e-9
    mu_threshold = 1e-9
    plt.rcParams['font.family'] = 'arial'

    # Draw two sets of graphs for lambda and mu
    for dataset in dataset_list:
        if dataset == "BTW17":
            draw_graphs(total_lambda_results, dataset_list, lambda_list, algo_list, algo_label_list, color_list, hatch_list, font_size, r"$\lambda$", lambda_threshold, -1.2, 0.25, 7, 0, 0.4, 6, "lambda")
            draw_graphs(total_mu_results, dataset_list, mu_list, algo_list, algo_label_list, color_list, hatch_list, font_size, r"$\mu$", mu_threshold, -0.01, 1.49, 6, 0, 0.4, 6, "mu")
        elif dataset == "Chicago_COVID":
            draw_graphs(total_lambda_results, dataset_list, lambda_list, algo_list, algo_label_list, color_list, hatch_list, font_size, r"$\lambda$", lambda_threshold, -0.02, 0.16, 6, 0, 7, 6, "lambda")
            draw_graphs(total_mu_results, dataset_list, mu_list, algo_list, algo_label_list, color_list, hatch_list, font_size, r"$\mu$", mu_threshold, -1, 1.5, 7, 0, 7, 6, "mu")
        elif dataset == "Crawled_Dataset144":
            draw_graphs(total_lambda_results, dataset_list, lambda_list, algo_sublist, algo_label_sublist, color_sublist, hatch_sublist, font_size, r"$\lambda$", lambda_threshold, -2.7, 0.6, 6, 0, 4, 6, "lambda")
            draw_graphs(total_mu_results, dataset_list, mu_list, algo_sublist, algo_label_sublist, color_sublist, hatch_sublist, font_size, r"$\mu$", mu_threshold, -12, 12, 6, 0, 4, 6, "mu")
        
        elif dataset == "Crawled_Dataset26":
            draw_graphs(total_lambda_results, dataset_list, lambda_list, algo_sublist, algo_label_sublist, color_sublist, hatch_sublist, font_size, r"$\lambda$", lambda_threshold, -5, 165, 6, 0, 1.5, 6, "lambda")
            draw_graphs(total_mu_results, dataset_list, mu_list, algo_sublist, algo_label_sublist, color_sublist, hatch_sublist, font_size, r"$\mu$", mu_threshold, -1.5, 6, 5, 0, 1.5, 6, "mu")