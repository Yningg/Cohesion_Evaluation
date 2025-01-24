"""
This script is used to extract the cohesiveness data in each dataset from each algo and plot the results(focus on sentiment analysis techniques)
"""

import ast
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


"""
Organize dataset results in a dictionary
1. Each dataset contains a dictionary
2. For each dataset dict, the key is (algo name, beta/alpha, and beta/alpha value), and the value being a list of two lists
"""


def load_llama_results(dataset, condense_result_dir, algo_list):

    dataset_dir = condense_result_dir + dataset + "/"
    # In each dataset, extract the condensed results for each algo + beta, and store the results in a list
    dataset_results = {}

    for algo in algo_list:
        algo_beta_file = dataset_dir + algo + "_results_" + dataset + "_exp_0.0001_condensed.txt"
        with open(algo_beta_file, 'r') as f:
            lines = f.readlines()

        algo_avg_results = []
        algo_std_results = []
        for line in lines:
            parts = line.strip().split("\t")
            algo_avg_results.append(ast.literal_eval(parts[1]))
            algo_std_results.append(ast.literal_eval(parts[2]))
        
        print(f"Number of results: {len(algo_avg_results)}")
        dataset_results[algo] = [np.mean(algo_avg_results, axis=0), np.mean(algo_std_results, axis=0)]
    

    print("Finished processing beta results")
    
    return dataset_results


def load_vader_results(dataset, condense_result_dir, algo_list):

    dataset_dir = condense_result_dir + dataset + "/"

    # In each dataset, extract the condensed results for each algo + beta, and store the results in a list
    dataset_results = {}

    for algo in algo_list:
        algo_alpha_file = dataset_dir + algo + "_results_" + dataset + "_vader_condensed.txt"
        with open(algo_alpha_file, 'r') as f:
            lines = f.readlines()

        algo_avg_results = []
        algo_std_results = []
        for line in lines:
            parts = line.strip().split("\t")
            algo_avg_results.append(ast.literal_eval(parts[1]))
            algo_std_results.append(ast.literal_eval(parts[2]))
        
        print(f"Number of results: {len(algo_avg_results)}")
        dataset_results[algo] = [np.mean(algo_avg_results, axis=0), np.mean(algo_std_results, axis=0)]

    print("Finished processing alpha results")
    
    return dataset_results


def draw_graphs(dataset_llama_results, dataset_vader_results, dataset_label, algo_list, algo_label_list, color_list, hatch_list, threshold, font_size, CED_y_min, CED_y_max, CED_y_num):
    for measure, measure_idx in measure_label.items():

        fig, ax = plt.subplots(figsize=(10, 6))

        # Prepare data for each measure
        avg_data = []
        std_data = []

        avg_data.append([dataset_llama_results[algo][0][measure_idx] for algo in algo_list])
        avg_data.append([dataset_vader_results[algo][0][measure_idx] for algo in algo_list])

        std_data.append([dataset_llama_results[algo][1][measure_idx] for algo in algo_list])
        std_data.append([dataset_vader_results[algo][1][measure_idx] for algo in algo_list])
        
        # Plot data
        avg_data = np.array(avg_data)
        std_data = np.array(std_data)

        bar_width = 0.1
        params_list = ["Llama", "VADER"]
        x = np.arange(len(params_list))

        for i, algo in enumerate(algo_label_list):
            ax.bar(x + i * bar_width, avg_data[:, i], bar_width, label=algo, color=color_list[i], hatch=hatch_list[i])

        ax.set_xticks(x + bar_width * (len(algo_list) - 1) / 2)
        ax.set_xticklabels(params_list, fontsize=font_size)
        # ax.set_xlabel("Sentiment Analysis Techniques", fontsize=font_size)

        # Find the maximum value of the y data and set the y limit
        max_y = np.max(avg_data + std_data)
        min_y = np.min(avg_data - std_data)

        if measure in ["EI", "SIT"]:
            # Set y scale to ensure negative values and positive values are shown and in log scale
            ax.set_ylim(10**-1, 10**0)
            
            ax.set_yscale('symlog', linthresh=threshold)
            y_ticks = [-1, -pow(10, -3), -pow(10, -6), 0, pow(10, -8), pow(10, -4), 1]
            
            def format_func(value, tick_number):
                if value in [-1, 0, 1]:
                    return str(int(value))
                exponent = int(np.log10(abs(value)))
                sign = "-" if value < 0 else ""
                return f"${sign}10^{{{exponent}}}$"

            ax.yaxis.set_major_formatter(FuncFormatter(format_func))
            ax.set_yticks(y_ticks)
            ax.tick_params(axis='y', labelsize=font_size)

        elif measure == "CED":
            y_ticks = np.linspace(CED_y_min, CED_y_max, num=CED_y_num)
            ax.set_yticklabels([f"{tick:.2f}" for tick in y_ticks], fontsize=font_size)


        ax.set_xlabel(f"{dataset_label}", fontsize=font_size)
        ax.set_yticks(y_ticks)
        ax.set_ylabel(measure, fontsize=font_size)

        # Add legend on the upper center, two rows
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.32), ncol=4, fontsize=21, handlelength=0.9, handletextpad=0.6, columnspacing=0.5)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 1]) # type: ignore

        # Save the figure
        save_path = "D:/Cohesion_Evaluation/Figures/Cohesiveness_Senti/"
        plt.savefig(f"{save_path}{dataset}_{measure}_senti.png", dpi=300, bbox_inches='tight')

        # Show the figure
        # plt.show()

if __name__ == "__main__":
    condense_result_dir = "D:/Cohesion_Evaluation/Cohesiveness_Output/"

    measure_label = {"EI": 0, "SIT": 1, "CED": 2}
    dataset_label_list = ["BTW", "CC", "C144", "C26"]

    # For BTW17 and Chicago_COVID datasets
    algo_list = ["ALS", "WCF-CRC", "CSD", "ST-Exa", "Repeeling", "I2ACSM", "TransZero_LS"]
    algo_label_list = ["ALS", "WCF-CRC", "CSD", "ST-Exa", "Repeeling+", "I2ACSM", "TransZero_LS"]
    color_list = [(53, 78, 151), (112, 163, 196), (199, 229, 236), (245, 180, 111), (223, 91, 63), (251, 236, 171), (175, 175, 175)]
    color_list = [(r/255, g/255, b/255) for r, g, b in color_list]
    hatch_list = ['/', '\\', '|', '-', '+', 'x','.']

    # For Crawled_Dataset144 and Crawled_Dataset26
    algo_sublist = ["ALS", "WCF-CRC", "CSD", "ST-Exa", "I2ACSM", "TransZero_LS"]
    algo_label_sublist = ["ALS", "WCF-CRC", "CSD", "ST-Exa", "I2ACSM", "TransZero_LS"]
    color_sublist = [(53, 78, 151), (112, 163, 196), (199, 229, 236), (245, 180, 111), (251, 236, 171), (175, 175, 175)]
    color_sublist = [(r/255, g/255, b/255) for r, g, b in color_sublist]
    hatch_sublist = ['/', '\\', '|', '-', 'x','.']

    dataset_list = ["BTW17", "Chicago_COVID", "Crawled_Dataset144", "Crawled_Dataset26"]

    font_size = 25
    llama_threshold = 1e-9
    # Set the font family to Times New Roman
    plt.rcParams['font.family'] = 'arial'


    # Draw two sets of graphs for llama and vader results
    for dataset, dataset_label in zip(dataset_list, dataset_label_list):
        if dataset == "BTW17":
            llama_results = load_llama_results(dataset, condense_result_dir, algo_list)
            vader_results = load_vader_results(dataset, condense_result_dir, algo_list)
            draw_graphs(llama_results, vader_results, dataset_label, algo_list, algo_label_list, color_list, hatch_list, llama_threshold, font_size, -0.6, 0.6, 5)
        elif dataset == "Chicago_COVID":
            llama_results = load_llama_results(dataset, condense_result_dir, algo_list)
            vader_results = load_vader_results(dataset, condense_result_dir, algo_list)
            draw_graphs(llama_results, vader_results, dataset_label, algo_list, algo_label_list, color_list, hatch_list, llama_threshold, font_size, -0.8, 0.8, 6)
        elif dataset == "Crawled_Dataset144":
            llama_results = load_llama_results(dataset, condense_result_dir, algo_sublist)
            vader_results = load_vader_results(dataset, condense_result_dir, algo_sublist)
            draw_graphs(llama_results, vader_results, dataset_label, algo_sublist, algo_label_sublist, color_sublist, hatch_sublist, llama_threshold, font_size, -1.1, 1.1, 6)
        elif dataset == "Crawled_Dataset26":
            llama_results = load_llama_results(dataset, condense_result_dir, algo_sublist)
            vader_results = load_vader_results(dataset, condense_result_dir, algo_sublist)
            draw_graphs(llama_results, vader_results, dataset_label, algo_sublist, algo_label_sublist, color_sublist, hatch_sublist, llama_threshold, font_size, -120, 120, 7)