"""
This script is used to extract the coheisveness data in each dataset from each algo and plot the results(focus on sentiment analysis techniques)
"""

import ast
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


"""
Organize dataset results in a dictionary
1. Each dataset contains a dictionary
2. For each daatset dict, the key is (algo name, beta/alpha, and beta/alpha value), and the value being a list of two lists
"""


def load_llama_results(dataset_list, condense_result_dir, algo_list):
    total_results = {}
    for dataset in dataset_list:
        dataset_dir = condense_result_dir + dataset + "/"

        # In each dataset, extract the condensed results for each algo + beta, and store the results in a list
        dataset_results = {}
        for algo in algo_list:
            algo_beta_file = dataset_dir + algo + "_results_" + dataset + "_condensed.txt"
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
        
        total_results[dataset] = dataset_results
    
    print("Finished processing beta results")
    
    return total_results


def load_vader_results(dataset_list, condense_result_dir, algo_list):
    total_results = {}
    for dataset in dataset_list:
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
    
        total_results[dataset] = dataset_results
    
    print("Finished processing alpha results")
    
    return total_results

def draw_graphs(total_llama_results, total_vader_results, dataset_list, algo_list, algo_label_list, color_list, hatch_list, threshold, font_size, CED_y_min, CED_y_max, CED_y_num, GID_y_min, GID_y_max, GID_y_num):
    for measure, measure_idx in measure_label.items():
        for dataset in dataset_list:
            fig, ax = plt.subplots(figsize=(9, 5))

            # Prepare data for each measure
            avg_data = []
            std_data = []

            dataset_llama_results = total_llama_results[dataset]
            dataset_vader_results = total_vader_results[dataset]

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

            elif measure == "GIP":
                y_ticks = np.linspace(0, 1, num=6)
                ax.set_yticklabels([f"{tick:.1f}" for tick in y_ticks], fontsize=font_size)

            elif measure == "GID":
                y_ticks = np.linspace(GID_y_min, GID_y_max, num=GID_y_num)
                ax.set_yticklabels([f"{tick:.1f}" for tick in y_ticks], fontsize=font_size)

            ax.set_yticks(y_ticks)
            ax.set_ylabel(measure, fontsize=font_size)

            # Add legend on the upper center, two rows
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.23), ncol=4, fontsize=16, columnspacing=0.5)

            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 1]) # type: ignore

            # Save the figure
            save_path = "D:/Cohesion_Evaluation/Figures/Cohesiveness_Senti/"
            plt.savefig(f"{save_path}{dataset}_{measure}_senti.png")

            # Show the figure
            plt.show()

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

    total_llama_results = load_llama_results(dataset_list, condense_result_dir, algo_sublist)
    total_vader_results = load_vader_results(dataset_list, condense_result_dir, algo_sublist)

    font_size = 19
    llama_threshold = 1e-9
    # Set the font family to Times New Roman
    plt.rcParams['font.family'] = 'arial'


    # Draw two sets of graphs for llama and vader results
    for dataset in dataset_list:
        if dataset == "BTW17":
            draw_graphs(total_llama_results, total_vader_results, dataset_list, algo_list, algo_label_list, color_list, hatch_list, llama_threshold, font_size, -2.2, 0.3, 5, 0, 0.4, 6)
        elif dataset == "Chicago_COVID":
            draw_graphs(total_llama_results, total_vader_results, dataset_list, algo_list, algo_label_list, color_list, hatch_list, llama_threshold, font_size, 0, 0.15, 7, 0, 7, 6)
        elif dataset == "Crawled_Dataset144":
            draw_graphs(total_llama_results, total_vader_results, dataset_list, algo_sublist, algo_label_sublist, color_sublist, hatch_sublist, llama_threshold, font_size, -2.2, 0.7, 7, 0, 4, 6)
        elif dataset == "Crawled_Dataset26":
            draw_graphs(total_llama_results, total_vader_results, dataset_list, algo_sublist, algo_label_sublist, color_sublist, hatch_sublist, llama_threshold, font_size, -6.8, 2, 8, 0, 1.5, 6)