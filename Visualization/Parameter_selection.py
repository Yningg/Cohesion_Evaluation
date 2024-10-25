"""
Using STExa results on COVID as an example, group the results by the parameters and calculate the average coheisveness scores for each measure
"""

import os
import ast
import joblib
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import tqdm


import sys
target_path = "./"
sys.path.append(target_path)
import Cohesiveness_Calculation.General_function as gf


# Group the results refer to parameters
def group_results(results, param_index, cohessiveness_index):
    grouped_results = {}
    for result in results:
        params = result[param_index][0] # Convert list to tuple
        if params in grouped_results:
            grouped_results[params].append([result[cohessiveness_index], result[cohessiveness_index+1]])
        else:
            grouped_results[params] = [[result[cohessiveness_index], result[cohessiveness_index+1]]]
    return grouped_results


def get_network_stats(G, community_node_list):
    subgraph = G.subgraph(community_node_list)
    diameter, size, min_degree = nx.diameter(subgraph), subgraph.number_of_nodes(), min(dict(subgraph.degree()).values())
    return [diameter, size, min_degree]


def process_ALS_line(Graph, line):
    parts = line.strip().split("\t")
    node = int(parts[0])
    beta_value = float(parts[1])
    params = ast.literal_eval(parts[2])
    community_node_list = list(ast.literal_eval(parts[3]))
    community_node_list = [str(node) for node in community_node_list]
    coheisveness_dim = ast.literal_eval(parts[4])

    structural_cohesiveness = get_network_stats(Graph, community_node_list)
    return [node, beta_value, params, community_node_list, coheisveness_dim, structural_cohesiveness]


def process_ALS_results(Graph, file):
    with open(file, 'r') as f:
        lines = f.readlines()

    results = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(process_ALS_line)(Graph, line) for line in tqdm.tqdm(lines)
    )
    return results



def process_STExa_results(Graph, file):
    results = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in tqdm.tqdm(lines):
            parts = line.strip().split("\t")
            node = int(parts[0])
            params = ast.literal_eval(parts[1])
            community_node_list = ast.literal_eval(parts[2])
            coheisveness_dim = ast.literal_eval(parts[3])
            structural_cohesiveness = get_network_stats(Graph, community_node_list)
            results.append([node, params, community_node_list, coheisveness_dim, structural_cohesiveness])
    return results


def get_mean_std(grouped_results):
    cohesiveness_mean = {}
    cohesiveness_std = {}
    structural_cohesiveness_mean = {}
    structural_cohesiveness_std = {}

    for params, results in tqdm.tqdm(grouped_results.items()):
        cohesiveness = [result[0] for result in results]
        structural_cohesiveness = [result[1] for result in results]

        cohesiveness_mean[params] = np.mean(cohesiveness, axis=0)
        cohesiveness_std[params] = np.std(cohesiveness, axis=0)

        structural_cohesiveness_mean[params] = np.mean(structural_cohesiveness, axis=0)
        structural_cohesiveness_std[params] = np.std(structural_cohesiveness, axis=0)

    return cohesiveness_mean, cohesiveness_std, structural_cohesiveness_mean, structural_cohesiveness_std


def read_results(results_dir, algorithm, dataset_name, param_index, cohessiveness_index):
    results = []
    
    attribute_file = attribute_dir + dataset_name + "_attributed.txt"

    # Build the graph with original nodes and edges attributes
    G = gf.graph_construction(attribute_file)
   
    file = results_dir + algorithm + "_results/" + algorithm + "_results_" + dataset_name + "_exp_0.0001.txt"
    
    if algorithm == "ST-Exa":
        results = process_STExa_results(G, file)
    elif algorithm == "ALS":
        results = process_ALS_results(G, file)
    
    print(f"Number of results: {len(results)}") # type: ignore

    # Group the results refer to the parameters
    grouped_results = group_results(results, param_index, cohessiveness_index)
    
    print(f"Number of grouped results: {len(grouped_results)}")

    cohesiveness_mean, cohesiveness_std, structural_cohesiveness_mean, structural_cohesiveness_std = get_mean_std(grouped_results)
    return list(cohesiveness_mean), list(cohesiveness_std), list(structural_cohesiveness_mean), list(structural_cohesiveness_std)

"""
Use params as x-axis, measures as y-axis
1. Draw a bar chart, with each parameter as a group, each measure as a bar
2. Draw a line chart, with each parameter as a line, each structure measure as a point
"""
def draw_graph(cohesiveness_mean, cohesiveness_std, structural_mean, structural_std, algorithm, dataset_name, y_lb, y_ub):
    # Draw the bar chart
    params = list(cohesiveness_mean.keys())
    cohesiveness = list(cohesiveness_mean.values())
    std_devs = list(cohesiveness_std.values())

    # sorted_params_cohesiveness = sorted(zip(params, cohesiveness, std_devs), key=lambda x: x[0][0])
    sorted_params_cohesiveness = sorted(zip(params, cohesiveness, std_devs), key=lambda x: x[0])
    sorted_params, sorted_cohesiveness, sorted_std = zip(*sorted_params_cohesiveness)

    bar_width = 0.14

    fig, ax = plt.subplots()
    fig.set_size_inches(7, 4)
    x = np.arange(len(sorted_params))
    for i, measure in enumerate(measures):
        ax.bar(x + i * bar_width, [cohesiveness[i] for cohesiveness in sorted_cohesiveness], bar_width, label=measure, color=color_list[i], hatch=hatch_list[i])

    ax.set_xlabel(r'$[l, h]$', fontsize=font_size)
    ax.set_ylabel('Cohesiveness', fontsize=font_size)
    ax.legend(fontsize=15, ncol=5, columnspacing=0.5, loc="upper center")
    ax.set_xticks(x + bar_width * (len(measures) - 1) / 2)
    ax.set_xticklabels([f"[{lb}-{ub}]" for (lb, ub) in sorted_params], fontsize=font_size)
    ax.set_ylim(0, 8)
    ax.set_yticklabels([f"{y:.1f}" for y in np.arange(0, 9, 1)], fontsize=font_size)
    plt.tight_layout()
    
    # save the figure
    save_path = "D:/Cohesion_Evaluation/Figures/Param_Selection/"
    plt.savefig(f"{save_path}{algorithm}_{dataset_name}_params_cohesiveness.png")
    plt.show()

    # Draw the line chart
    fig, ax = plt.subplots()
    fig.set_size_inches(7, 4)
    x = np.arange(len(sorted_params))
    for i, measure in enumerate(structural_measures):
        ax.plot(x, [structural_mean[param][i] for param in sorted_params], label=measure, color=color_list[i], marker=marker_list[i])
    
    ax.set_xlabel(r'$[l, h]$', fontsize=font_size)
    ax.set_ylabel('Structural Cohesiveness', fontsize=font_size)
    ax.legend(fontsize=15, ncol=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"[{lb}-{ub}]" for (lb, ub) in sorted_params], fontsize=font_size)
    ax.set_ylim(y_lb, y_ub)
    ax.set_yticklabels([f"{y}" for y in np.arange(y_lb, y_ub+1, 10)], fontsize=font_size)
    plt.tight_layout()
    plt.savefig(f"{save_path}{algorithm}_{dataset_name}_params_structural_cohesiveness.png")
    plt.show()



def draw_graph_ALS(cohesiveness_mean, cohesiveness_std, structural_mean, structural_std, algorithm, dataset_name, y_lb, y_ub):
    # Draw the bar chart
    params = list(cohesiveness_mean.keys())
    cohesiveness = list(cohesiveness_mean.values())
    std_devs = list(cohesiveness_std.values())

    # sorted_params_cohesiveness = sorted(zip(params, cohesiveness, std_devs), key=lambda x: x[0][0])
    sorted_params_cohesiveness = sorted(zip(params, cohesiveness, std_devs), key=lambda x: x[0])
    sorted_params, sorted_cohesiveness, sorted_std = zip(*sorted_params_cohesiveness)

    bar_width = 0.1

    fig, ax = plt.subplots()
    fig.set_size_inches(7, 4)
    x = np.arange(len(sorted_params))
    for i, measure in enumerate(measures):
        ax.bar(x + i * bar_width, [cohesiveness[i] for cohesiveness in sorted_cohesiveness], bar_width, label=measure, color=color_list[i], hatch=hatch_list[i])

    ax.set_xlabel(r'$\alpha$', fontsize=font_size)
    ax.set_ylabel('Cohesiveness', fontsize=font_size)
    ax.legend(fontsize=16, ncol=5, columnspacing=0.5, loc="upper center")
    ax.set_xticks(x + bar_width * (len(measures) - 1) / 2)
    ax.set_xticklabels([f"{param}" for param in sorted_params], fontsize=font_size)
    ax.set_ylim(0, 0.35)
    ax.set_yticklabels([f"{y:.2f}" for y in np.arange(0, 0.40, 0.05)], fontsize=font_size)
    plt.tight_layout()
    
    # save the figure
    save_path = "D:/Cohesion_Evaluation/Figures/Param_Selection/"
    plt.savefig(f"{save_path}{algorithm}_{dataset_name}_params_cohesiveness.png")
    plt.show()

    # Draw the line chart
    fig, ax = plt.subplots()
    fig.set_size_inches(7, 4)
    x = np.arange(len(sorted_params))
    # Transform the second parameter to from number 40000 to 4 for better visualization
    structural_mean = {param: [value[0], value[1]/1000, value[2]] for param, value in structural_mean.items()}

    for i, measure in enumerate(structural_measures):
        ax.plot(x, [structural_mean[param][i] for param in sorted_params], label=measure, color=color_list[i], marker=marker_list[i])
    
    ax.set_xlabel(r'$\alpha$', fontsize=font_size)
    ax.set_ylabel('Structural Cohesiveness', fontsize=font_size)
    ax.legend(fontsize=16, ncol=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{param}" for param in sorted_params], fontsize=font_size)
  
    # Set y-ticks and labels
    y_lb, y_ub = 0, 20 # Example bounds
    y_ticks = np.linspace(y_lb, y_ub, num=5)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y}" for y in y_ticks], fontsize=font_size)

    plt.tight_layout()
    plt.savefig(f"{save_path}{algorithm}_{dataset_name}_params_structural_cohesiveness.png")
    plt.show()


# Attribute directory
attribute_dir = "D:/Cohesion_Evaluation/Original_Datasets/Preprocessed_Datasets/"
results_dir = "D:/Cohesion_Evaluation/Cohesiveness_Output/"

# Parameters for the ST-Exa results
STExa_params_list = [[1, 10], [11, 20], [21, 30], [31, 40], [41, 50], [51, 60], [61, 70], [71, 80]]
ALS_params_list = [0.1, 0.15, 0.2, 0.25, 0.3]

algo = "ST-Exa"
measures = ['EL', 'SIT', 'CED', 'GIP', 'GID']

color_list = [(112, 163, 196), (245, 180, 111), (223, 91, 63), (251, 236, 171), (175, 175, 175), (219, 219, 219)]
color_list = [(r/255, g/255, b/255) for r, g, b in color_list]
hatch_list = ['/', '\\', '|', 'x','.']
marker_list = ['o', '^', 's']
font_size = 17
# Set the font family to Times New Roman
plt.rcParams['font.family'] = 'arial'


if algo == "ALS":
    structural_measures = [r'$d$', r'Size ($10^3$)', r'$Deg_{min}$']
    # cohesiveness_mean, cohesiveness_std, structural_mean, structural_std = read_results(results_dir, "ALS", "Chicago_COVID", 2, 4)
    cohesiveness_mean = {0.1: [0.0096272 , 0.00877609, 0.03635546, 0.2779106 , 0.01174341], 0.15: [0.01026318, 0.00938989, 0.03670313, 0.27530327, 0.01633922], 0.2: [0.01177796, 0.01083901, 0.04019605, 0.26604006, 0.04273258], 0.25: [0.0168045 , 0.01548076, 0.04287817, 0.25521285, 0.08060348], 0.3: [0.02562361, 0.02390711, 0.04659558, 0.25879281, 0.1410857 ]}
    cohesiveness_std = {0.1: [0.00096757, 0.00088203, 0.00365386, 0.02705455, 0.08835948], 0.15: [0.00540395, 0.00516395, 0.00707459, 0.03315818, 0.09410886], 0.2: [0.01035706, 0.00991178, 0.03133792, 0.05633179, 0.19533132], 0.25: [0.02456482, 0.02278977, 0.0351652 , 0.0809418 , 0.25411966], 0.3: [0.05067353, 0.0483413 , 0.05547948, 0.11084486, 0.31110428]}
    structural_mean = {0.1: [1.88400e+01, 4.86794e+03, 1.01000e+00], 0.15: [1.82200e+01, 4.66386e+03, 1.01000e+00], 0.2: [1.68100e+01, 4.22524e+03, 1.12000e+00], 0.25: [1.47800e+01, 3.52806e+03, 1.32000e+00], 0.3: [1.24000e+01, 2.70653e+03, 1.46000e+00]}
    structural_std = {0.1: [1.59197990e+00, 4.88140837e+02, 9.94987437e-02], 0.15: [3.11313347e+00, 1.02025099e+03, 9.94987437e-02], 0.2: [4.91466174e+00, 1.56960740e+03, 9.08625335e-01], 0.25: [6.26191664e+00, 2.06667222e+03, 1.40627167e+00], 0.3: [6.95125888e+00, 2.31354122e+03, 1.64572173e+00]}
    
    draw_graph_ALS(cohesiveness_mean, cohesiveness_std, structural_mean, structural_std, "ALS", "Chicago_COVID", 0, 400)

elif algo == "ST-Exa":
    structural_measures = [r'$d$', r'$Size$', r'$Deg_{min}$']
    # cohesiveness_mean, cohesiveness_std, structural_mean, structural_std = read_results(results_dir, "STExa", "Chicago_COVID", 1, 3)
    
    cohesiveness_mean = {('11', '20'): [0.78110709, 0.78262173, 0.14026461, 0.17582545, 4.16227299], ('1', '10'): [1.10609874, 1.11118601, 0.14514248, 0.12069323, 6.27286905], ('21', '30'): [0.56483016, 0.56253875, 0.14098072, 0.20071249, 2.74840105], ('31', '40'): [0.45709466, 0.44036697, 0.14082621, 0.22936697, 2.29770378], ('41', '50'): [0.38483921, 0.36921545, 0.13326832, 0.22618261, 1.77001443], ('51', '60'): [0.36875574, 0.35177879, 0.15101348, 0.22494084, 1.45429944], ('61', '70'): [0.34506246, 0.3298005 , 0.15997496, 0.22973856, 1.21157557], ('71', '80'): [0.31747808, 0.30377107, 0.16551793, 0.22964257, 1.03175337]}
    cohesiveness_std = {('11', '20'): [0.73639279, 0.7375801 , 0.11194181, 0.08069734, 2.46829573], ('1', '10'): [1.38755145, 1.39392211, 0.14744716, 0.11229284, 5.1989379 ], ('21', '30'): [0.50695043, 0.50449741, 0.1061029 , 0.07409881, 1.47520877], ('31', '40'): [0.3937373 , 0.37858033, 0.09412551, 0.0787711 , 1.28843665], ('41', '50'): [0.31969057, 0.30645713, 0.08632907, 0.06932565, 0.89992178], ('51', '60'): [0.28343315, 0.26922284, 0.10051729, 0.06626986, 0.68818671], ('61', '70'): [0.23874338, 0.22686533, 0.10039387, 0.05992445, 0.52520308], ('71', '80'): [0.20641864, 0.19622938, 0.09801992, 0.06042764, 0.4213347 ]}
    structural_mean = {('11', '20'): [ 4.96, 19.91, 15.4 ], ('1', '10'): [ 3.76,  9.11, 24.03], ('21', '30'): [ 5.34653465, 30.        , 14.43564356], ('31', '40'): [ 5.73, 39.99, 14.25], ('41', '50'): [ 5.8989899 , 50.        , 10.17171717], ('51', '60'): [ 6.19, 60.  , 10.56], ('61', '70'): [ 6.35, 70.  ,  9.8 ], ('71', '80'): [ 6.62, 79.99,  7.76]}
    structural_std = {('11', '20'): [ 1.56792857,  0.80118662, 14.15697708], ('1', '10'): [ 1.43610585,  2.23112079, 32.74582569], ('21', '30'): [ 1.81542153,  0.        , 13.03264435], ('31', '40'): [ 1.85932784,  0.09949874, 13.14410514], ('41', '50'): [1.99238423, 0.        , 7.65419328], ('51', '60'): [2.15728997, 0.        , 8.35741587], ('61', '70'): [2.29510348, 0.        , 8.40119039], ('71', '80'): [2.34      , 0.09949874, 5.13053603]}
    
    draw_graph(cohesiveness_mean, cohesiveness_std, structural_mean, structural_std, "ST-Exa", "Chicago_COVID", 0, 90)


