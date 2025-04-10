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


# Group the results refer to parameters
def group_results(algorithm, results, param_index, cohesiveness_index):
    grouped_results = {}
    if algorithm == "ST-Exa":
        for result in results:
            params = result[param_index] # Convert list to tuple
            if params in grouped_results:
                grouped_results[params].append([result[cohesiveness_index], result[cohesiveness_index+1]])
            else:
                grouped_results[params] = [[result[cohesiveness_index], result[cohesiveness_index+1]]]
    else:
        for result in results:
            params = result[param_index][0] # Convert list to tuple
            if params in grouped_results:
                grouped_results[params].append([result[cohesiveness_index], result[cohesiveness_index+1]])
            else:
                grouped_results[params] = [[result[cohesiveness_index], result[cohesiveness_index+1]]]
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
    cohesiveness_dim = ast.literal_eval(parts[4])

    structural_cohesiveness = get_network_stats(Graph, community_node_list)
    return [node, beta_value, params, community_node_list, cohesiveness_dim, structural_cohesiveness]


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
            params = tuple([str(para) for para in params])
            community_node_list = ast.literal_eval(parts[2])
            cohesiveness_dim = ast.literal_eval(parts[3])
            structural_cohesiveness = get_network_stats(Graph, community_node_list)
            results.append([node, params, community_node_list, cohesiveness_dim, structural_cohesiveness])
    return results


def get_mean_std(grouped_results):
    cohesiveness_mean = {}
    cohesiveness_std = {}
    structural_cohesiveness_mean = {}
    structural_cohesiveness_std = {}

    for params, results in tqdm.tqdm(grouped_results.items()):
        cohesiveness = [result[0] for result in results]
        structural_cohesiveness = [result[1] for result in results]

        cohesiveness_mean[params] = list(np.mean(cohesiveness, axis=0))
        cohesiveness_std[params] = list(np.std(cohesiveness, axis=0))

        structural_cohesiveness_mean[params] = list(np.mean(structural_cohesiveness, axis=0))
        structural_cohesiveness_std[params] = list(np.std(structural_cohesiveness, axis=0))

    return cohesiveness_mean, cohesiveness_std, structural_cohesiveness_mean, structural_cohesiveness_std


def read_results(results_dir, algorithm, dataset_name, param_index, cohesiveness_index):
    results = []
    
    attribute_file = attribute_dir + dataset_name + "_attributed.txt"

    # Build the graph with original nodes and edges attributes
    G = nx.read_edgelist(attribute_file, nodetype=str, data=(('timestamp', str), ('sentiment', str)), create_using=nx.MultiGraph()) # type: ignore
   
    file = results_dir + algorithm + "_Results/" + algorithm + "_results_" + dataset_name + "_exp_0.0001.txt"
    
    if algorithm == "ST-Exa":
        results = process_STExa_results(G, file)
    elif algorithm == "ALS":
        results = process_ALS_results(G, file)
    
    print(f"Number of results: {len(results)}") # type: ignore

    # Group the results refer to the parameters
    grouped_results = group_results(algorithm, results, param_index, cohesiveness_index)
    
    print(f"Number of grouped results: {len(grouped_results)}")

    cohesiveness_mean, cohesiveness_std, structural_cohesiveness_mean, structural_cohesiveness_std = get_mean_std(grouped_results)
    return cohesiveness_mean, cohesiveness_std, structural_cohesiveness_mean, structural_cohesiveness_std

"""
Use params as x-axis, measures as y-axis
1. Draw a bar chart, with each parameter as a group, each measure as a bar
2. Draw a line chart, with each parameter as a line, each structure measure as a point
"""
def draw_graph_STExa(cohesiveness_mean, cohesiveness_std, structural_mean, structural_std, algorithm, dataset_name, y_lb, y_ub):
    global save_path
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
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=5, fontsize=15, handlelength=0.9, handletextpad=0.6, columnspacing=0.5)
    ax.set_xticks(x + bar_width * (len(measures) - 1) / 2)
    ax.set_xticklabels([f"[{lb}-{ub}]" for (lb, ub) in sorted_params], fontsize=font_size)
    ax.set_ylim(0, 7)
    ax.set_yticklabels([f"{y:.1f}" for y in np.arange(0, 8, 1)], fontsize=font_size)
    plt.tight_layout()
    
    # save the figure
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
    global save_path
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
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=5, fontsize=16, handlelength=0.9, handletextpad=0.6, columnspacing=0.5)
    ax.set_xticks(x + bar_width * (len(measures) - 1) / 2)
    ax.set_xticklabels([f"{param}" for param in sorted_params], fontsize=font_size)
    ax.set_ylim(0, 0.30)
    ax.set_yticklabels([f"{y:.2f}" for y in np.arange(0, 0.35, 0.05)], fontsize=font_size)
    plt.tight_layout()
    
    # save the figure
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
    ax.set_yticklabels([f"{int(y)}" for y in y_ticks], fontsize=font_size)

    plt.tight_layout()
    plt.savefig(f"{save_path}{algorithm}_{dataset_name}_params_structural_cohesiveness.png")
    plt.show()


# Attribute directory
attribute_dir = "D:/Cohesion_Evaluation/Original_Datasets/Preprocessed_Datasets/"
results_dir = "D:/Cohesion_Evaluation/Cohesiveness_Output/"
save_path = "D:/Cohesion_Evaluation/Figures/Param_Selection/"
    
# Parameters for the ST-Exa results
STExa_params_list = [[1, 10], [11, 20], [21, 30], [31, 40], [41, 50], [51, 60], [61, 70], [71, 80]]
ALS_params_list = [0.1, 0.15, 0.2, 0.25, 0.3]

algo = "ALS"
measures = ['EI', 'SIT', 'CED', 'GIP', 'GID']

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
    cohesiveness_mean = {0.1: [0.0028348293805377928, -3.943318856516453e-08, 0.0028348293805377928, 0.2779106012785452, 0.01174340537875847], 0.15: [0.002981548952111736, -4.296945797411619e-08, 0.002938534263530323, 0.27530327435241964, 0.0163392223194381], 0.2: [0.0033255095639530712, -4.734898268193645e-08, 0.00327119141713949, 0.2660400585669525, 0.042732578063490054], 0.25: [0.004629854985771444, -6.96966132529493e-08, 0.00455942715338269, 0.2552128520431046, 0.08060348112767819], 0.3: [0.006493411959997443, -5.856853839345115e-08, 0.0062662391938188575, 0.2587928092148737, 0.14108569681556593]}
    cohesiveness_std = {0.1: [0.00028491107272483375, 3.963184568423761e-09, 0.00028491107272483375, 0.02705455388782668, 0.0883594759836271], 0.15: [0.0013377613766788529, 2.8628117729097236e-08, 0.001507473350835971, 0.03315817661943525, 0.09410886033111145], 0.2: [0.002466283033673564, 4.5884646456404205e-08, 0.0023310920010255923, 0.05633178775762061, 0.19533132390479915], 0.25: [0.006193938557990232, 1.22050204353031e-07, 0.006246211781278841, 0.08094179635197817, 0.2541196617026462], 0.3: [0.01111683091606338, 1.6444947056850663e-07, 0.010720634952969007, 0.11084485955259595, 0.31110427795405865]}
    structural_mean = {0.1: [18.84, 4867.94, 1.01], 0.15: [18.22, 4663.86, 1.01], 0.2: [16.81, 4225.24, 1.12], 0.25: [14.78, 3528.06, 1.32], 0.3: [12.4, 2706.53, 1.46]}
    structural_std = {0.1: [1.5919798993705914, 488.1408366445079, 0.09949874371066196], 0.15: [3.113133469673277, 1020.2509889238053, 0.09949874371066196], 0.2: [4.914661738105687, 1569.6073975360835, 0.9086253353280426], 0.25: [6.2619166394962456, 2066.67221793878, 1.4062716664997548], 0.3: [6.951258878793109, 2313.5412183706594, 1.6457217261736576]}

    draw_graph_ALS(cohesiveness_mean, cohesiveness_std, structural_mean, structural_std, "ALS", "Chicago_COVID", 0, 400)

elif algo == "ST-Exa":
    structural_measures = [r'$d$', r'$Size$', r'$Deg_{min}$']
    # cohesiveness_mean, cohesiveness_std, structural_mean, structural_std = read_results(results_dir, "ST-Exa", "Chicago_COVID", 1, 3)
    
    cohesiveness_mean = {("11", "20"): [0.1615997222393276, -5.875550990233185e-07, 0.14976179602684672, 0.17582545393569604, 4.162272993088784], ("1", "10"): [0.2027202162803341, -1.2403940979381168e-06, 0.19357930074615426, 0.1236931326522336, 6.090646825396826], ("21", "30"): [0.13135091005807917, -4.569872992403587e-07, 0.12750011914888024, 0.20210303646811956, 2.770965517241379], ("31", "40"): [0.11220962519487182, -6.365180239419283e-07, 0.10955471385763618, 0.22957158275599776, 2.3143832658569496], ("41", "50"): [0.09436393219727687, -8.617474785675342e-07, 0.09146696120996488, 0.2265146363352783, 1.7799224489795922], ("51", "60"): [0.0908633849410931, -1.5668135974757747e-06, 0.08835758948978689, 0.2249408422419007, 1.4542994350282488], ("61", "70"): [0.08550440912207939, -1.622771226011513e-06, 0.08311931393031291, 0.2304207471120586, 1.217372670807453], ("71", "80"): [0.07946012299054918, -1.4447161771644813e-06, 0.07833122050671937, 0.2296425709335233, 1.0317533674131776]}
    cohesiveness_std = {("11", "20"): [0.15330733072461933, 2.325610080262108e-06, 0.14621209054570947, 0.0806973397970272, 2.468295734619508], ("1", "10"): [0.26257047716245574, 7.25180424527702e-06, 0.26245255619189367, 0.11386962066692155, 5.065839206415121], ("21", "30"): [0.1161840028060218, 1.6656992279104705e-06, 0.11692806480626608, 0.07368003403744672, 1.4749013292537516], ("31", "40"): [0.09109378016539477, 1.646639980918146e-06, 0.09174912912522293, 0.0786216205292985, 1.2799394773520727], ("41", "50"): [0.07405156642563536, 1.622616852659548e-06, 0.07421930925959817, 0.06905721700771027, 0.9008214728406855], ("51", "60"): [0.06796897687490527, 1.6307912966832677e-06, 0.06837294739578068, 0.06626985744796668, 0.6881867142839967], ("61", "70"): [0.05829365347341467, 1.380918178946333e-06, 0.0585449166871577, 0.05976675244185963, 0.5279725778562404], ("71", "80"): [0.05129305319550141, 1.2043421097836228e-06, 0.05271151234951, 0.06042764240581968, 0.4213346971750918]}
    structural_mean = {("11", "20"): [4.96, 19.91, 15.4], ("1", "10"): [3.76, 9.11, 22.26], ("21", "30"): [5.34, 30.0, 14.19], ("31", "40"): [5.7, 39.99, 14.33], ("41", "50"): [5.89, 50.0, 10.24], ("51", "60"): [6.19, 60.0, 10.56], ("61", "70"): [6.35, 70.0, 9.97], ("71", "80"): [6.62, 79.99, 7.76]}
    structural_std = {("11", "20"): [1.5679285698015715, 0.8011866199581715, 14.15697707845852], ("1", "10"): [1.4361058456812994, 2.2311207945783664, 30.85891119271709], ("21", "30"): [1.7900837969212497, 0.0, 12.887742238266561], ("31", "40"): [1.8466185312619385, 0.09949874371066207, 12.974632942784929], ("41", "50"): [1.984414271264949, 0.0, 7.646070886409567], ("51", "60"): [2.1572899666016156, 0.0, 8.357415868556501], ("61", "70"): [2.2951034835057, 0.0, 8.45393990988817], ("71", "80"): [2.34, 0.09949874371066199, 5.1305360343730175]}
    
    draw_graph_STExa(cohesiveness_mean, cohesiveness_std, structural_mean, structural_std, "ST-Exa", "Chicago_COVID", 0, 90)


