"""
Using nodes in Chicago_COVID as the query node
1. Read results of all algorithms on this dataset
2. Find the query node with its resulted communities within size of [5, 30]
3. Draw the communities identified by all algorithms, and plot the cohesiveness score of each community.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.ticker import FuncFormatter, FixedLocator
import ast
import numpy as np

import sys
target_path = "./"
sys.path.append(target_path)
import Cohesiveness_Calculation.Utils.Graph_utils as gu
import Cohesiveness_Calculation.Utils.Process_algo as pa

def parse_cohesiveness_dim(cohesiveness_str):
    # Remove the brackets and split the string by commas
    cohesiveness_str = cohesiveness_str.strip('[]')
    parts = cohesiveness_str.split(',')
    
    # Convert each part to a float, handling 'nan' values
    cohesiveness_dim = []
    for part in parts:
        part = part.strip()
        if part == 'nan':
            cohesiveness_dim.append(0)
        else:
            cohesiveness_dim.append(float(part))
    
    return cohesiveness_dim

# Only save the results with valid community nodes
def process_algo_results(algo_result_dir, algorithm, file):
    results = {}
    with open(algo_result_dir + file, 'r') as f:
        lines = f.readlines()
        if algorithm in ["ALS", "WCF-CRC", "I2ACSM"]:
            for line in lines:
                parts = line.strip().split("\t")
                node = str(parts[0])
                value = float(parts[1])
                params = ast.literal_eval(parts[2])
                community_node_list = list(ast.literal_eval(parts[3]))
                community_node_list = [str(node) for node in community_node_list]
                if len(community_node_list) > 0:
                    cohesiveness_dim = parse_cohesiveness_dim(parts[4])
                    if node not in results:
                        results[node] = []
                    results[node].append([node, value, params, community_node_list, cohesiveness_dim])
            
        elif algorithm in ["ST-Exa", "CSD", "Repeeling"]:
            for line in lines:
                parts = line.strip().split("\t")
                node = str(parts[0])
                params = ast.literal_eval(parts[1])
                community_node_list = list(ast.literal_eval(parts[2]))
                community_node_list = [str(node) for node in community_node_list]
                
                if len(community_node_list) > 0:
                    cohesiveness_dim = ast.literal_eval(parts[3])
                    if node not in results:
                        results[node] = []
                    results[node].append([node, params, community_node_list, cohesiveness_dim])
                    
        elif algorithm == "TransZero_LS":
            for line in lines:
                parts = line.strip().split("\t")
                node = str(parts[0])
                community_node_list = ast.literal_eval(parts[1])
                if len(community_node_list) > 0:
                    cohesiveness_dim = ast.literal_eval(parts[2]) 
                    if node not in results:
                        results[node] = []
                    results[node].append([node, community_node_list, cohesiveness_dim])
                
    return results


def load_results(dataset, result_dir, algo_list):
    total_results = {}
    
    # In each algo result, extract the results with valid community nodes, and store the results as a dictionary
    for algo in algo_list:
        algo_dir = result_dir + algo + "_results/"
        algo_results = process_algo_results(algo_dir, algo, algo + "_results_" + dataset + "_exp_0.0001.txt")
        total_results[algo] = algo_results
    
    print("Finished processing results")
    
    return total_results


def find_community(results, query_nodes):
    node_community = {}
    for query_node in query_nodes:
        valid_community = True
        for algo, algo_results in results.items():
            if query_node not in algo_results.keys():
                valid_community = False
                break
        if valid_community:
            node_community[query_node] = {}
            for algo, algo_results in results.items():
                node_community[query_node][algo] = []
                for single_result in algo_results[query_node]:    
                    node_community[query_node][algo].append(single_result)
               
    return node_community


def print_query_community(node_community, algorithm_dict):
    # Store communities with valid size by query node
    valid_community = {}
    for query_node, algo_results in node_community.items():
        valid_community[query_node] = {}
        for algo, result in algo_results.items():
            valid_community[query_node][algo] = []
            for single_result in result:
                single_size = len(single_result[algorithm_dict[algo]])
                if 5 <= single_size <= 30:
                    valid_community[query_node][algo].append(single_result)
            print("\n")
    
    # Print the communities with valid size by query node
    for query_node, algo_results in valid_community.items():
        flag = [len(result) > 0 for result in algo_results.values()]
        valid_algo_num = [1 for f in flag if f]
        if np.sum(valid_algo_num) >= 3:
            print(f"Query node: {query_node}")
            for algo, result in algo_results.items():
                print(f"Algorithm: {algo}, Number of communities: {len(result)}")
                for single_result in result:
                    print(f"Community size: {len(single_result[algorithm_dict[algo]])}")
                    print(f"Community nodes: {single_result[algorithm_dict[algo]]}")
                    print(f"Cohesiveness score: {single_result[-1]}")
            print("=====================================")


def plot_network_with_background(combined_graph, graph, community, title, pos, node_color, edge_color, background_node_color, background_edge_color):
    plt.figure(figsize=(8, 6))
    
    # Draw the background
    nx.draw(combined_graph, pos=pos, node_color=background_node_color, node_size=30, edge_color=background_edge_color, with_labels=False, alpha=0.3)
    
     # Calculate edge widths based on the number of interactions and apply logarithmic scaling
    edge_widths = [graph.number_of_edges(u, v) for u, v in graph.edges()]
    log_edge_widths = [np.log1p(width) for width in edge_widths]  # Use log1p to handle zero widths
    max_log_width = max(log_edge_widths) if log_edge_widths else 1  # Avoid division by zero
    normalized_edge_widths = [width / max_log_width * 5 for width in log_edge_widths]  # Scale factor for visibility
    
    # Set node colors
    node_colors = ['red' if node == "162088750" else node_color for node in graph.nodes()]
    
    # Draw the community nodes and edges with normalized edge widths
    nx.draw(graph, pos=pos, node_color=node_colors, node_size=30, edge_color=edge_color, width=normalized_edge_widths, with_labels=False)
    
    # Draw the community node labels with mapped numbers
    labels = {node: node_mapping[int(node)] for node in community}
    label_pos = {node: (x, y - 0.05) for node, (x, y) in pos.items()}  # Adjust the y-coordinate to position the label below the node
    nx.draw_networkx_labels(graph, label_pos, labels=labels, font_color='black', font_size=14, verticalalignment='top')
    
    # plt.show()
    plt.tight_layout()
    plt.savefig("D:/Cohesion_Evaluation/Figures/Case_Study/" + title + ".png", dpi=600, bbox_inches='tight')


"""
    Calculate the number of triangles involving each edge in the given graph.
    Returns:  A dictionary where the keys are edges (as tuples) and values are the number of triangles.
"""
def count_triangles_per_edge(graph):    
    triangle_count = {}

    # Iterate over all edges in the graph
    for u, v in graph.edges():
        # Find common neighbors (nodes that form triangles with the edge u-v)
        common_neighbors = list(nx.common_neighbors(graph, u, v))
        
        # The number of triangles for this edge is the number of common neighbors
        triangle_count[(u, v)] = len(common_neighbors)
    
    return triangle_count


def get_core_truss_number(graph):
    undirected_graph = nx.Graph(graph)
    k_core_value = min(dict(nx.degree(undirected_graph)).values())
    k_truss_value = min(count_triangles_per_edge(undirected_graph).values())
    return k_core_value, k_truss_value + 2



def plot_broken_bar(combined_scores, measure_name, communities, low_ticks, high_ticks, height_ratios, bar_width):
    font_size = 17
    fig, (ax_high, ax_low) = plt.subplots(2, 1, sharex=True, figsize=(8, 4), gridspec_kw={'height_ratios': height_ratios})
    ax_low.set_ylim(0, 3)
    ax_high.set_ylim(15, np.max(combined_scores) * 1.05)
    
    color_sublist = [(53, 78, 151), (112, 163, 196), (199, 229, 236), (245, 180, 111), (251, 236, 171), (175, 175, 175), (219, 219, 219)] 
    color_sublist = [(r/255, g/255, b/255) for r, g, b in color_sublist]
    hatch_list = ['/', '\\', '|', '-', '+', 'x','.']

    indices = np.arange(len(combined_scores))  # Number of metrics per community

    for j in range(len(combined_scores[0])):  # 每个 measure
        vals = [scores[j] for scores in combined_scores]
        pos  = indices + j * bar_width
        ax_low .bar(pos, vals, bar_width, label=measure_name[j], color=color_sublist[j], hatch=hatch_list[j])
        ax_high.bar(pos, vals, bar_width, label=measure_name[j], color=color_sublist[j], hatch=hatch_list[j])

    ax_high.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax_low.spines['top'].set_visible(False)
    ax_high.spines['bottom'].set_visible(False)
    ax_low.tick_params(labeltop=False)
    ax_high.tick_params(labelbottom=False)

    d = .015
    ax_low.plot((-d, +d), (1 - d, 1 + d), transform=ax_low.transAxes, color='k', clip_on=False)
    ax_low.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_low.transAxes, color='k', clip_on=False)

    ax_high.plot((-d, +d), (-d, +d), transform=ax_high.transAxes, color='k', clip_on=False)
    ax_high.plot((1 - d, 1 + d), (-d, +d), transform=ax_high.transAxes, color='k', clip_on=False)

    ax_low.set_xticks(indices + bar_width * (len(combined_scores[0]) - 1)/2)
    ax_low.set_xticklabels(communities, fontsize=17)
    ax_low.set_yticks(low_ticks)
    ax_high.set_yticks(high_ticks)
    ax_low.set_yticklabels(low_ticks, fontsize=font_size)
    ax_high.set_yticklabels(high_ticks, fontsize=font_size)

    ax_high.legend(loc='upper center', bbox_to_anchor=(0.5, 1.62), fontsize=17, ncol=7, columnspacing=0.5, handlelength=0.9, handletextpad=0.6)
    fig.supylabel('Values', fontsize=font_size)

    # Show the plot
    plt.tight_layout()
    # plt.show()
    plt.savefig("D:/Cohesion_Evaluation/Figures/Case_Study/Case_study_measures.png")


if __name__ == "__main__":
    # Set the dataset and algo list
    dataset = "Chicago_COVID"
    algo_list = ["ALS", "WCF-CRC", "CSD", "ST-Exa", "Repeeling", "I2ACSM", "TransZero_LS"]

    # Set directory
    result_dir = "D:/Cohesion_Evaluation/Cohesiveness_Output/"
    query_dir = "D:/Cohesion_Evaluation/Original_Datasets/Query_Nodes/"
    

    # Set the loc of community nodes is each algo
    algorithm_dict = {"ALS": 3, "WCF-CRC": 3, "CSD": 2, "ST-Exa": 2, "Repeeling": 2, "I2ACSM": 3, "TransZero_LS": 1}

    # Read the node mapping file
    node_mapping_file = "D:/Cohesion_Evaluation/Original_Datasets/Node_Mapping/Chicago_COVID_node_mapping.txt"
    node_mapping = pa.read_node_mapping(node_mapping_file)
    # Reverse the mapping
    node_mapping = {v: k for k, v in node_mapping.items()}

    # Read attribute file
    attribute_file = "D:/Cohesion_Evaluation/Original_Datasets/Preprocessed_Datasets/Chicago_COVID_attributed.txt"
    reply_G = gu.graph_construction(attribute_file)

    # Read query node file
    query_nodes = gu.get_query_nodes(query_dir, dataset)

    # Read the results of all algorithms
    total_results = load_results(dataset, result_dir, algo_list)


    # Find communities for query node that have valid communities in all algorithms
    node_community = find_community(total_results, query_nodes)
    print(f"Number of query nodes with valid communities: {len(node_community)}")

    # For each query node, print: query node, and number of communities identified by each algorithm
    # For each community, print the size of the community
    print_query_community(node_community, algorithm_dict)


    # Record the selected communities to speed up the visualization
    # WCF-CRC Community: reliability score: 0.2874583795782464, parameter: [0.0, 2, 8, 5], for communities with same size, no matter how to change the parameter, the community and cohesiveness score is the same
    CRC_community = ['94512839', '5080801075847665442', '162088750', '741999510', '150694468', '7776953622', '7261737861155061262', '75977458', '40353110', '4137952675600432269', '837746582', '6394328176', '378050237', '3841782767']
    CRC_cohesiveness = [0.49813319197913625, 7.287152212926613e-32, 0.5137679826307116, 0.2432345774015647, 20.84065934065934]

    # STExa: size 30
    STExa_community = ['33925570', '109840068', '1316737784', '7222563205723261216', '378050237', '228128246', '5854801956', '304730466', '5080801075847665442', '564495923', '901095202', '633990130', '150694468', '6501281292778189260', '9990212403', '7943261589', '4137952675600432269', '76037566', '162088750', '895069896', '2529232071985340475', '468500969', '94512839', '7124806190109198830', '600551878', '7776953622', '821289741', '18736702', '84881588', '226307942']
    STExa_cohesiveness = [0.23215962555227074, -6.528389989147983e-06, 0.22761240045036332, 0.15215159152431626, 3.7471264367816093]

    # Repeeling: Among size 6, 9, 10, 15, 21 and 29, choose the largest one, [50000000, 250000, 0, 2], for communities with same size, no matter how to change the parameter, the community and cohesiveness score is the same
    Repeeling_community = ['2247039860705069469', '4137952675600432269', '7943261589', '564495923', '76037566', '18736702', '2458852536', '5854801956', '162088750', '468500969', '895069896', '7124806190109198830', '7222563205723261216', '821289741', '378050237', '4167528449671673144', '94512839', '228128246', '5080801075847665442', '150694468', '600551878', '1316737784', '901095202', '7089516562377264512', '109840068', '221698907989844221', '7776953622', '633990130', '7934378213']
    Repeeling_cohesiveness = [0.21291206102763036, -6.7535068853255e-06, 0.18852692197912613, 0.13970983342289092, 3.8423645320197046]


    # I2ACSM: score: 0.7287828, parameter: [1, 6, 0.2], for communities with same size, no matter how to change the parameter, the community and cohesiveness score is the same
    # Among size 5, 6, 7 and 12, choose the largest one
    I2ACSM_community = ['162088750', '313928883', '2443485611', '543585783', '3107304570', '449220885', '83819299', '50253428', '603404173', '63004935', '63348468', '3820169876654478576']
    I2ACSM_cohesiveness = [2.8936112510389412e-09, 0.0, 8.163235291235083e-06, 0.22448979591836735, 0.75]

    # TransZero_LS: 
    TransZero_LS_community = ['162088750', '564495923', '8384922591', '603404173', '2886016294']
    TransZero_LS_cohesiveness = [-3.6800212730827588e-22, 0.0, 1.9581453645148323e-05, 0.02826585179526356, 1.85]


    # Find the combination of the community nodes in all algorithms
    combined_community = list(set(CRC_community + STExa_community + Repeeling_community + I2ACSM_community + TransZero_LS_community))

    # Extract corresponding subgraph
    CRC_graph = reply_G.subgraph(CRC_community)
    STExa_graph = reply_G.subgraph(STExa_community)
    I2ACSM_graph = reply_G.subgraph(I2ACSM_community)
    Repeeling_graph = reply_G.subgraph(Repeeling_community)
    TransZero_LS_graph = reply_G.subgraph(TransZero_LS_community)
    combined_graph = reply_G.subgraph(combined_community)

    print(f"WCF-CRC graph info: nodes: {CRC_graph.number_of_nodes()}, edges: {CRC_graph.number_of_edges()}")
    print(f"ST-Exa graph info: nodes: {STExa_graph.number_of_nodes()}, edges: {STExa_graph.number_of_edges()}")
    print(f"I2ACSM graph info: nodes: {I2ACSM_graph.number_of_nodes()}, edges: {I2ACSM_graph.number_of_edges()}")
    print(f"Repeeling graph info: nodes: {Repeeling_graph.number_of_nodes()}, edges: {Repeeling_graph.number_of_edges()}")
    print(f"TransZero_LS graph info: nodes: {TransZero_LS_graph.number_of_nodes()}, edges: {TransZero_LS_graph.number_of_edges()}")
    print(f"Combined graph info: nodes: {combined_graph.number_of_nodes()}, edges: {combined_graph.number_of_edges()}")

    # For easy presentation, we exclude the self-loop edges
    # Create modifiable copies of the graphs
    CRC_graph = CRC_graph.copy()
    STExa_graph = STExa_graph.copy()
    I2ACSM_graph = I2ACSM_graph.copy()
    Repeeling_graph = Repeeling_graph.copy()
    TransZero_LS_graph = TransZero_LS_graph.copy()
    combined_graph = combined_graph.copy()

    # Remove self-loop edges from each graph
    CRC_graph.remove_edges_from(nx.selfloop_edges(CRC_graph))
    STExa_graph.remove_edges_from(nx.selfloop_edges(STExa_graph))
    I2ACSM_graph.remove_edges_from(nx.selfloop_edges(I2ACSM_graph))
    Repeeling_graph.remove_edges_from(nx.selfloop_edges(Repeeling_graph))
    TransZero_LS_graph.remove_edges_from(nx.selfloop_edges(TransZero_LS_graph))
    combined_graph.remove_edges_from(nx.selfloop_edges(combined_graph))

    """
    Draw all five communities in the same background graph
    For each figure, use combine graph as background, and draw the community nodes and edges on top of it.
    """
    # Set the font family to Arial
    plt.rcParams['font.family'] = 'arial'

    background_node_color = 'lightgrey'  
    background_edge_color = 'lightgrey'
    node_color = 'lightskyblue'
    edge_color = 'black'

    pos = nx.kamada_kawai_layout(combined_graph)

    # plot_network_with_background(combined_graph, CRC_graph, CRC_community, 'WCF-CRC Network', pos, node_color, edge_color, background_node_color, background_edge_color)
    # plot_network_with_background(combined_graph, STExa_graph, STExa_community, 'STExa Network',pos, node_color, edge_color, background_node_color, background_edge_color)
    # plot_network_with_background(combined_graph, I2ACSM_graph, I2ACSM_community, 'I2ACSM Network', pos, node_color, edge_color, background_node_color, background_edge_color)
    # plot_network_with_background(combined_graph, Repeeling_graph, Repeeling_community, 'Repeeling Network', pos, node_color, edge_color, background_node_color, background_edge_color)
    # plot_network_with_background(combined_graph, TransZero_LS_graph, TransZero_LS_community, 'TransZero_LS Network', pos, node_color, edge_color, background_node_color, background_edge_color)

    """
    Find the max core number and truss number for each community
    """
    CRC_k_core, CRC_truss = get_core_truss_number(CRC_graph)
    STExa_k_core, STExa_truss = get_core_truss_number(STExa_graph)
    Repeeling_k_core, Repeeling_truss = get_core_truss_number(Repeeling_graph)
    I2ACSM_k_core, I2ACSM_truss = get_core_truss_number(I2ACSM_graph)
    TransZero_LS_k_core, TransZero_LS_truss = get_core_truss_number(TransZero_LS_graph)

    print(f"WCF-CRC community: k-core: {CRC_k_core}, truss: {CRC_truss}")
    print(f"ST-Exa community: k-core: {STExa_k_core}, truss: {STExa_truss}")
    print(f"Repeeling community: k-core: {Repeeling_k_core}, truss: {Repeeling_truss}")
    print(f"I2ACSM community: k-core: {I2ACSM_k_core}, truss: {I2ACSM_truss}")
    print(f"TransZero_LS community: k-core: {TransZero_LS_k_core}, truss: {TransZero_LS_truss}")

    """
    Draw the cohesiveness score of each community
    """
    communities = ['WCF-CRC', 'ST-Exa', 'Repeeling+', 'I2ACSM', 'TransZero_LS']
    measure_name = ['EI', 'SIT', 'CED', 'GIP', 'GID', r'$k$-core', r'$k$-truss']
    cohesiveness_scores = [CRC_cohesiveness, STExa_cohesiveness, I2ACSM_cohesiveness, Repeeling_cohesiveness, TransZero_LS_cohesiveness]
    k_core_numbers = [CRC_k_core, STExa_k_core, Repeeling_k_core, I2ACSM_k_core, TransZero_LS_k_core]
    truss_numbers = [CRC_truss, STExa_truss, Repeeling_truss, I2ACSM_truss , TransZero_LS_truss]

    # Combine cohesiveness scores with k-core and truss numbers
    combined_scores = [scores + [k_core, truss] for scores, k_core, truss in zip(cohesiveness_scores, k_core_numbers, truss_numbers)]

    low_ticks = [1, 2, 3]
    high_ticks = [15, 20]
    plot_broken_bar(combined_scores, measure_name, communities, low_ticks, high_ticks, [1, 2], 0.1)