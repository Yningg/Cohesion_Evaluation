"""
Using nodes in Chicago_COVID as the query node
1. Read results of all algorithms on this dataset
2. Find the query node with its resulted communities within size of [5, 30]
3. Draw the communities identified by all algorithms, and plot the cohesiveness score of each community.
"""

import os
import networkx as nx
import matplotlib.pyplot as plt
import ast
import numpy as np

import sys
target_path = "./"
sys.path.append(target_path)
import Cohesiveness_Calculation.General_function as gf



# Only save the results with valid community nodes
def process_algo_results(algo_result_dir, algorithm, file):
    results = {}
    with open(algo_result_dir + file, 'r') as f:
        lines = f.readlines()
        if algorithm == ["ALS", "WCF-CRC", "I2ACSM"]:
            for line in lines:
                parts = line.strip().split("\t")
                node = str(parts[0])
                value = float(parts[1])
                params = ast.literal_eval(parts[2])
                community_node_list = list(ast.literal_eval(parts[3]))
                community_node_list = [str(node) for node in community_node_list]
                coheisveness_dim = ast.literal_eval(parts[4])
                if node not in results:
                    results[node] = []
                results[node].append([node, value, params, community_node_list, coheisveness_dim])
        
        elif algorithm in ["ST-Exa", "CSD", "Repeeling"]:
            for line in lines:
                parts = line.strip().split("\t")
                node = str(parts[0])
                params = ast.literal_eval(parts[1])
                community_node_list = list(ast.literal_eval(parts[2]))
                community_node_list = [str(node) for node in community_node_list]
                
                if len(community_node_list) > 0:
                    coheisveness_dim = ast.literal_eval(parts[3])
                    if node not in results:
                        results[node] = []
                    results[node].append([node, params, community_node_list, coheisveness_dim])
                    
        elif algorithm == "TransZero_LS":
            for line in lines:
                parts = line.strip().split("\t")
                node = str(parts[0])
                community_node_list = ast.literal_eval(parts[1])
                coheisveness_dim = ast.literal_eval(parts[2]) 
                if node not in results:
                    results[node] = []
                results[node].append([node, community_node_list, coheisveness_dim])
            
    return results


def load_results(dataset, result_dir, algo_list):
    total_results = {}
    
    # In each algo result, extract the results with valid community nodes, and store the results as a dictionary
    for algo in algo_list:
        algo_dir = result_dir + algo + "_results/"
        algo_results = process_algo_results(algo_dir, algo, algo + "_results_" + dataset + ".txt")
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
    for query_node, algo_results in node_community.items():
        print(f"Query node: {query_node}")
        for algo, result in algo_results.items():
            print(f"Algorithm: {algo}")
            for single_result in result:
                print(f"Community size: {len(single_result[algorithm_dict[algo]])}")
            print("\n")


def plot_network_with_background(combined_graph, graph, community, title, pos, node_color, edge_color, background_node_color, background_edge_color):
    plt.figure(figsize=(10, 10))
    
    # Draw the background
    nx.draw(combined_graph, pos=pos, node_color=background_node_color, edge_color=background_edge_color, with_labels=False, alpha=0.3)
    
     # Calculate edge widths based on the number of interactions and apply logarithmic scaling
    edge_widths = [graph.number_of_edges(u, v) for u, v in graph.edges()]
    log_edge_widths = [np.log1p(width) for width in edge_widths]  # Use log1p to handle zero widths
    max_log_width = max(log_edge_widths) if log_edge_widths else 1  # Avoid division by zero
    normalized_edge_widths = [width / max_log_width * 5 for width in log_edge_widths]  # Scale factor for visibility
    
    # Draw the community nodes and edges with normalized edge widths
    nx.draw(graph, pos=pos, node_color=node_color, edge_color=edge_color, width=normalized_edge_widths, with_labels=False)
    
    
    # Draw the community node labels with mapped numbers
    labels = {node: node_mapping[int(node)] for node in community}
    label_pos = {node: (x, y - 0.05) for node, (x, y) in pos.items()}  # Adjust the y-coordinate to position the label below the node
    nx.draw_networkx_labels(graph, label_pos, labels=labels, font_color='black', verticalalignment='top')
    
    # plt.show()
    plt.tight_layout()
    plt.savefig("D:/Cohesion_Evaluation/Figures/Case_Study/" + title + ".png")


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

if __name__ == "__main__":
    # Set the dataset and algo list
    dataset = "Chicago_COVID"
    algo_list = ["ALS", "WCF-CRC", "CSD", "ST-Exa", "Repeeling", "I2ACSM", "TransZero_LS"]

    # Set the result directory
    result_dir = "D:/Cohesion_Evaluation/Cohesiveness_Output/"


    # Set the loc of community nodes is each algo
    algorithm_dict = {"ALS": 3, "WCF-CRC": 3, "CSD": 2, "ST-Exa": 2, "Repeeling": 2, "I2ACSM": 3, "TransZero_LS": 1}


    # Read the node mapping file
    node_mapping_file = "D:/Cohesion_Evaluation/Original_Datasets/Node_Mapping/Chicago_COVID_node_mapping.txt"
    node_mapping = gf.read_node_mapping(node_mapping_file)

    # Read attribute file
    attribute_file = "D:/Cohesion_Evaluation/Original_Datasets/Preprocessed_Datasets/Chicago_COVID_attributed.txt"
    reply_G = gf.graph_construction(attribute_file)

    # Read query node file
    query_node_file = "D:/Cohesion_Evaluation/Original_Datasets/Query_Nodes/Chicago_COVID_query_node.txt"
    # query_nodes = gf.read_query_nodes(query_node_file)

    # Read the results of all algorithms
    # total_results = load_results(dataset, result_dir, algo_list)


    # Find communities for query node that have valid communities in all algorithms
    # node_community = find_community(total_results, query_nodes)
    # print(f"Number of query nodes with valid communities: {len(node_community)}")

    # For each query node, print: query node, and number of communities identified by each algorithm
    # For each community, print the size of the community
    # print_query_community(node_community, algorithm_dict)



    # WCF-CRC Community: reliability score: 0.3371472158657513 parameter: [0.0, 2, 4, 6], for communities with same size, no matter how to change the parameter, the community and cohesiveness score is the same
    CRC_community = ['1047977523320160384', '3403201023', '387676464', '1089495612', '2579503212', '195639161', '261514017', '826495641983528960', '1132020458868609024', '875179181331673089', '26174781', '101868197', '1140372799594389632', '26708318', '59253988', '1156006445114171392', '232785287', '1055847424747622400', '1043676244771053568', '1032433488560054400', '97613792', '105581789', '120677834', '1099134757', '114170205', '22578954']
    CRC_cohesiveness = [0.040675335086251395, 0.040675335086251395, 0.020210775983764067, 0.42924528301886794, 2.8]

    # STExa: size 20
    STExa_community = ['826495641983528960', '714669102', '37831300', '31049466', '1511414336', '1217938840834465798', '1050655603', '1156006445114171392', '1043676244771053568', '101598743', '1026320534647263232', '1092566360771739648', '2649430386', '1151565926350897152', '1099134757', '110801838', '1143262478027186176', '106010466', '1163317745628450818', '1165117868150796288']
    STExa_cohesiveness = [1.4995193999858507, 1.4995193999854792, 0.29777059622094726, 0.13584078875296696, 5.873684210526315]

    # I2ACSM: score: 0.9779547, parameter: [1, 4, 0.1], for communities with same size, no matter how to change the parameter, the community and cohesiveness score is the same
    # Among size 8, 9, and 14, choose the smallest one
    I2ACSM_community = ['1099134757', '1511414336', '1253831120', '826495641983528960', '1156006445114171392', '282949556', '570477589', '2767170695']
    I2ACSM_cohesiveness = [0.13219483903031704, 0.13219483903031704, 0.125, 0.10501355013550136, 2.767857142857143]

    # Repeeling: ['50000000', '5000000', '0', '2'], for communities with same size, no matter how to change the parameter, the community and cohesiveness score is the same
    Repeeling_community = ['101823277', '1026320534647263232', '1092566360771739648', '360188566', '101598743', '119191281', '21213366', '1076713315102089217', '31049466', '102807189', '208322515', '2649430386', '1020328250', '826495641983528960', '117879582', '1165117868150796288', '1142864340', '1050655603', '287286722', '37831300', '106010466', '110801838', '102128701', '1109445494390980608', '1217938840834465798', '116977325', '1099134757', '1304129868', '1156006445114171392']
    Repeeling_cohesiveness = [1.0469521609682646, 1.0578246129873792, 0.23910855272644574, 0.13970983342289092, 3.8423645320197046]


    # TransZero_localsearch: 
    Local_community = ['1099134757', '1143262478027186176', '1028356298503532555', '1151565926350897152', '1156006445114171392', '1163317745628450818', '1511414336']
    Local_cohesiveness = [0.1523896587364759, 0.1523896587354136, 0.42857142495909567, 0.011089681774349084, 0.5476190476190477]


    # Find the combination of the community nodes in all algorithms
    combined_community = list(set(CRC_community + STExa_community + Repeeling_community + I2ACSM_community + Local_community))

    # Extract corresponding subgraph
    CRC_graph = reply_G.subgraph(CRC_community)
    STExa_graph = reply_G.subgraph(STExa_community)
    I2ACSM_graph = reply_G.subgraph(I2ACSM_community)
    Repeeling_graph = reply_G.subgraph(Repeeling_community)
    Local_graph = reply_G.subgraph(Local_community)
    combined_graph = reply_G.subgraph(combined_community)

    print(f"WCF-CRC graph info: nodes: {CRC_graph.number_of_nodes()}, edges: {CRC_graph.number_of_edges()}")
    print(f"ST-Exa graph info: nodes: {STExa_graph.number_of_nodes()}, edges: {STExa_graph.number_of_edges()}")
    print(f"I2ACSM graph info: nodes: {I2ACSM_graph.number_of_nodes()}, edges: {I2ACSM_graph.number_of_edges()}")
    print(f"Repeeling graph info: nodes: {Repeeling_graph.number_of_nodes()}, edges: {Repeeling_graph.number_of_edges()}")
    print(f"LS graph info: nodes: {Local_graph.number_of_nodes()}, edges: {Local_graph.number_of_edges()}")
    print(f"Combined graph info: nodes: {combined_graph.number_of_nodes()}, edges: {combined_graph.number_of_edges()}")

    # For easy presentation, we exclude the self-loop edges
    # Create modifiable copies of the graphs
    CRC_graph = CRC_graph.copy()
    STExa_graph = STExa_graph.copy()
    I2ACSM_graph = I2ACSM_graph.copy()
    Repeeling_graph = Repeeling_graph.copy()
    Local_graph = Local_graph.copy()
    combined_graph = combined_graph.copy()

    # Remove self-loop edges from each graph
    CRC_graph.remove_edges_from(nx.selfloop_edges(CRC_graph))
    STExa_graph.remove_edges_from(nx.selfloop_edges(STExa_graph))
    I2ACSM_graph.remove_edges_from(nx.selfloop_edges(I2ACSM_graph))
    Repeeling_graph.remove_edges_from(nx.selfloop_edges(Repeeling_graph))
    Local_graph.remove_edges_from(nx.selfloop_edges(Local_graph))
    combined_graph.remove_edges_from(nx.selfloop_edges(combined_graph))

    """
    Draw all the communities in the same figure
    1. For each figure, use combine graph as background, and draw the community nodes and edges on top of it.
    2. Seven communities placed with two rows and four columns. 
    3. The last figure is the cohesiveness score of each community, draw as a line plot, each line represents a community.
    """
    # Set the font family to Times New Roman
    plt.rcParams['font.family'] = 'arial'

    background_node_color = 'lightgrey'  
    background_edge_color = 'lightgrey'
    node_color = 'lightskyblue'
    edge_color = 'black'

    pos = nx.kamada_kawai_layout(combined_graph)

    # plot_network_with_background(combined_graph, CRC_graph, CRC_community, 'CRC Network', pos, node_color, edge_color, background_node_color, background_edge_color)
    # plot_network_with_background(combined_graph, STExa_graph, STExa_community, 'STExa Network',pos, node_color, edge_color, background_node_color, background_edge_color)
    # plot_network_with_background(combined_graph, I2ACSM_graph, I2ACSM_community, 'I2ACSM Network', pos, node_color, edge_color, background_node_color, background_edge_color)
    # plot_network_with_background(combined_graph, Repeeling_graph, Repeeling_community, 'Repeeling Network', pos, node_color, edge_color, background_node_color, background_edge_color)
    # plot_network_with_background(combined_graph, Local_graph, Local_community, 'Local Network', pos, node_color, edge_color, background_node_color, background_edge_color)

    """
    Find the max core number and truss number for each community
    """
    CRC_k_core, CRC_truss = get_core_truss_number(CRC_graph)
    STExa_k_core, STExa_truss = get_core_truss_number(STExa_graph)
    Repeeling_k_core, Repeeling_truss = get_core_truss_number(Repeeling_graph)
    I2ACSM_k_core, I2ACSM_truss = get_core_truss_number(I2ACSM_graph)
    Local_k_core, Local_truss = get_core_truss_number(Local_graph)

    print(f"WCF-CRC community: k-core: {CRC_k_core}, truss: {CRC_truss}")
    print(f"ST-Exa community: k-core: {STExa_k_core}, truss: {STExa_truss}")
    print(f"Repeeling community: k-core: {Repeeling_k_core}, truss: {Repeeling_truss}")
    print(f"I2ACSM community: k-core: {I2ACSM_k_core}, truss: {I2ACSM_truss}")
    print(f"LS community: k-core: {Local_k_core}, truss: {Local_truss}")

    """
    Draw the cohesiveness score of each community
    1. The figure contains five groups of bars, each group represents a algo's community
    2. Each group contains seven bars
    --> For first five bars, each bar represents a cohesiveness score of a community
    --> For the last two bars, each bar represents the k-core number and truss number of the community
    """

    font_size = 17
    fig, ax = plt.subplots(figsize=(8, 4))
    communities = ['WCF-CRC', 'ST-Exa', 'Repeeling+', 'I2ACSM', 'TransZero_LS']
    measure_name = ['EL', 'SIT', 'CED', 'GIP', 'GID', r'$k$-core', r'$k$-truss']
    cohesiveness_scores = [
        CRC_cohesiveness,
        STExa_cohesiveness,
        I2ACSM_cohesiveness,
        Repeeling_cohesiveness,
        Local_cohesiveness
    ]
    k_core_numbers = [CRC_k_core, STExa_k_core, Repeeling_k_core, I2ACSM_k_core, Local_k_core]
    truss_numbers = [CRC_truss, STExa_truss, Repeeling_truss, I2ACSM_truss , Local_truss]

    # Combine cohesiveness scores with k-core and truss numbers
    combined_scores = [scores + [k_core, truss] for scores, k_core, truss in zip(cohesiveness_scores, k_core_numbers, truss_numbers)]

    color_sublist = [(53, 78, 151), (112, 163, 196), (199, 229, 236), (245, 180, 111), (251, 236, 171), (175, 175, 175), (219, 219, 219)] 
    color_sublist = [(r/255, g/255, b/255) for r, g, b in color_sublist]
    hatch_list = ['/', '\\', '|', '-', '+', 'x','.']

    bar_width = 0.1
    indices = np.arange(len(combined_scores))  # Number of metrics per community

    # Plot the bars for each measure
    for j in range(len(combined_scores[0])):  # Iterate over each measure
        measure_values = [scores[j] for scores in combined_scores]  # Values for the current measure
        bar_positions = indices + j * bar_width  # Shift each bar group
        ax.bar(bar_positions, measure_values, bar_width, 
            label=measure_name[j], color=color_sublist[j], hatch=hatch_list[j])

    # Set the x-axis labels
    ax.set_xticks(indices + bar_width * (len(combined_scores[0]) - 1) / 2)
    ax.set_xticklabels(communities, fontsize=font_size)

    # Set labels, title, and legend
    ax.set_ylabel('Values', fontsize=font_size)
    ax.set_ysticks = np.linspace(0, 1, num=6)
    ax.set_yticklabels(ax.get_yticks(), fontsize=font_size)
    ax.legend(loc='best', fontsize=15, ncol=3, columnspacing=0.5)

    # Show the plot
    plt.tight_layout()
    # plt.show()
    plt.savefig("D:/Cohesion_Evaluation/Figures/Case_Study/Case_study_measures.png")