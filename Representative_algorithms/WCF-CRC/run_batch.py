import CRC
from time import time
import itertools
from joblib import Parallel, delayed
import tqdm


def query(graph_path, theta, k, query_node, method, alpha, start_time, end_time):
    global n_jobs
    theta = float(theta)  # parameter of the edge weight threshold, float number in [0,1]
    k = int(k)  # parameter of the k-core constraint, integer
    ts = int(start_time)  # starting timestamp of the query interval (included), integer
    te = int(end_time)  # ending timestamp of the query interval (excluded), integer
    alpha = float(alpha)  # parameter to balance the importance of community size and duration, positive float

    list_G = CRC.get_list_G(graph_path, ts, te)  # name of the dataset folder, string
    # print("list_G:", list_G)
    V_MAX = CRC.get_V_max(list_G, k)  # 返回最大的k-core子图的顶点数

    if method == "1":  # EEF
        maxS, C_opt, duration = CRC.EEF(list_G, query_node, theta, k, V_MAX, alpha)  # type: ignore # query: query vertex, string format
    if method == "2":  # WCF
        theta_thres_all = []
        wcf_indices = []
        theta_thres_all = Parallel(n_jobs=n_jobs)(delayed(CRC.theta_thres_table)(g) for g in list_G)
        wcf_indices = Parallel(n_jobs=n_jobs)(delayed(CRC.theta_tree)(theta_thres_all[i], g) for i, g in enumerate(list_G)) # type: ignore

        maxS, C_opt, score, L_c, duration = CRC.WCF_search(list_G, wcf_indices, query_node, theta, k, V_MAX, alpha) # type: ignore

    return maxS, C_opt, duration


# Get query nodes from a file
def get_query_nodes(query_node_path, dataset_name):
    query_node_file = query_node_path + dataset_name + "_query_node.txt"
    query_nodes = []
    
    with open(query_node_file, 'r') as f:
        for line in f:
            query_nodes.append(line.strip())
        f.close()
    
    return query_nodes


def process_combination(graph_path, theta, k, query_node, method, alpha, start_time, end_time):
    max_score, C_opt, duration = query(graph_path, theta, k, query_node, method, alpha, start_time, end_time)
    crc_nodes = list(C_opt)
    return max_score, crc_nodes, [theta, k, end_time, alpha]



if __name__ == '__main__':
    dataset_name = "BTW17" # "BTW17", "Chicago_COVID", "Crawled_Dataset144", "Crawled_Dataset26"
    CRC_dataset_dir = "./Input_Datasets/WCF-CRC_Dataset/"
    query_node_dir = "./Original_Datasets/Query_Nodes/"
    result_dir = "./Algorithm_Output/WCF-CRC_Results/"

    result_file = result_dir + "WCF-CRC_results_" + dataset_name + ".txt"
    graph_path = CRC_dataset_dir + dataset_name + "/"

    # Read the query node list
    query_nodes_list = get_query_nodes(query_node_dir, dataset_name)

    # Set the parameters for WCF-CRC
    theta_list = [theta / 10 for theta in range(0, 5)] # theta: edge weight threshold
    k_list = [2, 3, 4, 5, 6] # k: k-core constraint
    end_time_list = [4, 6, 8, 10] # end_time: ending timestamp of the query interval, for BTW17, end_time_list = [2, 3]
    alpha_list = [alpha for alpha in range(0, 7)] # alpha: parameter to balance the importance of community size and duration

    # Generate all possible combinations of the parameters
    combinations = list(itertools.product(theta_list, k_list, end_time_list, alpha_list))

    start_time = 0
    method = '2'
    n_jobs = -1

    # Check if the results file exists
    try:
        with open(result_file, 'r') as f:
            pass
    except FileNotFoundError:
        with open(result_file, 'w') as f:
            f.close()

    # For each query node, try all combinations of theta, k, end_time, and alpha
    for query_node in tqdm.tqdm(query_nodes_list):
        query_start_time = time.time()

       # Use Parallel and delayed to process combinations in parallel
        processed_results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(process_combination)(graph_path, theta, k, query_node, method, alpha, start_time, end_time)
            for theta, k, end_time, alpha in combinations)

        # Save the results after trying all combinations of parameters for the query node
        for max_score, crc_nodes, params in processed_results: # type: ignore
            with open(result_file, 'a') as f:
                f.write(f"{query_node}\t{max_score}\t{params}\t{crc_nodes}\n")
                f.close()
        
        query_end_time = time.time()
    
    print("All query nodes processed!")
    print("Results saved to: " + result_file)