import torch
from utils_exp import load_query, cosin_similarity, coo_matrix_to_nx_graph_efficient, evaluation
import argparse
import numpy as np
from tqdm import tqdm
from numpy import *
import time
from utils import find_all_neighbors_bynx
import os

def parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()
    # main parameters
    parser.add_argument('--dataset', type=str, default='cora', help='dataset name')
    parser.add_argument('--embedding_tensor_name', type=str, help='embedding tensor name')
    parser.add_argument('--EmbeddingPath', type=str, default='./pretrain_result/', help='embedding path')
    parser.add_argument('--topk', type=int, default=400, help='the number of nodes selected.')

    return parser.parse_args()

def subgraph_density(candidate_score, avg_weight):
    weight_gain = (sum(candidate_score)-len(candidate_score)*avg_weight)/(len(candidate_score)**0.5)
    return weight_gain


def mwg_subgraph_heuristic(query_index, graph_score, graph):

    candidates = query_index

    selected_candidate = candidates
    max_density = -1000

    avg_weight = sum(graph_score)/len(graph_score)

    count = 0
    endpoint = int(0.50*len(graph_score))
    if endpoint >= 10000:
        endpoint = 10000
    
    while True:

        neighbors = find_all_neighbors_bynx(candidates, graph)
        
        if len(neighbors) == 0 or count>endpoint:
            break
        
        # select the index with the largest score.
        neighbor_score = [graph_score[i]for i in neighbors]
        i_index = neighbor_score.index(max(neighbor_score))
        
        candidates = candidates+[neighbors[i_index]]

        candidate_score = [graph_score[i]for i in candidates]
        candidates_density = subgraph_density(candidate_score, avg_weight)
        if candidates_density > max_density:
            max_density = candidates_density
            selected_candidate = candidates
        else:
            break

        count += 1
    
    return selected_candidate

def mwg_subgraph_heuristic_fast(query_index, graph_score, graph):

    candidates = query_index

    selected_candidate = candidates
    max_density = -1000

    avg_weight = sum(graph_score)/len(graph_score)

    count = 0
    endpoint = int(0.50*len(graph_score))
    if endpoint >= 10000:
        endpoint = 10000
    
    current_neighbors = find_all_neighbors_bynx(candidates, graph)
    current_neighbors_score = [graph_score[i]for i in current_neighbors]

    candidate_score = [graph_score[i]for i in candidates]
    
    while True:

        if len(current_neighbors_score)==0 or count>endpoint:
            break
        
        i_index = current_neighbors_score.index(max(current_neighbors_score))
        
        candidates = candidates+[current_neighbors[i_index]]
        candidate_score = candidate_score+[graph_score[current_neighbors[i_index]]]

        candidates_density = subgraph_density(candidate_score, avg_weight)
        if candidates_density > max_density:
            max_density = candidates_density
            selected_candidate = candidates
            
            new_neighbors = find_all_neighbors_bynx([current_neighbors[i_index]], graph)
            
            del current_neighbors[i_index]
            del current_neighbors_score[i_index]

            new_neighbors_unique = list(set(new_neighbors) - set(current_neighbors)-set(candidates))
            
            new_neighbors_score = [graph_score[i]for i in new_neighbors_unique]
            current_neighbors = current_neighbors+new_neighbors_unique
            current_neighbors_score = current_neighbors_score+new_neighbors_score

        else:
            break

        count += 1
    
    return selected_candidate

if __name__ == "__main__":
    args = parse_args()
    print(args)

    if args.embedding_tensor_name is None:
        args.embedding_tensor_name = args.dataset

    embedding_tensor = torch.from_numpy(np.load(args.EmbeddingPath + args.embedding_tensor_name + '.npy'))
    
    # load queries
    query = load_query("/root/Cohesion_Evaluation/Input_Datasets/TransZero_LS_GS_Dataset/", args.dataset, embedding_tensor.shape[0])

    # load adj
    if args.dataset in {"BTW17", "Chicago_COVID", "Crawled_Dataset144", "Crawled_Dataset26"}:
        file_path = '/root/Cohesion_Evaluation/Input_Datasets/TransZero_LS_GS_Dataset/' + args.dataset +'.pt'
    data_list = torch.load(file_path)
    adj = data_list[0]

    graph = coo_matrix_to_nx_graph_efficient(adj)
    
    start = time.time()

    query_feature = torch.mm(query, embedding_tensor) # (query_num, embedding_dim)
    query_num = torch.sum(query, dim=1)
    query_feature = torch.div(query_feature, query_num.view(-1, 1))
    
    # cosine similarity
    query_score = cosin_similarity(query_feature, embedding_tensor) # (query_num, node_num)
    query_score = torch.nn.functional.normalize(query_score, dim=1, p=1)

    
    print("query_score.shape: ", query_score.shape)

    output_dir = "/root/Cohesion_Evaluation/Algorithm_Output/TransZero_GS_Results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(os.path.join(output_dir, "TransZero_GS_results_" + args.dataset + ".txt"), "w") as f:
        for i in tqdm(range(query_score.shape[0])):
            query_index = (torch.nonzero(query[i]).squeeze()).reshape(-1)
            # selected_candidates = mwg_subgraph_heuristic(query_index.tolist(), query_score[i].tolist(), graph)
            selected_candidates = mwg_subgraph_heuristic_fast(query_index.tolist(), query_score[i].tolist(), graph)
            f.write(f"{query_index.tolist()}\t{selected_candidates}\n")
            print(f"Query node {query_index.tolist()}, Community size: {len(selected_candidates)}")
            
    end = time.time()
    print("The local search using time: {:.4f}".format(end-start)) 
    print("The local search using time (one query): {:.4f}".format((end-start)/query_feature.shape[0])) 
