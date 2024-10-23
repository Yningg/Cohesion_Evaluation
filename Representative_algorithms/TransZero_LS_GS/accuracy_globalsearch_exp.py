import torch
from utils import f1_score_calculation, load_query, cosin_similarity, get_gt_legnth, evaluation
import argparse
import numpy as np
from tqdm import tqdm
from numpy import *
import time

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

def subgraph_density_controled(candidate_score, graph_score):
    
    weight_gain = (sum(candidate_score)-sum(graph_score)*(len(candidate_score)**1)/(len(graph_score)**1))/(len(candidate_score)**0.50)
    return weight_gain

def GlobalSearch(query_index, graph_score):

    candidates = query_index
    selected_candidate = candidates

    graph_score=np.array(graph_score)
    max2min_index = np.argsort(-graph_score)
    
    startpoint = 0
    endpoint = int(0.50*len(max2min_index))
    if endpoint >= 10000:
        endpoint = 10000
    
    while True:
        candidates_half = query_index+[max2min_index[i] for i in range(0, int((startpoint+endpoint)/2))]
        candidate_score_half = [graph_score[i] for i in candidates_half]
        candidates_density_half = subgraph_density_controled(candidate_score_half, graph_score)

        candidates = query_index+[max2min_index[i] for i in range(0, endpoint)]
        candidate_score = [graph_score[i] for i in candidates]
        candidates_density = subgraph_density_controled(candidate_score, graph_score)

        if candidates_density>= candidates_density_half:
            startpoint = int((startpoint+endpoint)/2)
            endpoint = endpoint
        else:
            startpoint = startpoint
            endpoint = int((startpoint+endpoint)/2)
        
        if startpoint == endpoint or startpoint+1 == endpoint:
            break

    selected_candidate = query_index+[max2min_index[i] for i in range(0, startpoint)] 
    
    return selected_candidate

if __name__ == "__main__":
    args = parse_args()
    print(args)


    if args.embedding_tensor_name is None:
        args.embedding_tensor_name = args.dataset
    

    embedding_tensor = torch.from_numpy(np.load(args.EmbeddingPath + args.embedding_tensor_name + '.npy'))
    
    # load queries and labels
    query = load_query("D:/Cohesion_Evaluation/Input_Datasets/TransZero_LS_GS_Dataset/", args.dataset, embedding_tensor.shape[0])

    start = time.time()
    query_feature = torch.mm(query, embedding_tensor) # (query_num, embedding_dim)
    
    query_num = torch.sum(query, dim=1)
    query_feature = torch.div(query_feature, query_num.view(-1, 1))
    
    # cosine similarity
    print(query_feature.shape, embedding_tensor.shape)
    
    query_score = cosin_similarity(query_feature, embedding_tensor) # (query_num, node_num)
    query_score = torch.nn.functional.normalize(query_score, dim=1, p=1)

    print("query_score.shape: ", query_score.shape)

    with open("D:/Cohesion_Evaluation/Algorithm_Output/TransZero_GS_Results/TransZero_globalsearch_results" +  args.dataset + ".txt", "w") as f:
        for i in tqdm(range(query_score.shape[0])):
            query_index = (torch.nonzero(query[i]).squeeze()).reshape(-1)
            selected_candidates = GlobalSearch(query_index.tolist(), query_score[i].tolist()) 
            f.write(f"{query_index.tolist()}\t{selected_candidates}\n")
            print(f"Query node {query_index.tolist()}: {selected_candidates}")
        
    end = time.time()
    print("The global search using time: {:.4f}".format(end-start)) 
    print("The global search using time (one query): {:.4f}".format((end-start)/query_feature.shape[0])) 

