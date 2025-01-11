import dgl
import torch
import scipy.sparse as sp
import utils

def get_dataset(dataset, pe_dim):
    if dataset in {"BTW17", "Chicago_COVID", "Crawled_Dataset144", "Crawled_Dataset26"}:
        file_path = "D:/Cohesion_Evaluation/Input_Datasets/TransZero_LS_GS_Dataset/" + dataset + ".pt"
        print("D:/Cohesion_Evaluation/Input_Datasets/TransZero_LS_GS_Dataset/" + dataset + ".pt")
        print("Loading....")

        data_list = torch.load(file_path)
       
        adj = data_list[0]
        
        features = data_list[1]
        
        adj_scipy = utils.torch_adj_to_scipy(adj)
        graph = dgl.from_scipy(adj_scipy)
        lpe = utils.laplacian_positional_encoding(graph, pe_dim) 
        features = torch.cat((features, lpe), dim=1)
    
    else:
        raise ValueError(f"Dataset {dataset} is not recognized.")

    print(type(adj), type(features))
    
    return adj.cpu().type(torch.LongTensor), features.long()




