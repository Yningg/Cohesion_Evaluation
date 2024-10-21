import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"device : {device}")

def edge_index_to_sparse_coo(edge_index):
    row = edge_index[0].long()
    col = edge_index[1].long()

    # Build the sparse matrix
    num_nodes = torch.max(edge_index) + 1
    size = (num_nodes.item(), num_nodes.item())

    values = torch.ones_like(row)
    edge_index_sparse = torch.sparse_coo_tensor(torch.stack([row, col]), values, size)

    return edge_index_sparse


dataset_str = 'Chicago_COVID'

# Load the edge list and transform it to the format of PyG
file_path = f"D:/Cohesion_Evaluation/Input_Datasets/TransZero_LS_GS_Dataset/{dataset_str}/" + dataset_str + ".edges"

# Refer to edge list, extract edge_index, x, y, x.type = LongTensor
# every edge has two features:timestamp, sentiment
with open(file_path, 'r') as f:
    lines = f.readlines()
    edge_index = []
    edge_attr = []
    x = set()
    

    for line in lines:
        line = line.strip().split(' ')
        node1, node2 = int(line[0]), int(line[1])
        x.add(node1)
        x.add(node2)
        edge_index.append([node1, node2])


# Transform edge_index to the format: first row is the source node, second row is the target node
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
x = torch.tensor(list(x), dtype=torch.long).view(-1, 1)  # Assuming each node has a single feature (its ID)
edge_attr = torch.tensor(edge_attr, dtype=torch.int)

# Create a Data object
graph = Data(x=x, edge_index=edge_index)
graph.y = torch.tensor([])

# print(graph.x, graph.edge_index, graph.y)
print(graph.edge_index)

torch.save([edge_index_to_sparse_coo(graph.edge_index).type(torch.LongTensor), graph.x.type(torch.LongTensor), graph.y.type(torch.LongTensor)], "../dataset/"+dataset_str+".pt")

