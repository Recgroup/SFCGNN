import torch
from data import *
import torch.nn.functional as F

def generateAugAdj(preds,adj,topk,t,augAdj):
    preds = preds.topk(topk)[-1]
    adj_dense = adj.to_dense()
    new_adjs = [adj]
    adj_matrixs = torch.zeros_like(adj_dense)>0
    for i in range(topk):
        adjacency_matrix = torch.zeros((preds.size(0), preds.size(0)), dtype=torch.int)
        category_matrix = preds[:,i].unsqueeze(1).expand(preds.size(0), preds.size(0))
        adj_matrixs += (category_matrix == category_matrix.t())

    if t<=1:
        adj = adj_dense
    else:
        for i in range(t-1):
            adj = adj.mm(adj_dense)

    if augAdj == 0:
        adj = adj_dense*adj_matrixs

    adj = adj*adj_matrixs
    new_adj = torch.where(adj.cpu()>0,torch.ones(1),torch.zeros(1)).numpy()
    new_adjs.append(sparse_mx_to_torch_sparse_tensor(normalize_adj(new_adj)).to(adj.device))
    return new_adjs
