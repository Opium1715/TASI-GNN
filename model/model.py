import torch
from torch import nn
import torch.nn.functional as F


class GGNN(nn.Module):
    def __init__(self, emb_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_size = emb_size
        self.gnn_cell = nn.GRUCell(input_size=2 * self.emb_size,
                                   hidden_size=2 * self.emb_size,
                                   bias=True)
        self.linear_H_in = nn.Linear(self.emb_size, self.emb_size,
                                     bias=False)
        self.linear_H_out = nn.Linear(self.emb_size, self.emb_size,
                                      bias=False)
        self.bias_in = nn.Parameter(torch.Tensor(self.emb_size))
        self.bias_out = nn.Parameter(torch.Tensor(self.emb_size))

    def forward(self, A, X):
        A_in, A_out = torch.chunk(A, chunks=2, dim=-1)
        adj_in = torch.matmul(A_in, self.linear_H_in(X)) + self.bias_in
        adj_out = torch.matmul(A_out, self.linear_H_out(X)) + self.bias_out
        inputs = torch.concat([adj_in, adj_out], -1)
        return self.gnn_cell(inputs)



class TASI_GNN(nn.Module):
    def __init__(self, emb_size, item_num, max_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_size = emb_size
        self.item_num = item_num
        self.max_len = max_len
        self.ggnn_layer = GGNN(self.emb_size)
        self.item_embedding = nn.Embedding(self.item_num, self.emb_size, padding_idx=0, max_norm=1.5)
        self.position_embedding = nn.Embedding(self.max_len, self.emb_size, max_norm=1.5)
