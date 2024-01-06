import math

import torch
from torch import nn
import torch.nn.functional as F
from entmax import entmax_bisect


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
        self.bias_in = nn.Parameter(torch.Tensor(self.emb_size * 2))
        self.bias_out = nn.Parameter(torch.Tensor(self.emb_size * 2))

    def forward(self, A, X):
        A_in, A_out = torch.chunk(A, chunks=2, dim=-1)
        adj_in = torch.matmul(A_in, self.linear_H_in(X)) + self.bias_in
        adj_out = torch.matmul(A_out, self.linear_H_out(X)) + self.bias_out
        inputs = torch.concat([adj_in, adj_out], -1)
        return self.gnn_cell(inputs)


class S_ATT(nn.Module):
    def __init__(self, emb_size, drop_out, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_size = emb_size
        self.linear_q = nn.Linear(2 * self.emb_size, 2 * self.emb_size)
        self.drop_out = drop_out
        self.scale = math.sqrt(self.emb_size * 2)

        self.ffn = nn.Sequential(nn.Linear(2 * self.emb_size, 2 * self.emb_size),
                                 nn.ReLU(),
                                 nn.Linear(2 * self.emb_size, 2 * self.emb_size),
                                 nn.Dropout(self.drop_out))
        self.layer_norm = nn.LayerNorm(2 * self.emb_size, eps=1e-12)

    def forward(self, X):
        X_target_plus = torch.concat([X, torch.zeros((1,), dtype=torch.float32, device='cuda')], -1)
        q = torch.relu(self.linear_q(X_target_plus))  # q drop
        k = X_target_plus
        v = X_target_plus
        # 这里舍弃掉了序列式的注意
        attention_score = torch.matmul(q, k.transpose(1, 2)) / self.scale
        # alpha
        # [1, 2]
        attention_score = entmax_bisect(attention_score, alpha=alpha, dim=-1)
        attention_result = torch.matmul(attention_score, v)
        C_hat = self.layer_norm(self.ffn(attention_result) + attention_result)
        C, target_emb = torch.chunk(C_hat, chunks=2, dim=1)
        return C, target_emb
        # target_emb = C_hat[:, -1, :]
        # C = C_hat[:, :-1, :]


class TargetEnhance(nn.Module):
    def __init__(self, emb_size, gamma, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_size = emb_size
        self.gamma = gamma
        self.linear_c = nn.Linear(self.emb_size * 4, self.emb_size * 2, bias=False)

    def forward(self, C_last, C_target):
        Sim = F.cosine_similarity(C_last, C_target)
        threshold = self.gamma * torch.mean(torch.sum(Sim,0))
        condition = (Sim - threshold) > 0
        positive = self.linear_c(torch.concat(C_target, C_last))
        C_target = torch.where(condition, positive, C_target)
        return C_target


class G_ATT(nn.Module):
    def __init__(self, emb_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_size = emb_size
        self.linear_Wg1 = nn.Linear(self.emb_size * 2, self.emb_size * 2, bias=False)
        self.linear_Wg2 = nn.Linear(self.emb_size * 2, self.emb_size * 2, bias=False)
        self.linear_Wg0 = nn.Linear(self.emb_size * 2, self.emb_size * 2, bias=False)
        self.bias_alpha = nn.Parameter(torch.Tensor(self.emb_size * 2))

    def forward(self, C_target, C, X):
        q = C_target
        k = C
        v = X
        attention_score = self.linear_Wg0(torch.relu(self.linear_Wg1(k) + self.linear_Wg2(v) + self.bias_alpha))
        entmax_bisect(attention_score, alpha=, dim=-1)


class TASI_GNN(nn.Module):
    def __init__(self, emb_size, item_num, max_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_size = emb_size
        self.item_num = item_num
        self.max_len = max_len
        self.ggnn_layer = GGNN(self.emb_size)
        self.item_embedding = nn.Embedding(self.item_num, self.emb_size, padding_idx=0, max_norm=1.5)
        self.position_embedding = nn.Embedding(self.max_len, self.emb_size, max_norm=1.5)
        self.linear_alpha = nn.Linear(self.emb_size, 1)


    def forward(self):
        alpha = torch.sigmoid(self.linear_alpha(X_target_plus[:, -1, :])) + 1