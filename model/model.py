import math

import torch
from torch import nn
import torch.nn.functional as F
from entmax import entmax_bisect


class GGNN(nn.Module):
    def __init__(self, emb_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_size = emb_size
        self.gnn_cell = nn.GRUCell(input_size=4 * self.emb_size,
                                   hidden_size=2 * self.emb_size,
                                   bias=True)
        self.linear_H_in = nn.Linear(self.emb_size * 2, self.emb_size * 2,
                                     bias=False)
        self.linear_H_out = nn.Linear(self.emb_size * 2, self.emb_size * 2,
                                      bias=False)
        self.bias_in = nn.Parameter(torch.Tensor(self.emb_size * 2))
        self.bias_out = nn.Parameter(torch.Tensor(self.emb_size * 2))

    def forward(self, A, X):
        batch_size = A.shape[0]
        length = A.shape[1]
        A_in, A_out = torch.chunk(A, chunks=2, dim=-1)
        adj_in = torch.matmul(A_in, self.linear_H_in(X)) + self.bias_in
        adj_out = torch.matmul(A_out, self.linear_H_out(X)) + self.bias_out
        inputs = torch.concat([adj_in, adj_out], -1)
        inputs = inputs.contiguous().view([batch_size * length, -1])
        result = self.gnn_cell(inputs, X.contiguous().view([batch_size * length, -1])).contiguous().view(
            [batch_size, length, -1])
        return result


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
        self.drop_q = nn.Dropout(self.drop_out)
        self.drop_attn = nn.Dropout(self.drop_out)

    def forward(self, X_target_plus, alpha):
        q = self.drop_q(torch.relu(self.linear_q(X_target_plus)))  # q drop
        k = X_target_plus
        v = X_target_plus
        attention_score = torch.matmul(q, k.transpose(1, 2)) / self.scale
        attention_score = entmax_bisect(attention_score, alpha=alpha, dim=-1)
        attention_score = self.drop_attn(attention_score)
        attention_result = torch.matmul(attention_score, v)
        C_hat = self.layer_norm(self.ffn(attention_result) + attention_result)
        target_emb = C_hat[:, -1, :]
        C = C_hat[:, :-1, :]
        return C, target_emb


class TargetEnhance(nn.Module):
    def __init__(self, emb_size, gamma, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_size = emb_size
        self.gamma = gamma
        self.linear_c = nn.Linear(self.emb_size * 4, self.emb_size * 2, bias=False)

    def forward(self, C_last, C_target):
        Sim = F.cosine_similarity(C_last, C_target)
        threshold = self.gamma * torch.mean(Sim, 0)
        condition = (Sim - threshold) > 0
        positive = self.linear_c(torch.concat([C_target, C_last], dim=-1))
        C_target = torch.where(condition.unsqueeze(1), positive, C_target)
        return C_target


class G_ATT(nn.Module):
    def __init__(self, emb_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_size = emb_size
        self.linear_Wg1 = nn.Linear(self.emb_size * 2, self.emb_size * 2, bias=False)
        self.linear_Wg2 = nn.Linear(self.emb_size * 2, self.emb_size * 2, bias=False)
        self.linear_Wg0 = nn.Linear(self.emb_size * 2, 1, bias=False)
        self.bias_alpha = nn.Parameter(torch.Tensor(self.emb_size * 2))

    def forward(self, C_target, C, X, alpha, mask):
        q = C_target.unsqueeze(1)
        k = C
        v = X  # 注意这里不一样
        attention_score = self.linear_Wg0(torch.relu(self.linear_Wg1(k) + self.linear_Wg2(q) + self.bias_alpha))
        # mask
        attention_score = attention_score.masked_fill(mask.unsqueeze(-1) == 0, -torch.inf)
        attention_score = entmax_bisect(attention_score, alpha=alpha, dim=1)
        attention_result = torch.matmul(attention_score.transpose(1, 2), v)
        return attention_result


class SimilarIntent(nn.Module):
    def __init__(self, theta, top_k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.theta = theta
        self.top_k = top_k
        self.dropout = nn.Dropout(0.4)

    def forward(self, h):
        Sim = F.cosine_similarity(h.unsqueeze(0), h.unsqueeze(1), dim=-1)  # 1 512 100   512 1 100  --> 512 512
        if h.shape[0] < self.top_k:  # 最后一批次可能小过下限
            self.top_k = h.shape[0]
        sim_topK, indices_topK = torch.topk(Sim, k=self.top_k, dim=-1)
        beta = torch.softmax(self.theta * sim_topK, dim=-1)  # 512 5
        h_topK = h[indices_topK]  # 512 5 100
        beta = beta.unsqueeze(-1)
        h_sim = self.dropout(torch.sum(beta * h_topK, 1))
        return h_sim


class TASI_GNN(nn.Module):
    def __init__(self, emb_size, item_num, max_len, drop_out, gamma, theta, top_k, omega, tau, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_size = emb_size
        self.item_num = item_num
        self.max_len = max_len
        self.dropout = drop_out
        self.gamma = gamma
        self.ggnn_layer = GGNN(self.emb_size)
        self.item_embedding = nn.Embedding(self.item_num + 1, self.emb_size, padding_idx=0, max_norm=1.5)
        self.position_embedding = nn.Embedding(self.max_len, self.emb_size, max_norm=1.5)
        self.linear_alpha = nn.Linear(self.emb_size * 2, 1)
        self.sa_layer = S_ATT(self.emb_size, self.dropout)
        self.target_enhance_layer = TargetEnhance(self.emb_size, self.gamma)
        self.ga_layer = G_ATT(self.emb_size)
        self.linear_wh = nn.Linear(self.emb_size * 4, self.emb_size)
        self.similar_intent_layer = SimilarIntent(theta, top_k)
        self.omega = omega
        self.tau = tau

    def forward(self, alias_index, A, item, mask):
        batch_size = mask.shape[0]
        unique_len = item.shape[1]
        item_emb = self.item_embedding(item)
        pos_emb = self.position_embedding(torch.arange(unique_len, device='cuda', dtype=torch.int64)).unsqueeze(
            0).repeat(batch_size, 1, 1)
        X = torch.concat([item_emb, pos_emb], dim=-1)
        X = self.ggnn_layer(A, X)

        # rebuild to seq
        session = X[torch.arange(batch_size, dtype=torch.int64, device='cuda').unsqueeze(1), alias_index,
                  :]  # broadcast

        # self-attention
        X_target_plus = torch.concat(
            [session, torch.zeros((batch_size, 1, self.emb_size * 2), dtype=torch.float32, device='cuda')], 1)
        alpha = torch.sigmoid(self.linear_alpha(X_target_plus[:, -1, :])) + 1  # [1, 2]
        alpha = torch.clip(alpha, min=1 + 1e-5).unsqueeze(1).expand(-1, X_target_plus.shape[1],
                                                                    -1)  # entmax不能实现alpha=1时的softmax,所以给个裁切
        C, target = self.sa_layer(X_target_plus, alpha)

        # target enhance
        seq_len = torch.sum(mask, -1).to(torch.int64) - 1
        C_last = C[torch.arange(batch_size, dtype=torch.int64, device='cuda'), seq_len, :]
        target = self.target_enhance_layer(C_last, target)

        # global-attention
        alpha = torch.sigmoid(self.linear_alpha(target)) + 1  # [1, 2]
        alpha = torch.clip(alpha, 1 + 1e-5).unsqueeze(1)
        C_global = self.ga_layer(target, C, session, alpha, mask)

        # decoder
        session_hidden = torch.selu(self.linear_wh(torch.concat([C_global.squeeze(1), target], dim=-1)))
        session_hidden_sim = self.similar_intent_layer(session_hidden)

        # predict
        session_final = F.normalize(session_hidden, dim=-1) + self.omega * session_hidden_sim
        item_emb_norm = F.normalize(
            self.item_embedding(torch.arange(1, self.item_num + 1, device='cuda', dtype=torch.int64)), dim=-1)
        scores = torch.matmul(session_final, item_emb_norm.transpose(1, 0))  # / self.tau

        return scores
