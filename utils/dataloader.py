from itertools import chain

import numpy as np
import torch
from torch.utils.data import Dataset


# max_length = 50


def compute_unique_max_len(datas):
    return max([len(set(data)) for data in datas])


class DataSet(Dataset):
    def __init__(self, rawData):
        self.data = rawData[0]
        self.target = rawData[1]
        self.max_length = compute_max_len(self.data)
        self.unique_max_length = compute_unique_max_len(self.data)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        # padding
        data = torch.cat(
            [torch.tensor(self.data[index]), torch.zeros(self.max_length - len(self.data[index]), dtype=torch.int64)])
        label = torch.tensor(self.target[index]) - 1
        mask = torch.cat([torch.ones([len(self.data[index])], dtype=torch.float32),
                          torch.zeros(self.max_length - len(self.data[index]), dtype=torch.float32)], dim=0)
        # build graph
        unique_node, alias_index = torch.unique(data, return_inverse=True)
        item = torch.concat(
            [unique_node, torch.zeros(self.unique_max_length - unique_node.shape[0], dtype=torch.int64)], dim=-1)
        A = torch.zeros((self.unique_max_length, self.unique_max_length), dtype=torch.float32)
        for i in range(data.shape[0] - 1):
            if data[i + 1] == 0:
                break
            u = torch.where(item == data[i])[0][0]
            v = torch.where(item == data[i + 1])[0][0]
            A[u][v] = 1  # 可以赋权值试试
        A_in_sum = torch.sum(A, 0)
        A_in_clip = torch.clip(A_in_sum, 1)
        A_in = torch.divide(A, A_in_clip)
        A_out_sum = torch.sum(A, 1)
        A_out_clip = torch.clip(A_out_sum, 1)
        A_out = torch.divide(A.transpose(1, 0), A_out_clip)
        A = torch.concat([A_in, A_out], 0).transpose(1, 0)
        return alias_index, A, item, label, mask


def compute_item_num(sequence):
    seq_in_1D = list(chain.from_iterable(sequence))
    items_num = len(np.unique(seq_in_1D))
    return items_num


def compute_max_len(sequence):
    len_list = [len(seq) for seq in sequence]
    return np.max(len_list)


def split_train_val(train_data, split_rate=0.1):
    session_total = len(train_data[0])
    split_num = int(session_total * split_rate)

    val_index = np.random.choice(a=np.arange(0, session_total), size=split_num, replace=False)
    np.random.shuffle(val_index)
    val_data = ([train_data[0][index] for index in val_index], [train_data[1][index] for index in val_index])

    train_index = np.setdiff1d(np.arange(0, session_total), val_index)
    train_data_new = ([train_data[0][index] for index in train_index], [train_data[1][index] for index in train_index])

    return train_data_new, val_data
