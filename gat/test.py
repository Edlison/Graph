# @Author  : Edlison
# @Date    : 3/4/22 22:22
import torch
import numpy as np
import scipy.sparse as sp
from utils import encode_onehot, normalize_adj, normalize_features


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)  # [2708, 1433]
    labels = encode_onehot(idx_features_labels[:, -1])  # [2708,]
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)  # [2708,]

    # build graph
    idx_map = {j: i for i, j in enumerate(idx)}  # paper id -> 索引
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)  # [5429, 2]
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)  # 所有边信息中的paper id -> 索引
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)  # 构建邻接矩阵(有向图)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)  # 对称邻接矩阵 也就是无向图

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))  # 加上一个对角矩阵 自己与自己关联

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))  # 从3元组转为2d
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


if __name__ == '__main__':
    a = np.arange(0, 3).reshape(3, 1)
    b = np.arange(0, 3).reshape(1, 3)
    res = a + a.T
    print(a)
    print(a.T)
    print(res)
    ...