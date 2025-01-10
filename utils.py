import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from scipy.sparse import coo_matrix
from torch_geometric.data import Data
import torch
import os
from typing import Tuple

def create_clique_expansion_graph(hypergraph_vertices, hypergraph_edges) -> Tuple[coo_matrix, np.ndarray, np.ndarray]:
    node_degree = np.zeros(len(hypergraph_vertices))
    pin_count = np.zeros(len(hypergraph_vertices))
    edge_dict = {}
    for edge in hypergraph_edges:
        for i in range(len(edge)):
            for j in range(i + 1, len(edge)):
                node_pair = (edge[i], edge[j])
                node_pair_reversed = (edge[j], edge[i])
                weight = 1 / (len(edge) - 1)
                if node_pair in edge_dict:
                    edge_dict[node_pair] += weight
                    edge_dict[node_pair_reversed] += weight
                else:
                    edge_dict[node_pair] = weight
                    edge_dict[node_pair_reversed] = weight
            pin_count[edge[i]] += 1
    row = []
    col = []
    data = []
    for (i, j), weight in edge_dict.items():
        row.append(i)
        col.append(j)
        data.append(weight)
        node_degree[i] += 1
    num_nodes = len(hypergraph_vertices)
    adj_matrix = coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    return adj_matrix, node_degree, pin_count

def normalize_adj(adj: coo_matrix) -> coo_matrix:
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def compute_topological_features(adj_matrix: coo_matrix, d: int, adj_normalize: bool, feature_abs: bool) -> np.ndarray:
    if adj_normalize:
       adj_matrix = normalize_adj(adj_matrix)
    lamb, X = eigs(adj_matrix, d)
    lamb, X = lamb.real, X.real
    X = X[:, np.argsort(lamb)]
    if feature_abs:
        X = np.abs(X)
    else:
        for i in range(X.shape[1]):    
            if X[np.argmax(np.absolute(X[:,i])),i] < 0:
                X[:,i] = -X[:,i]
    return X

def create_partition_id_feature(num_nodes, filename):
    partition_feature = []
    os.system(f'./exec/hmetis2.0pre1 ./data/{filename} 2 -ufactor=2 -ptype=rb -otype=cut -nruns=1 -seed=42')
    with open(f'./data/{filename}.part.2', 'r') as f:
        for line in f:
            partition_id = int(line.strip())
            partition_feature.append(partition_id)
    os.system(f'rm ./data/{filename}.part.2')
    if len(partition_feature) != num_nodes:
        print('Partition feature length does not match number of nodes!')
    partition_feature = np.array(partition_feature)
    return partition_feature

def normalize_features(features: np.ndarray) -> np.ndarray:
    deg_feature_norm = np.linalg.norm(features[:, 2])
    features[:, 0] = features[:, 0] / np.linalg.norm(features[:, 0]) * deg_feature_norm
    features[:, 1] = features[:, 1] / np.linalg.norm(features[:, 1]) * deg_feature_norm
    features[:, 3] = features[:, 3] / np.linalg.norm(features[:, 3]) * deg_feature_norm
    features[:, 4] = features[:, 4] / np.linalg.norm(features[:, 4]) * deg_feature_norm
    return features

def preprocess_data(hypergraph_vertices, hypergraph_edges, filename):
    adj_matrix, node_degree, pin_count = create_clique_expansion_graph(hypergraph_vertices, hypergraph_edges)
    topo_features = compute_topological_features(adj_matrix, 2, True, False)
    partition_feature = create_partition_id_feature(len(hypergraph_vertices), filename)
    features = np.column_stack([topo_features, node_degree, pin_count, partition_feature])
    normalized_features = normalize_features(features)
    edge_index = torch.tensor(np.array([adj_matrix.row, adj_matrix.col]), dtype=torch.long)
    edge_attr = torch.tensor(adj_matrix.data, dtype=torch.float)
    x = torch.tensor(normalized_features, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    D = torch.tensor(node_degree, dtype=torch.float).unsqueeze(1)
    return data, D, partition_feature

def evaluate_partition(partition: np.ndarray, hypergraph_vertices, hypergraph_edges, num_partitions):
    cut = 0
    partition_weights = np.zeros(num_partitions)
    for node in hypergraph_vertices:
        partition_id = int(partition[node])
        partition_weights[partition_id] += 1
    for edge in hypergraph_edges:
        edge_partitions = set()
        for node in edge:
            edge_partitions.add(partition[node])
        if len(edge_partitions) > 1:
            cut += 1
    imbalance = np.max((partition_weights - np.mean(partition_weights)) / np.mean(partition_weights))
    return cut, imbalance
