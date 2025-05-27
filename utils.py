import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from scipy.sparse import coo_matrix
from models import NewHyperData
import torch
import os
from typing import Tuple
import subprocess
from scipy.sparse.linalg import svds
import kahypar as kahypar

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
    # rowsum = np.array(adj_matrix.sum(1)).squeeze(1)
    # data = -adj_matrix.data
    # laplace_data = np.concatenate((data, rowsum))
    # diag = np.arange(len(rowsum))
    # laplace = coo_matrix((laplace_data, (np.concatenate((adj_matrix.row, diag)), np.concatenate((adj_matrix.col, diag)))), shape=(len(rowsum), len(rowsum)))
    # if adj_normalize:
    #    laplace = normalize_adj(laplace)
    # lamb, X = eigs(laplace, d)
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

def preprocess_data(hypergraph_vertices, hypergraph_edges, filename, num_nodes, num_nets):
    adj_matrix, node_degree, pin_count = create_clique_expansion_graph(hypergraph_vertices, hypergraph_edges)
    clique_topo_features = compute_topological_features(adj_matrix, 2, True, False)
    row, col, value = [], [], []
    for i, e in enumerate(hypergraph_edges):
        for v in e:
            row.append(v)
            col.append(i)
            value.append(1)
    H = coo_matrix((value, (row, col)), shape=(num_nodes, num_nets), dtype=float)
    U, S, Vt = svds(H, k=2, which='LM', random_state=42, solver='lobpcg', maxiter=10000)
    U = U[:, np.argsort(S)[::-1]]
    for i in range(U.shape[1]):    
        if U[np.argmax(np.absolute(U[:,i])),i] < 0:
            U[:,i] = -U[:,i]
    star_topo_features = torch.tensor(U.copy(), dtype=torch.float)
    partition_feature = create_partition_id_feature(len(hypergraph_vertices), filename)
    features = np.column_stack([clique_topo_features, star_topo_features, node_degree, pin_count, partition_feature])
    del adj_matrix, node_degree, pin_count, clique_topo_features, star_topo_features, partition_feature, H, U, S, Vt
    deg_feature_norm = np.linalg.norm(features[:, 4])
    features[:, 0] = features[:, 0] / np.linalg.norm(features[:, 0]) * deg_feature_norm
    features[:, 1] = features[:, 1] / np.linalg.norm(features[:, 1]) * deg_feature_norm
    features[:, 2] = features[:, 2] / np.linalg.norm(features[:, 2]) * deg_feature_norm
    features[:, 3] = features[:, 3] / np.linalg.norm(features[:, 3]) * deg_feature_norm
    features[:, 5] = features[:, 5] / np.linalg.norm(features[:, 5]) * deg_feature_norm
    features[:, 6] = features[:, 6] / np.linalg.norm(features[:, 6]) * deg_feature_norm
    hyperedge_index = torch.tensor(np.array([
        np.concatenate(hypergraph_edges),
        np.repeat(np.arange(len(hypergraph_edges)), [len(e) for e in hypergraph_edges])
    ]), dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float)
    data = NewHyperData(x=x, hyperedge_index=hyperedge_index)
    return data

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

def evalPoint(id: int, partition, hypergraph_vertices, hypergraph_edges, num_partitions, filename, use_kahypar=False):
    num_nodes = len(hypergraph_vertices)
    num_nets = len(hypergraph_edges)
    with open(f'./data/{filename}.part.2.{id}', 'w') as f:
        for i in range(len(partition)):
            f.write(f'{partition[i]}\n')
    command = ['./exec/EasyPart', '-g', f'./data/{filename}', '-e', '0.04', '-p', '2', '-t', '1', '-m', 'quality', '-f', f'./data/{filename}.part.2.{id}', '-o', '1']
    channel = subprocess.DEVNULL
    try:
        subprocess.run(command, shell=False, stdout=channel, stderr=channel)
    except Exception:
        pass

    # command = ['./exec/KaHyPar', '-h', f'./data/{filename}', '-k', '2', '-e', '0.04', '-o', 'cut', '-m', 'direct', '-p', './exec/cut_kKaHyPar_sea20.ini', '--part-file', f'./data/{filename}.part.2.{id}', '-w', '1']
    # try:
    #     subprocess.run(command, shell=False, stdout=channel, stderr=channel)
    # except Exception:
    #     pass

    if use_kahypar:
        hyperedge_indices = []
        hyperedges = []
        index = 0
        for edge in hypergraph_edges:
            hyperedge_indices.append(index)
            index += len(edge)
            hyperedges.extend(edge)
        hyperedge_indices.append(index)
        hypergraph = kahypar.Hypergraph(num_nodes, num_nets, hyperedge_indices, hyperedges, num_partitions)
        context = kahypar.Context()
        context.loadINIconfiguration("./exec/cut_kKaHyPar_sea20.ini")
        context.setK(num_partitions)
        context.setEpsilon(0.04)
        context.setInputPartitionFileName(f'./data/{filename}.part.2.{id}')
        context.writePartitionFile(True)
        context.setPartitionFileName(f'./data/{filename}.part.2.{id}')
        context.suppressOutput(True)
        kahypar.partition(hypergraph, context)

    partition_id = np.zeros(num_nodes)
    index = 0
    with open(f'./data/{filename}.part.2.{id}', 'r') as f:
        for line in f:
            partition_id[index] = int(line.strip())
            index += 1
    command = ['rm', f'./data/{filename}.part.2.{id}']
    try:
        subprocess.run(command, shell=False, stdout=channel, stderr=channel)
    except Exception:
        pass
    cut, imbalance = evaluate_partition(partition_id, hypergraph_vertices, hypergraph_edges, num_partitions)
    return cut, imbalance, partition_id

def basic_randomized_svd(X, k, q=2, s=10):
    """
    计算一个矩阵的近似低秩 SVD 分解。
    这份代码是基于 "Finding structure with randomness: 
    Probabilistic algorithms for constructing approximate matrix decompositions" 
    一文中的 Algorithm 2。

    参数:
    ----------
    X : numpy.ndarray
        输入的 m x n 矩阵。
    k : int
        目标秩，即要计算的奇异值数量。
    q : int, optional
        功率迭代的次数 (默认为 1)。
        用于提高近似精度的可选步骤，q=0, 1, 或 2 通常就足够了。
    s : int, optional
        过采样参数 (oversampling parameter)，(默认为 10)。
        k + s 不应大于 m 或 n。

    返回:
    -------
    U : numpy.ndarray
        m x k 的左奇异向量矩阵。
    Sigma : numpy.ndarray
        k x k 的对角矩阵，包含前 k 个奇异值。
    V : numpy.ndarray
        n x k 的右奇异向量矩阵。
    """
    # 获取 X 的维度
    m, n = X.shape

    # 确保 k+s 的大小是合理的
    if k + s > min(m, n):
        raise ValueError("k + s 必须小于或等于 X 的最小维度")

    # --- 算法步骤 ---

    # 第 2 行: 生成一个 n x (k+s) 的随机高斯矩阵 Ω
    Omega = np.random.randn(n, k + s)

    # 第 3 行: 计算素描矩阵 Y = XΩ
    Y = X @ Omega

    # 第 4 行: 计算 Y 的 QR 分解，得到 Q
    # np.linalg.qr 提供了一个正交基 Q
    Q, _ = np.linalg.qr(Y)

    # 第 5-7 行: 可选的功率迭代 (Power Iteration)
    # 这个循环可以提高近似的准确度
    for i in range(q):
        T, _ = np.linalg.qr(X.T @ Q)  # 第 6 行
        Q, _ = np.linalg.qr(X @ T)  # 第 7 行

    # 第 8 行: 将 X 投影到 Q 的基上，形成较小的矩阵 B
    B = Q.T @ X

    # 第 9 行: 计算 B 的完整 SVD
    # np.linalg.svd 返回的 V 是 V.T (伪代码中的 V_hat)
    U_hat, S_hat, V_hat_T = np.linalg.svd(B, full_matrices=False)

    # 第 10 行: 截断并重构原始矩阵的 SVD
    # 获取前 k 个分量
    U = Q @ U_hat[:, :k]
    Sigma = np.diag(S_hat[:k])
    V = V_hat_T[:k, :].T # V_hat_T 的前 k 列是 V 的前 k 行

    # 第 11 行: 返回 U, Σ, V
    return U, Sigma, V

def generate_sparse_sign_matrix(n, h, z):
    """
    根据 Algorithm 3 生成一个 n x h 的稀疏符号矩阵。

    参数:
    ----------
    n : int
        矩阵的行数。
    h : int
        矩阵的列数 (即素描的大小)。
    z : int
        每列的非零元素数量 (列密度)。

    返回:
    -------
    S : numpy.ndarray
        生成的 n x h 稀疏符号矩阵。
    """
    # 第 2 行: 初始化一个 n x h 的零矩阵
    S = np.zeros((n, h), dtype=np.int8)
    
    # 第 4 行: 遍历矩阵的每一列
    for j in range(h):
        # 第 5 行: 从 n 行中随机选择 z 个不重复的索引
        # randperm(n, z) 相当于从 0 到 n-1 中不重复地抽取 z 个数
        indices = np.random.choice(n, size=z, replace=False)
        
        # 第 6 行: 在选定的索引位置上填充 +1 或 -1
        # sign(randn(z, 1)) 是一种生成随机符号的方式
        signs = np.random.choice([-1, 1], size=z)
        
        S[indices, j] = signs
        
    # 第 8 行: 返回生成的稀疏矩阵 S
    # 伪代码中的 p 在此应用中不是必需的，因此我们只返回 S
    return S

def randomized_svd_with_sparse_matrix(X, k, q=2, s=10):
    """
    计算一个矩阵的近似低秩 SVD 分解。
    此版本使用稀疏符号矩阵进行素描，以提高计算效率。

    参数:
    ----------
    X : numpy.ndarray
        输入的 m x n 矩陣。
    k : int
        目标秩，即要计算的奇异值数量。
    q : int, optional
        功率迭代的次数 (默认为 1)。
    s : int, optional
        过采样参数 (默认为 10)。

    返回:
    -------
    U : numpy.ndarray
        m x k 的左奇异向量矩阵。
    Sigma : numpy.ndarray
        k x k 的对角矩阵，包含前 k 个奇异值。
    V : numpy.ndarray
        n x k 的右奇异向量矩阵。
    """
    m, n = X.shape
    
    if k + s > min(m, n):
        raise ValueError("k + s 必须小于或等于 X 的最小维度")

    # --- 核心修改部分 ---
    # 使用稀疏符号矩阵替换密集高斯矩阵
    h = k + s
    z = min(8, n)  # 根据要求设置列密度
    Omega = generate_sparse_sign_matrix(n, h, z)
    # --- 修改结束 ---

    # 后续步骤与原算法完全相同
    Y = X @ Omega
    Q, _ = np.linalg.qr(Y)

    for i in range(q):
        T, _ = np.linalg.qr(X.T @ Q)
        Q, _ = np.linalg.qr(X @ T)

    B = Q.T @ X
    U_hat, S_hat, V_hat_T = np.linalg.svd(B, full_matrices=False)

    U = Q @ U_hat[:, :k]
    Sigma = np.diag(S_hat[:k])
    V = V_hat_T[:k, :].T

    return U, Sigma, V
