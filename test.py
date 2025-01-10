import argparse
import torch
import numpy as np
import time
import multiprocessing as mp
import subprocess
from transformers import set_seed
from utils import *
from models import GraphPartitionModel
import kahypar as kahypar

def evalPoint(id: int, partition, hypergraph_vertices, hypergraph_edges, num_partitions):
    with open(f'./data/{filename}.part.2.{id}', 'w') as f:
        for i in range(len(partition)):
            f.write(f'{partition[i]}\n')
    command = ['./exec/EasyPart', '-g', f'./data/{filename}', '-e', '0.04', '-p', '2', '-t', '1', '-m', 'quality', '-f', f'./data/{filename}.part.2.{id}', '-o', '1']
    channel = subprocess.DEVNULL
    try:
        subprocess.run(command, shell=False, stdout=channel, stderr=channel)
    except Exception:
        pass

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

if __name__ == '__main__':
    torch.cuda.set_device(0)
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='ibm02.hgr')
    parser.add_argument('--modelname', type=str, default='model')
    parser.add_argument('--num_partitions', type=int, default=2)
    args = parser.parse_args()
    filename = args.filename
    modelname = args.modelname
    num_partitions = args.num_partitions
    preprocess_time = 0
    t0 = time.time()
    with open(f'./data/{filename}', 'r') as f:
        lines = f.readlines()
        num_nets, num_nodes = map(int, lines[0].split())
        hypergraph_vertices = list(range(num_nodes))
        hypergraph_edges = []
        for line in lines[1:]:
            hypergraph_edges.append([int(node) - 1 for node in line.split()])
    data, D, best_partition_id = preprocess_data(hypergraph_vertices, hypergraph_edges, filename)
    data = data.to(device)
    D = D.to(device)
    W = torch.sparse_coo_tensor(data.edge_index, data.edge_attr, (len(data.x), len(data.x)))
    W = W.to(device)
    t1 = time.time()
    preprocess_time += t1 - t0
    model = GraphPartitionModel()
    model.load_state_dict(torch.load(f'./models/{modelname}.pt'))
    model = model.to(device)
    model.eval()
    best_cut, best_imbalance = evaluate_partition(best_partition_id, hypergraph_vertices, hypergraph_edges, num_partitions)
    reason_time = 0
    vcycle_time = 0
    for tau in range(11):
        tau_best_cut = float('inf')
        t0 = time.time()
        partitions = model.sample(data, m=12)
        t1 = time.time()
        reason_time += t1 - t0
        t0 = time.time()
        processes = []
        pool = mp.Pool(processes=12)
        for i in range(12):
            processes.append(pool.apply_async(evalPoint, args=(i, partitions[i], hypergraph_vertices, hypergraph_edges, num_partitions)))
            time.sleep(0.01)
        pool.close()
        pool.join()
        for process in processes:
            cut, imbalance, partition_id = process.get()
            tau_best_cut = min(tau_best_cut, cut)
            if cut < best_cut:
                best_cut = cut
                best_imbalance = imbalance
                best_partition_id = partition_id
        t1 = time.time()
        vcycle_time += t1 - t0
        print(f'Tau: {tau}, Best Cut: {best_cut}, Imbalance: {best_imbalance:.3f}, Cut: {tau_best_cut}')
        best_partition_id = best_partition_id / np.linalg.norm(best_partition_id) * np.linalg.norm(data.x[:, 2].cpu().numpy())
        data.x[:, 4] = torch.tensor(best_partition_id, dtype=torch.float).to(device)
    print(f'Best Cut: {best_cut}, Imbalance: {best_imbalance:.3f}, Preprocess Time: {preprocess_time:.3f}, Reasoning Time: {reason_time:.3f}, V-Cycle Time: {vcycle_time:.3f}')