import glob
import os
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from torch_sparse import coalesce

from torch_geometric.data import Data
from torch_geometric.io import read_txt_array
from torch_geometric.utils import remove_self_loops, to_networkx, unbatch_edge_index
import networkx as nx
import pynauty as nauty




def pyg_to_pynauty(n, edge_index):
    edges = list(zip(edge_index[0], edge_index[1]))
    G = nauty.Graph(number_of_vertices=n, directed=False)
    for u, v in edges:
        G.connect_vertex(u, v)
    return G

names = [
    'A', 'graph_indicator', 'node_labels', 'node_attributes'
    'edge_labels', 'edge_attributes', 'graph_labels', 'graph_attributes'
]

def compute_pr_and_orbit(data, batch):
    
    num_graphs = batch.max().item() + 1
    num_all_nodes = batch.shape[0]
    pagerank_results = []
    orbit_results = []
    
    num_of_nodes_in_graph = torch.bincount(batch).tolist()
    
    edges = unbatch_edge_index(data.edge_index, batch)

    for g in range(num_graphs):
    
        n = num_of_nodes_in_graph[g]

        edge_index = edges[g]
        

        datag = Data(edge_index=edge_index, num_nodes=n)
        G = to_networkx(datag, to_undirected=True)
        pr = nx.pagerank(G, alpha=0.9)
        node_order = list(G.nodes())

        pr_values = [pr[node] for node in node_order]
        
        G_ = pyg_to_pynauty(n, edge_index)
        generators, grpsize1, grpsize2, orbits, numorbits = nauty.autgrp(G_)
        
        orbit_results.append(torch.tensor(orbits, dtype=torch.float))
        
        pagerank_results.append(torch.tensor(pr_values, dtype=torch.float))

    
    data.pr = torch.cat(pagerank_results)
    data.orbit_label = torch.cat(orbit_results)
    
    assert data.pr.shape[0] == num_all_nodes
    assert data.orbit_label.shape[0] == num_all_nodes

    return data
    
    
def read_tu_data(folder, prefix):
    files = glob.glob(osp.join(folder, f'{prefix}_*.txt'))
    names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in files]

    edge_index = read_file(folder, prefix, 'A', torch.long).t() - 1
    batch = read_file(folder, prefix, 'graph_indicator', torch.long) - 1

    node_attributes = torch.empty((batch.size(0), 0))
    if 'node_attributes' in names:
        node_attributes = read_file(folder, prefix, 'node_attributes')
        if node_attributes.dim() == 1:
            node_attributes = node_attributes.unsqueeze(-1)

    node_labels = torch.empty((batch.size(0), 0))
    if 'node_labels' in names:
        node_labels = read_file(folder, prefix, 'node_labels', torch.long)
        if node_labels.dim() == 1:
            node_labels = node_labels.unsqueeze(-1)
        node_labels = node_labels - node_labels.min(dim=0)[0]
        node_labels = node_labels.unbind(dim=-1)
        node_labels = [F.one_hot(x, num_classes=-1) for x in node_labels]
        node_labels = torch.cat(node_labels, dim=-1).to(torch.float)

    edge_attributes = torch.empty((edge_index.size(1), 0))
    if 'edge_attributes' in names:
        edge_attributes = read_file(folder, prefix, 'edge_attributes')
        if edge_attributes.dim() == 1:
            edge_attributes = edge_attributes.unsqueeze(-1)

    edge_labels = torch.empty((edge_index.size(1), 0))
    if 'edge_labels' in names:
        edge_labels = read_file(folder, prefix, 'edge_labels', torch.long)
        if edge_labels.dim() == 1:
            edge_labels = edge_labels.unsqueeze(-1)
        edge_labels = edge_labels - edge_labels.min(dim=0)[0]
        edge_labels = edge_labels.unbind(dim=-1)
        edge_labels = [F.one_hot(e, num_classes=-1) for e in edge_labels]
        edge_labels = torch.cat(edge_labels, dim=-1).to(torch.float)

    x = cat([node_attributes, node_labels])
    edge_attr = cat([edge_attributes, edge_labels])

    y = None
    if 'graph_attributes' in names:  # Regression problem.
        y = read_file(folder, prefix, 'graph_attributes')
    elif 'graph_labels' in names:  # Classification problem.
        y = read_file(folder, prefix, 'graph_labels', torch.long)
        _, y = y.unique(sorted=True, return_inverse=True)

    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                     num_nodes)
    
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    data = compute_pr_and_orbit(data, batch)    
    
    data, slices = split(data, batch)

    sizes = {
        'num_node_attributes': node_attributes.size(-1),
        'num_node_labels': node_labels.size(-1),
        'num_edge_attributes': edge_attributes.size(-1),
        'num_edge_labels': edge_labels.size(-1),
    }

    return data, slices, sizes


def read_file(folder, prefix, name, dtype=None):
    path = osp.join(folder, f'{prefix}_{name}.txt')
    return read_txt_array(path, sep=',', dtype=dtype)


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item for item in seq if item.numel() > 0]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1) if len(seq) > 0 else None


def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    else:
        # Imitate `collate` functionality:
        data._num_nodes = torch.bincount(batch).tolist()
        data.num_nodes = batch.numel()
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
        
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    
    if data.pr is not None:
        if data.pr.size(0) == batch.size(0):
            slices['pr'] = node_slice
            slices['orbit_label'] = node_slice

    return data, slices
