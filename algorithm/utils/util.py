import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn import GATConv
import numpy as np
import csv
import random
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
import time

from pygsp import graphs, filters, reduction
from scipy.sparse import csr_matrix
from algorithm.utils.coarsening_utils import coarsen
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp

def check_self_loop(g):
    sources = g.edges()[0].cpu().numpy()
    targets = g.edges()[1].cpu().numpy()
    n = sum(sources == targets)
    return n>0

def dgl2pyg(dgl_g):
    sources = dgl_g.edges()[0].cpu().numpy()
    targets = dgl_g.edges()[1].cpu().numpy()
    weights = np.ones(sources.shape[0])
    n_nodes = dgl_g.ndata['x'].size()[0]
    A = csr_matrix((weights, (sources, targets)), shape=(n_nodes, n_nodes))
    pyg_g = graphs.Graph(A)
    return pyg_g

def pyg2dgl(pyg_g):
    sources = pyg_g.W.tocoo().row
    targets = pyg_g.W.tocoo().col
    
    dgl_g = dgl.DGLGraph()
    dgl_g.add_nodes(pyg_g.N)
    dgl_g.add_edges(sources, targets)
    
    
    return dgl_g

def csr2tensor(Acsr):
    Acoo = Acsr.tocoo()
    return torch.sparse.FloatTensor(torch.LongTensor([Acoo.row.tolist(), Acoo.col.tolist()]),
                              torch.FloatTensor(Acoo.data.astype(np.int32))).cuda()


def graph_coarsening(g, node_hidden_features, c2n, n2c, train_ratio=0.5, reduction=0.5, replay_nodes=None, device=None):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    num_nodes = g.number_of_nodes()
    start_time = time.time()
    
    features = g.ndata['x'].float()
    labels = g.ndata['y']
    g_pyg = dgl2pyg(g)
    C, Gc, Call, Gall = coarsen(G_orig=g_pyg, G=g_pyg, r=reduction, replay_nodes=replay_nodes, feature=node_hidden_features)
    C_tensor = csr2tensor(C)
    
    coarsened_features = C_tensor@features
    
    g_dgl = pyg2dgl(Gc).to(device)



    g_dgl.ndata['x'] = coarsened_features
    n2c_new = torch.tensor([-1 for el in range(n2c.size()[0])]).to(device)
    c2n_new = dict((el,set()) for el in range(C.shape[0]))

    mapping_counts = []
    
    for i in range(C.shape[0]):
        cluster_i = [x for x in C.getrow(i).nonzero()[1]]
        for super_node in cluster_i:
            nodes = c2n[super_node]
            c2n_new[i].update(nodes)
            n2c_new[torch.tensor(list(nodes)).to(device)] = torch.tensor([i for _ in range(len(nodes))]).to(device)
        

        mapping_counts.append(len(c2n_new[i]))

    if mapping_counts:
        avg_mapping = sum(mapping_counts) / len(mapping_counts)
        max_mapping = max(mapping_counts)
        min_mapping = min(mapping_counts)
    
        mapping_distribution = {}
        for count in mapping_counts:
            if count not in mapping_distribution:
                mapping_distribution[count] = 0
            mapping_distribution[count] += 1
        
        sorted_distribution = sorted(mapping_distribution.items(), key=lambda x: x[1], reverse=True)

    

    compression_ratio = g.number_of_nodes() / g_dgl.number_of_nodes()


    train_val_mask = torch.logical_or(g.ndata['train_mask'].to(device), g.ndata['valid_mask'].to(device))
    coarsened_mask = C_tensor@(train_val_mask.float())
    coarsened_mask = coarsened_mask>0

    if g.ndata['y'].dtype == torch.long or g.ndata['y'].dtype == torch.int:
        labels_float = g.ndata['y'].float().to(device)
    else:
        labels_float = labels
    coarsened_labels = C_tensor@(labels_float.t()*train_val_mask.float()).t()
    g_dgl.ndata['y'] = coarsened_labels

    mask_idxs = coarsened_mask.nonzero().squeeze().tolist()
    if not isinstance(mask_idxs, list):
        mask_idxs = [mask_idxs]
    random.shuffle(mask_idxs)
    num_training_sample = int(len(mask_idxs)*train_ratio)
    coarsened_train_mask = torch.zeros(len(c2n_new)).bool().to(device)
    coarsened_valid_mask = torch.zeros(len(c2n_new)).bool().to(device)
    if mask_idxs:
        coarsened_train_mask[torch.tensor(mask_idxs[:num_training_sample]).long()] = True
        coarsened_valid_mask[torch.tensor(mask_idxs[num_training_sample:]).long()] = True
    g_dgl.ndata['train_mask'] = coarsened_train_mask
    g_dgl.ndata['valid_mask'] = coarsened_valid_mask
    
    end_time = time.time()

    return g_dgl, C, c2n_new, n2c_new

def combine_graph(g, Gc=None, C=None, c2n=None, n2c=None, device=None):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    

    if len(g.ndata['y'].size()) == 1:
        g.ndata['y'] = F.one_hot(g.ndata['y']).float()

    if Gc is None:

        n_nodes = g.ndata['node_idxs'].size()[0]
        c2n = dict((i,set([g.ndata['node_idxs'][i].item()])) for i in range(n_nodes))
        

        max_node_idx = torch.max(g.ndata['node_idxs']).item()
        n2c = torch.tensor([-1 for i in range(max_node_idx+1)], device=device)
        n2c[g.ndata['node_idxs']] = torch.tensor([i for i in range(n_nodes)], device=device)
        return g, c2n, n2c


    new_node_idxs = g.ndata['new_nodes_mask'].nonzero().squeeze()
    if new_node_idxs.dim() == 0 and new_node_idxs.numel() > 0:
        new_node_idxs = new_node_idxs.unsqueeze(0)
    
    new_node_features = g.ndata['x'][new_node_idxs].float()
    new_node_labels = g.ndata['y'][new_node_idxs]
    num_new_nodes = new_node_features.size()[0]
    
    new_train_mask = g.ndata['train_mask'][new_node_idxs]
    new_valid_mask = g.ndata['valid_mask'][new_node_idxs]
    
    Gc.add_nodes(num_new_nodes, {'x':new_node_features, 'y':new_node_labels,
                               'train_mask':new_train_mask, 'valid_mask':new_valid_mask})
    
    max_new_node_idx = torch.max(g.ndata['node_idxs'][new_node_idxs]).item()
    if n2c.size()[0] < max_new_node_idx+1:
        n2c = torch.cat([n2c, torch.tensor([-1 for i in range(max_new_node_idx+1-n2c.size()[0])]).to(device)], 0)
    
    n2c[g.ndata['node_idxs'][new_node_idxs]] = torch.tensor([i+len(c2n) for i in range(new_node_idxs.shape[0])]).to(device)
    
    rows = g.ndata['node_idxs'][g.edges()[0]].to(device)
    cols = g.ndata['node_idxs'][g.edges()[1]].to(device)
    
    valid_row_idx = set((n2c[rows]>=0).nonzero().squeeze().tolist())
    valid_col_idx = set((n2c[cols]>=0).nonzero().squeeze().tolist())
    
    valid_edge_idx = torch.tensor(list(valid_col_idx.intersection(valid_row_idx))).to(device)
    
    Gc.add_edges(n2c[rows][valid_edge_idx], n2c[cols][valid_edge_idx])
    
    c2n_extend = dict((n2c[g.ndata['node_idxs'][idx]].item(),set([g.ndata['node_idxs'][idx].item()])) for i,idx in enumerate(new_node_idxs))
    c2n.update(c2n_extend)
    
    return Gc, c2n, n2c

def pyg_to_dgl(data):
    src, dst = data.edge_index[0], data.edge_index[1]
    g = dgl.graph((src, dst), num_nodes=data.num_nodes)
    

    g.ndata['x'] = data.x
    g.ndata['y'] = data.y
    
    return g

def evaluate(net, g, features, labels, test_mask):
    g.add_edges(g.nodes(), g.nodes())

    if hasattr(net, 'eval'):
        net.eval()
    y_pred = net(g, features)[test_mask].cpu()
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    y_true = labels[test_mask].cpu()
    bac = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    return bac, f1, acc