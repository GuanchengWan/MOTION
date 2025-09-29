import argparse
from dgl.data import register_data_args
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
# import pickle
# import random
import numpy as np
# from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
# import csv
import quadprog
import torch.optim as optim
import copy
import random
import collections

class DYGRA_reservior(torch.nn.Module):
    def __init__(self,
                 model, opt, num_class, buffer_size,
                 args):
        super(DYGRA_reservior, self).__init__()

        # self.task_manager = task_manager
        # setup network
        self.net = model
        # setup optimizer
        self.opt = opt
        # setup memories
        self.current_task = 0
        self.num_samples = 0
        self.buffer_size = buffer_size # memory size
        self.er_buffer = []

    def update_er_buffer(self, g):
        train_nodes = g.ndata['train_mask'].nonzero()
        

        if train_nodes.dim() == 0 and train_nodes.numel() > 0:
            train_nodes = train_nodes.unsqueeze(0)
            

        if train_nodes.numel() == 0:
            return self.er_buffer
            
        
        for node in train_nodes:
            if isinstance(node, torch.Tensor):
                node = node.item()
            node_idx = g.ndata['node_idxs'][node].item()
            self.num_samples += 1
            if len(self.er_buffer) < self.buffer_size:
                self.er_buffer.append(node_idx)
            else:
                rand_idx = random.randint(0, self.num_samples-1)
                if rand_idx < self.buffer_size:
                    old_node = self.er_buffer[rand_idx]
                    self.er_buffer[rand_idx] = node_idx
        return self.er_buffer

    def forward(self, g, features):
        # print (self.net.return_hidden)
        output = self.net(g, features)
        return output

    def observe(self, g_list, t, loss_func):
        g = copy.deepcopy(g_list[t])
        g.add_edges(g.nodes(), g.nodes())
        features = g.ndata['x']
        labels = torch.max(g.ndata['y'],1).indices
        train_mask = g.ndata['train_mask']
        valid_mask = g.ndata['valid_mask']
        self.net.train()
        self.net.zero_grad()

        output = self.net(g, features)
        output = F.log_softmax(output, 1)
        loss = loss_func((output[train_mask]), labels[train_mask])
        loss.backward()
        self.opt.step()
