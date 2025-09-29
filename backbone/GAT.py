import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn import GATConv, GraphConv, GINConv
import math

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = GATConv(input_dim, hidden_dim, num_heads,allow_zero_in_degree=True) #activation default=None
        self.layer2 = GATConv(hidden_dim, output_dim, 1,allow_zero_in_degree=True)    #activation default=None
        self.return_hidden = False
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, g, features=None):

        if features is None:
            if 'x' in g.ndata:
                features = g.ndata['x']
            else:
                raise ValueError("Graph does not have node features 'x' and no features were provided")
        

        if not features.is_floating_point():
            features = features.float()
            

        if features.device != g.device:
            features = features.to(g.device)
            print(f"Warning: Features were on {features.device} but graph is on {g.device}. Moving features to {g.device}.")
            
        try:

            x1 = self.layer1(g, features)
            if torch.isnan(x1).any():
                print(f"Warning: First layer output contains NaN values")
                
            x1 = F.relu(x1)
            x1 = torch.mean(x1, 1)
            

            x2 = self.layer2(g, x1).squeeze()
            
            if torch.isnan(x2).any():
                print(f"Warning: Second layer output contains NaN values")
                

            if not self.return_hidden:
                x2 = F.log_softmax(x2, dim=-1)
                

            if self.return_hidden:
                return x1
            return x2
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            print(f"Graph has {g.number_of_nodes()} nodes, features shape: {features.shape}")
            print(f"Graph device: {g.device}, features device: {features.device}")
            raise