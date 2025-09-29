import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pygsp as gsp
from pygsp import graphs, filters, reduction
import algorithm.utils.graph_utils as graph_utils
import algorithm.utils.general_utils as general_utils
import time
import networkx as nx
import dgl
from scipy.sparse import diags
from torch.distributions.normal import Normal
import copy
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import torch.distributions as distributions
import math

# Binary step function for binary decisions (keep node or not)
class BinaryStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero_index = torch.abs(input) > 1
        middle_index = (torch.abs(input) <= 1) * (torch.abs(input) > 0.4)
        additional = 2-4*torch.abs(input)
        additional[zero_index] = 0.
        additional[middle_index] = 0.4
        return grad_input*additional


# NodeScorer for each expert - this evaluates node importance
class NodeScorer(nn.Module):
    def __init__(self, nlayers, in_dim, hidden, activation, k=0.5):
        super(NodeScorer, self).__init__()
        self.activation = activation
        self.nlayers = nlayers
        self.k = k  # Sparsity ratio
        
        
        # Create a simple MLP
        self.layers = nn.ModuleList()
        if nlayers == 1:
            self.layers.append(nn.Linear(in_dim, 1))
        else:
            self.layers.append(nn.Linear(in_dim, hidden))
            for i in range(nlayers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
            self.layers.append(nn.Linear(hidden, 1))
            
        self.reset_parameters()
        
    def reset_parameters(self):
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            
    def forward(self, node_features, node_topo_features=None, temp=0.5, training=True):
        """Forward pass through the MLP to get node importance scores"""
        device = node_features.device
        
        if node_topo_features is not None:
            node_topo_features = node_topo_features.to(device)
            
            if node_topo_features.shape[0] != node_features.shape[0]:

                if node_topo_features.shape[0] < node_features.shape[0]:
                    repeat_times = (node_features.shape[0] + node_topo_features.shape[0] - 1) // node_topo_features.shape[0]
                    node_topo_features = node_topo_features.repeat(repeat_times, 1)
                    node_topo_features = node_topo_features[:node_features.shape[0]]
                else:
                    node_topo_features = node_topo_features[:node_features.shape[0]]
            
            combined_features = torch.cat([node_features, node_topo_features], dim=1)
        else:
            combined_features = node_features
        
        combined_features = combined_features.to(device)
        
        first_layer_weight_shape = self.layers[0].weight.shape

        if combined_features.shape[1] != first_layer_weight_shape[1]:

            hidden_dim = first_layer_weight_shape[0] 
            new_first_layer = nn.Linear(combined_features.shape[1], hidden_dim).to(device)
            
            nn.init.xavier_normal_(new_first_layer.weight)
            nn.init.zeros_(new_first_layer.bias)
            
            self.layers[0] = new_first_layer
        
        # Forward pass through MLP
        out = combined_features
        for i, layer in enumerate(self.layers):
            layer = layer.to(device)
            out = layer(out)
            if i < len(self.layers) - 1:
                out = self.activation(out)
        
        # Return node importance scores
        return out.view(-1)


# NodeMoE - Mixture of Experts for node importance evaluation
class NodeMoE(nn.Module):
    def __init__(self, input_size, hidden_size, num_experts, nlayers, activation, k_list, 
                 expert_select=2, noisy_gating=True, coef=1e-2, 
                 use_node_positional=True, use_node_mmd=True, 
                 use_node_mahalanobis=True, similarity_threshold=0.7):
        super(NodeMoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.k = expert_select  # how many experts to use for each node
        self.loss_coef = coef
        self.k_list = k_list
        
        self.use_node_positional = use_node_positional
        self.use_node_mmd = use_node_mmd
        self.use_node_mahalanobis = use_node_mahalanobis
        self.similarity_threshold = similarity_threshold
        
        base_expert_input_dim = input_size + 8 + 4
        additional_dim = 0
        if use_node_positional:
            additional_dim += 8
        if use_node_mmd:
            additional_dim += 1
        if use_node_mahalanobis:
            additional_dim += 1
        additional_dim += 2
        
        expert_input_dim = base_expert_input_dim + additional_dim
        
        self.experts = nn.ModuleList([
            NodeScorer(nlayers=nlayers, in_dim=expert_input_dim, hidden=hidden_size, activation=activation, k=k) 
            for k in k_list
        ])
        
        # Gating network parameters
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        
        self.topo_features = None
        self.subgraph_features = None
        self.degree_embeddings = None
        self.positional_features = None
        self.similarity_features = None
        
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates."""
        return (gates > 0).sum(0)
        
    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating - computes probability that a value is in top k."""
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob
    
    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating to select experts."""
        clean_logits = x @ self.w_gate  # size:(nums_node,nums_expert)
        
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # Calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k+1, self.num_experts), dim=1) 
        top_k_logits = top_logits[:, :self.k]  # size:(batch_size,self.k)
        top_k_indices = top_indices[:, :self.k]  # size:(batch_size,self.k)
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)  # size:(batch_size,num_experts)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
            
        return gates, load
    
    def get_node_topo_features(self, g):
        """Calculate topological features for nodes to be used as additional features."""
        device = next(self.parameters()).device
        
        if g.device != torch.device('cpu'):
            g_cpu = g.cpu()
        else:
            g_cpu = g
            
        nx_g = dgl.to_networkx(g_cpu)
        
        if nx.is_directed(nx_g):
            nx_g = nx_g.to_undirected()
        
        if isinstance(nx_g, nx.MultiGraph) or isinstance(nx_g, nx.MultiDiGraph):
            nx_g = nx.Graph(nx_g)
        
        num_nodes = g.number_of_nodes()
        
        topo_features = torch.zeros((num_nodes, 8), device=device)
        
        try:
            degrees = torch.tensor([d for _, d in nx_g.degree()], device=device)
            max_degree = degrees.max().item() if degrees.numel() > 0 else 1
            degrees_normalized = degrees / max_degree
            
            try:
                betweenness = nx.betweenness_centrality(nx_g)
                betweenness_values = torch.tensor(list(betweenness.values()), device=device)
            except Exception as e:
                betweenness_values = torch.zeros(num_nodes, device=device)
            
            try:
                clustering = nx.clustering(nx_g)
                clustering_values = torch.tensor(list(clustering.values()), device=device)
            except Exception as e:
                clustering_values = torch.zeros(num_nodes, device=device)
            
            try:
                eigenvector_centrality = nx.eigenvector_centrality_numpy(nx_g)
                eigenvector_values = torch.tensor(list(eigenvector_centrality.values()), device=device)
            except Exception as e:
                eigenvector_values = torch.zeros(num_nodes, device=device)
            
            try:
                closeness_centrality = nx.closeness_centrality(nx_g)
                closeness_values = torch.tensor(list(closeness_centrality.values()), device=device)
            except Exception as e:
                closeness_values = torch.zeros(num_nodes, device=device)
            
            try:
                pagerank = nx.pagerank(nx_g)
                pagerank_values = torch.tensor(list(pagerank.values()), device=device)
            except Exception as e:
                pagerank_values = torch.zeros(num_nodes, device=device)
            
            topo_features[:, 0] = degrees_normalized
            topo_features[:, 1] = betweenness_values
            topo_features[:, 2] = clustering_values
            topo_features[:, 3] = eigenvector_values
            topo_features[:, 4] = closeness_values
            topo_features[:, 5] = pagerank_values
            
            neighborhood_degrees = []
            community_index = []
            
            for node in range(num_nodes):
                try:
                    neighbors = list(nx_g.neighbors(node))
                    if neighbors:
                        neighbor_degrees = [nx_g.degree(n) for n in neighbors]
                        if len(neighbor_degrees) > 1:
                            mean_degree = np.mean(neighbor_degrees)
                            if mean_degree > 0:
                                std_degree = np.std(neighbor_degrees)
                                degree_heterogeneity = std_degree / mean_degree
                            else:
                                degree_heterogeneity = 0.0
                        else:
                            degree_heterogeneity = 0.0
                        
                        if len(neighbors) > 1:
                            possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
                            actual_edges = 0
                            for i in range(len(neighbors)):
                                for j in range(i+1, len(neighbors)):
                                    if nx_g.has_edge(neighbors[i], neighbors[j]):
                                        actual_edges += 1
                            community_idx = actual_edges / possible_edges if possible_edges > 0 else 0
                        else:
                            community_idx = 0.0
                    else:
                        degree_heterogeneity = 0.0
                        community_idx = 0.0
                except Exception:
                    degree_heterogeneity = 0.0
                    community_idx = 0.0
                
                neighborhood_degrees.append(degree_heterogeneity)
                community_index.append(community_idx)
            
            topo_features[:, 6] = torch.tensor(neighborhood_degrees, device=device)
            topo_features[:, 7] = torch.tensor(community_index, device=device)
            
        except Exception as e:
            topo_features = torch.zeros((num_nodes, 8), device=device)
            
        topo_features = F.normalize(topo_features, p=2, dim=0)
        topo_features = topo_features.to(device)
        self.topo_features = topo_features
        
        return topo_features
    
    def calculate_subgraph_features(self, g, subgraphs):
        """Calculate features from subgraphs for each node."""
        device = next(self.parameters()).device
        num_nodes = g.number_of_nodes()
        node_sg_features = torch.zeros((num_nodes, 4), device=device)
        
        node_to_subgraphs = [[] for _ in range(num_nodes)]
        
        for hop_idx, hop_subgraphs in enumerate(subgraphs):
            for sg_idx, subgraph in enumerate(hop_subgraphs):
                subgraph_cpu = subgraph.cpu()
                for node in range(subgraph_cpu.number_of_nodes()):
                    orig_node_id = subgraph_cpu.ndata[dgl.NID][node].item()
                    if orig_node_id < num_nodes:
                        node_to_subgraphs[orig_node_id].append((hop_idx, sg_idx))
        
        subgraph_metrics = {}
        for hop_idx, hop_subgraphs in enumerate(subgraphs):
            for sg_idx, subgraph in enumerate(hop_subgraphs):
                try:
                    subgraph_cpu = subgraph.cpu()
                    nx_sg = dgl.to_networkx(subgraph_cpu).to_undirected()
                    
                    if isinstance(nx_sg, nx.MultiGraph) or isinstance(nx_sg, nx.MultiDiGraph):
                        nx_sg = nx.Graph(nx_sg)
                    
                    avg_degree = sum(dict(nx_sg.degree()).values()) / max(1, len(nx_sg))
                    
                    try:
                        clustering_coef = nx.average_clustering(nx_sg)
                    except Exception as e:
                        clustering_coef = 0.0
                    
                    try:
                        if nx.is_connected(nx_sg) and len(nx_sg) > 1:
                            diameter = nx.diameter(nx_sg)
                        else:
                            largest_cc = max(nx.connected_components(nx_sg), key=len)
                            sg_largest_cc = nx_sg.subgraph(largest_cc)
                            if len(sg_largest_cc) > 1:
                                diameter = nx.diameter(sg_largest_cc)
                            else:
                                diameter = 0
                    except Exception as e:
                        diameter = 0
                    
                    if len(nx_sg) > 1:
                        density = nx.density(nx_sg)
                    else:
                        density = 0.0
                    
                    subgraph_metrics[(hop_idx, sg_idx)] = [avg_degree, clustering_coef, diameter, density]
                except Exception as e:
                    subgraph_metrics[(hop_idx, sg_idx)] = [0.0, 0.0, 0.0, 0.0]
        
        for node in range(num_nodes):
            if not node_to_subgraphs[node]:
                continue
                
            node_subgraph_features = []
            for hop_idx, sg_idx in node_to_subgraphs[node]:
                if (hop_idx, sg_idx) in subgraph_metrics:
                    node_subgraph_features.append(subgraph_metrics[(hop_idx, sg_idx)])
            
            if node_subgraph_features:
                node_sg_features[node] = torch.tensor(np.mean(node_subgraph_features, axis=0), device=device)
        
        node_sg_features = F.normalize(node_sg_features, p=2, dim=0)
        node_sg_features = node_sg_features.to(device)
        self.subgraph_features = node_sg_features
        
        return node_sg_features
    
    def get_degree_embeddings(self, g):
        device = next(self.parameters()).device
        
        if g.device != torch.device('cpu'):
            g_cpu = g.cpu()
        else:
            g_cpu = g
            
        nx_g = dgl.to_networkx(g_cpu)
        
        if nx.is_directed(nx_g):
            nx_g = nx_g.to_undirected()
        
        if isinstance(nx_g, nx.MultiGraph) or isinstance(nx_g, nx.MultiDiGraph):
            nx_g = nx.Graph(nx_g)
        
        num_nodes = g.number_of_nodes()
        
        degree_embeddings = torch.zeros((num_nodes, 8), device=device)
        
        try:
            degrees = torch.tensor([d for _, d in nx_g.degree()], device=device)
            max_degree = degrees.max().item() if degrees.numel() > 0 else 1
            
            for i in range(num_nodes):
                degree = degrees[i].item()
                for j in range(8):
                    div_term = torch.exp(torch.arange(0, 8, 2, device=device) * -(math.log(10000.0) / 8))
                    if j % 2 == 0:
                        degree_embeddings[i, j] = torch.sin(degree * div_term[j//2])
                    else:
                        degree_embeddings[i, j] = torch.cos(degree * div_term[j//2])
            
            degree_embeddings = F.normalize(degree_embeddings, p=2, dim=0)
            
        except Exception as e:
            degree_embeddings = torch.zeros((num_nodes, 8), device=device)
        
        degree_embeddings = degree_embeddings.to(device)
        self.degree_embeddings = degree_embeddings
        
        return degree_embeddings
    
    def get_similarity_features(self, features):

        device = next(self.parameters()).device
        num_nodes = features.shape[0]
        
        similarity_features = torch.zeros((num_nodes, 4), device=device)
        
        if num_nodes > 500:
            sample_size = 500
            sample_indices = torch.randperm(num_nodes)[:sample_size]
            sample_features = features[sample_indices]
        else:
            sample_features = features
            
        try:
            mean_vec = torch.mean(features, dim=0, keepdim=True)
            
            if self.use_node_mmd:
                try:
                    def compute_kernel(x, y):
                        x_size = x.size(0)
                        y_size = y.size(0)
                        dim = x.size(1)
                        x = x.unsqueeze(1)  # (x_size, 1, dim)
                        y = y.unsqueeze(0)  # (1, y_size, dim)
                        tiled_x = x.expand(x_size, y_size, dim)
                        tiled_y = y.expand(x_size, y_size, dim)
                        kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / dim
                        return torch.exp(-kernel_input)
                    
                    x_kernel = compute_kernel(sample_features, sample_features)
                    mean_mmd = x_kernel.mean()
                    
                    for i in range(num_nodes):
                        node_feature = features[i:i+1]
                        node_kernel = compute_kernel(node_feature, sample_features)
                        mmd = torch.abs(node_kernel.mean() - mean_mmd)
                        similarity_features[i, 0] = mmd.item()
                        
                except Exception as e:
                    similarity_features[:, 0] = 0.0
            
            if self.use_node_mahalanobis:
                try:
                    centered_features = features - mean_vec
                    cov_matrix = torch.mm(centered_features.t(), centered_features) / (num_nodes - 1)
                    
                    cov_matrix += torch.eye(cov_matrix.shape[0], device=device) * 1e-5
                    
                    try:
                        inv_cov = torch.linalg.inv(cov_matrix)
                    except:
                        inv_cov = torch.linalg.pinv(cov_matrix)
                    
                    for i in range(num_nodes):
                        dev = features[i] - mean_vec.squeeze()
                        mahal_dist = torch.sqrt(torch.mm(torch.mm(dev.unsqueeze(0), inv_cov), dev.unsqueeze(1)))
                        similarity_features[i, 1] = mahal_dist.item()
                        
                except Exception as e:
                    similarity_features[:, 1] = 0.0
            
            features_normalized = F.normalize(features, p=2, dim=1)
            
            cosine_sim_matrix = torch.mm(features_normalized, features_normalized.t())
            cosine_sim_matrix.fill_diagonal_(0)
            
            avg_cosine_sim = cosine_sim_matrix.mean(dim=1)
            similarity_features[:, 3] = avg_cosine_sim
            
            try:
                features_np = features.cpu().numpy()
                
                if 'degree' in locals():
                    important_nodes = math.degrees.argsort(descending=True)[:min(50, num_nodes)].cpu().numpy()
                else:
                    important_nodes = np.random.choice(num_nodes, size=min(50, num_nodes), replace=False)
                
                important_feats = features_np[important_nodes]
                mean_important = np.mean(important_feats, axis=0)
                
                for i in range(num_nodes):
                    try:
                        r_values = []
                        for j in important_nodes:
                            if i != j: 
                                r, _ = pearsonr(features_np[i], features_np[j])
                                if not np.isnan(r):
                                    r_values.append(abs(r)) 
                        
                        if r_values:
                            pearson_coef = np.mean(r_values)
                        else:
                            pearson_coef = 0.0
                    except:
                        try:
                            r, _ = pearsonr(features_np[i], mean_important)
                            pearson_coef = abs(r) if not np.isnan(r) else 0.0
                        except:
                            pearson_coef = 0.0
                    
                    similarity_features[i, 2] = pearson_coef
                    
            except Exception as e:
                similarity_features[:, 2] = 0.0
                
            similarity_features = F.normalize(similarity_features, p=2, dim=0)
            
        except Exception as e:  
            similarity_features = torch.zeros((num_nodes, 4), device=device)
        
        similarity_features = similarity_features.to(device)
        self.similarity_features = similarity_features
        
        return similarity_features
    
    def forward(self, features, g, temp=0.5, training=True):
        """Forward pass to get node importance scores with MoE."""
        device = features.device
        num_nodes = features.shape[0]
        
        if g.device != device:
            g = g.to(device)
        
        try:
            if self.topo_features is None or self.topo_features.shape[0] != num_nodes:
                self.topo_features = self.get_node_topo_features(g)
        except Exception as e:
            self.topo_features = torch.zeros((num_nodes, 8), device=device)
        
        try:
            if self.subgraph_features is None or self.subgraph_features.shape[0] != num_nodes:
                subgraphs = decompose_to_subgraphs(g, hop_sizes=[1, 2])
                self.subgraph_features = self.calculate_subgraph_features(g, subgraphs)
        except Exception as e:
            self.subgraph_features = torch.zeros((num_nodes, 4), device=device)
        
        if self.use_node_positional and (self.degree_embeddings is None or self.degree_embeddings.shape[0] != num_nodes):
            try:
                self.degree_embeddings = self.get_degree_embeddings(g)
            except Exception as e:
                self.degree_embeddings = torch.zeros((num_nodes, 8), device=device)
        
        if (self.use_node_mmd or self.use_node_mahalanobis or True) and \
           (self.similarity_features is None or self.similarity_features.shape[0] != num_nodes):
            try:
                self.similarity_features = self.get_similarity_features(features)
            except Exception as e:
                self.similarity_features = torch.zeros((num_nodes, 4), device=device)
        
        self.topo_features = self.topo_features.to(device)
        self.subgraph_features = self.subgraph_features.to(device)
        
        if self.use_node_positional and self.degree_embeddings is not None:
            self.degree_embeddings = self.degree_embeddings.to(device)
        
        if self.similarity_features is not None:
            self.similarity_features = self.similarity_features.to(device)
        
        if self.topo_features.shape[0] != num_nodes:
            self.topo_features = torch.zeros((num_nodes, 8), device=device)
        
        if self.subgraph_features.shape[0] != num_nodes:
            self.subgraph_features = torch.zeros((num_nodes, 4), device=device)
        
        if self.use_node_positional and (self.degree_embeddings is None or self.degree_embeddings.shape[0] != num_nodes):
            self.degree_embeddings = torch.zeros((num_nodes, 8), device=device)
        
        if self.similarity_features is None or self.similarity_features.shape[0] != num_nodes:
            self.similarity_features = torch.zeros((num_nodes, 4), device=device)
        
        node_gates, load = self.noisy_top_k_gating(features, training)
        
        importance = node_gates.sum(0)  # size:(num_experts)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= self.loss_coef
        
        expert_outputs = []
        for i in range(self.num_experts):
            try:
                combined_features_list = [self.topo_features, self.subgraph_features]
                
                if self.use_node_positional and self.degree_embeddings is not None:
                    combined_features_list.append(self.degree_embeddings)
                
                if self.similarity_features is not None:
                    combined_features_list.append(self.similarity_features)
                
                combined_features = torch.cat(combined_features_list, dim=1)
                
                expert_i_output = self.experts[i](features, combined_features, temp, training)
                expert_outputs.append(expert_i_output)
            except Exception as e:
                expert_outputs.append(torch.rand(num_nodes, device=device))
        
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [num_nodes, num_experts]

        gated_output = node_gates * expert_outputs  # [num_nodes, num_experts]
        node_scores = gated_output.mean(dim=1)  # [num_nodes]
        

        try:
            cosine_sim = torch.mm(F.normalize(features, p=2, dim=1), F.normalize(features, p=2, dim=1).t())
            
            cosine_sim.fill_diagonal_(0)
            
            avg_sim = cosine_sim.mean(dim=1)
            
            importance_from_sim = 1.0 - avg_sim
            

            final_scores = node_scores * 0.6  
            final_scores += importance_from_sim * 0.1 
            
            if self.similarity_features is not None:

                if self.use_node_mmd:
                    final_scores += self.similarity_features[:, 0] * 0.1
                
                if self.use_node_mahalanobis:
                    final_scores += self.similarity_features[:, 1] * 0.1
                
                final_scores += (1.0 - self.similarity_features[:, 2]) * 0.1
                # final_scores += (1.0 - self.similarity_features[:, 3]) * 0.05
        except Exception as e:
            final_scores = node_scores
        
        return final_scores, loss


def decompose_to_subgraphs(g, hop_sizes=[1, 2]):
    """
    Decompose the input graph into subgraphs based on hop sizes.
    Returns a list of subgraphs for each hop size.
    
    Args:
        g: Input graph in DGL format
        hop_sizes: List of hop sizes to extract
    
    Returns:
        List of subgraphs for each hop size
    """
    device = g.device
    
    g_cpu = g.cpu()
    nx_g = dgl.to_networkx(g_cpu)
    
    if nx.is_directed(nx_g):
        nx_g = nx_g.to_undirected()
    
    if isinstance(nx_g, nx.MultiGraph) or isinstance(nx_g, nx.MultiDiGraph):
        nx_g = nx.Graph(nx_g)
    
    subgraphs = []
    max_nodes = g.number_of_nodes()
    
    for hop in hop_sizes:
        hop_subgraphs = []
        if max_nodes > 200 and hop > 1:
            sample_size = min(100, max_nodes)
            sample_nodes = np.random.choice(max_nodes, size=sample_size, replace=False)
            
            for node in sample_nodes:
                k_hop_nodes = set()
                current_layer = {node}
                
                for _ in range(hop):
                    next_layer = set()
                    for n in current_layer:
                        try:
                            next_layer.update(nx_g.neighbors(n))
                        except Exception:
                            continue
                    k_hop_nodes.update(current_layer)
                    current_layer = next_layer - k_hop_nodes
                
                k_hop_nodes.update(current_layer)
                
                if len(k_hop_nodes) > 0:
                    k_hop_nodes = list(k_hop_nodes)
                    try:
                        subgraph = dgl.node_subgraph(g_cpu, k_hop_nodes)
                        if device.type != 'cpu':
                            subgraph = subgraph.to(device)
                        hop_subgraphs.append(subgraph)
                    except Exception as e:
                        continue
        else:
            for node in range(g.number_of_nodes()):
                nodes_within_khop = set([node])
                current_neighbors = set([node])
                
                for _ in range(hop):
                    next_neighbors = set()
                    for n in current_neighbors:
                        try:
                            next_neighbors.update(nx_g.neighbors(n))
                        except Exception:
                            continue
                    current_neighbors = next_neighbors - nodes_within_khop
                    nodes_within_khop.update(current_neighbors)
                
                if len(nodes_within_khop) > 0:
                    try:
                        subgraph = dgl.node_subgraph(g_cpu, list(nodes_within_khop))
                        if device.type != 'cpu':
                            subgraph = subgraph.to(device)
                        hop_subgraphs.append(subgraph)
                    except Exception as e:
                        continue

        if len(hop_subgraphs) == 0:
            if device.type != 'cpu':
                hop_subgraphs.append(g_cpu.to(device))
            else:
                hop_subgraphs.append(g_cpu)
        
        subgraphs.append(hop_subgraphs)
    
    return subgraphs

def node_coarsening_with_moe(g, node_features, node_reduction=0.5, replay_nodes=None, device=None, 
                      k_list=None, c2n=None, n2c=None, 
                      use_node_positional=True, use_node_mmd=True, 
                      use_node_mahalanobis=True, similarity_threshold=0.7):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        g = g.to(device)
        
        if k_list is None:
            k_list = torch.tensor([0.2, 0.4, 0.6, 0.8], device=device)
        
        k_list = k_list.to(device)

        num_nodes = g.number_of_nodes()
        
        if node_features is None:
            if 'x' in g.ndata:
                features = g.ndata['x'].float().to(device)
            else:
                features = torch.rand((num_nodes, 64), device=device)
        elif not isinstance(node_features, torch.Tensor):
            features = torch.tensor(node_features, dtype=torch.float32, device=device)
        elif node_features.dtype != torch.float32 and node_features.dtype != torch.float64:
            features = node_features.float().to(device)
        else:
            features = node_features.to(device)
        
        if features.shape[0] != num_nodes:
            if features.shape[0] > num_nodes:
                features = features[:num_nodes]
            else:
                repeat_times = (num_nodes + features.shape[0] - 1) // features.shape[0]
                features_repeated = features.repeat(repeat_times, 1)
                features = features_repeated[:num_nodes]
        
        if 'x' in g.ndata:
            original_features = g.ndata['x'].to(device)
            original_feature_dim = original_features.shape[1]
        else:
            original_features = features
            original_feature_dim = features.shape[1]
        
        if c2n is None or not isinstance(c2n, dict) or len(c2n) == 0:
            c2n = {}
        
        if n2c is None or not isinstance(n2c, torch.Tensor) or n2c.numel() == 0:
            n2c = torch.zeros(num_nodes, dtype=torch.long, device=device)
        elif n2c.numel() != num_nodes:
            n2c = torch.zeros(num_nodes, dtype=torch.long, device=device)
        else:
            n2c = n2c.to(device)
        
        try:
            input_size = features.shape[1]
            
            moe = NodeMoE(
                input_size=input_size,
                hidden_size=64,
                num_experts=len(k_list),
                nlayers=2,
                activation=nn.ReLU(),
                k_list=k_list,
                expert_select=min(2, len(k_list)),
                noisy_gating=True,
                coef=1e-2,
                use_node_positional=use_node_positional,
                use_node_mmd=use_node_mmd,
                use_node_mahalanobis=use_node_mahalanobis,
                similarity_threshold=similarity_threshold
            ).to(device)
            

    
            
           
            try:
                node_scores, _ = moe(features, g, temp=0.5, training=False)
            except Exception as e:
                node_scores = torch.rand(num_nodes, device=device)
            
           
            try:
                features_norm = F.normalize(features, p=2, dim=1)
                similarity_matrix = torch.mm(features_norm, features_norm.t())
            except Exception as e:
                
                similarity_matrix = torch.eye(num_nodes, device=device)
            

            num_nodes_to_keep = int(num_nodes * (1.0 - node_reduction))
            num_nodes_to_keep = max(num_nodes_to_keep, 1)  
            
            
            
            
            if replay_nodes is not None and len(replay_nodes) > 0:
                try:
                    replay_nodes = replay_nodes.to(device)
                    replay_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
                    
                    
                    valid_replay_nodes = replay_nodes[replay_nodes < num_nodes]
                    
                    if len(valid_replay_nodes) > 0:
                        replay_mask[valid_replay_nodes] = True
                        
                        node_scores[valid_replay_nodes] = float('inf')
                except Exception as e:
                    replay_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            else:
                replay_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            
            _, indices = torch.sort(node_scores, descending=True)
            
            kept_nodes = indices[:num_nodes_to_keep]
            removed_nodes = indices[num_nodes_to_keep:]
            
            node_mapping = torch.arange(num_nodes, device=device)
            
            for node in removed_nodes:
                similarities = similarity_matrix[node, kept_nodes]
                
                most_similar = kept_nodes[similarities.argmax()]

                node_mapping[node] = most_similar
            

            

            edge_index = torch.stack(g.edges(), dim=0)
            

            src_nodes = node_mapping[edge_index[0]]
            dst_nodes = node_mapping[edge_index[1]]
            

            new_edges = []
            for i in range(len(src_nodes)):
                if src_nodes[i] != dst_nodes[i]:  
                    new_edges.append((src_nodes[i].item(), dst_nodes[i].item()))
            

            if not new_edges:
                new_edges.append((kept_nodes[0].item(), kept_nodes[0].item()))
                

            unique_kept_nodes = torch.unique(kept_nodes).sort()[0]
            node_id_map = {old_id.item(): i for i, old_id in enumerate(unique_kept_nodes)}
            

            remapped_edges = []
            for src, dst in new_edges:
                if src in node_id_map and dst in node_id_map:
                    remapped_edges.append((node_id_map[src], node_id_map[dst]))
            
            if not remapped_edges:
                remapped_edges.append((0, 0))  
                

            coarsened_g = dgl.graph(remapped_edges, num_nodes=len(node_id_map)).to(device)
            

            coarsened_hidden_features = torch.zeros((len(node_id_map), features.shape[1]), device=device)
            coarsened_original_features = torch.zeros((len(node_id_map), original_feature_dim), device=device)
            coarsened_y = None
            if 'y' in g.ndata:
                try:
                    if len(g.ndata['y'].shape) > 1:  
                        coarsened_y = torch.zeros((len(node_id_map), g.ndata['y'].shape[1]), device=device)
                    else:
                        coarsened_y = torch.zeros(len(node_id_map), device=device, dtype=g.ndata['y'].dtype)
                except Exception as e:

                    coarsened_y = None
            

            new_c2n = {}
            new_n2c = torch.zeros(num_nodes, dtype=torch.long, device=device)
            
            for old_id, new_id in node_id_map.items():

                mapped_nodes = (node_mapping == old_id).nonzero().view(-1)
                

                new_c2n[new_id] = mapped_nodes.cpu().tolist()
                

                new_n2c[mapped_nodes] = new_id
                

                try:
                    coarsened_hidden_features[new_id] = features[mapped_nodes].mean(dim=0)
                except Exception as e:

                    
                    coarsened_hidden_features[new_id] = torch.rand(features.shape[1], device=device)
                

                try:
                    coarsened_original_features[new_id] = original_features[mapped_nodes].mean(dim=0)
                except Exception as e:
                    
                    
                    coarsened_original_features[new_id] = torch.rand(original_feature_dim, device=device)
                
                
                if coarsened_y is not None:
                    try:
                        if len(g.ndata['y'].shape) > 1:  
                            coarsened_y[new_id] = g.ndata['y'][mapped_nodes].float().mean(dim=0)
                        else:
                            labels = g.ndata['y'][mapped_nodes]
                            unique_labels, counts = torch.unique(labels, return_counts=True)
                            coarsened_y[new_id] = unique_labels[counts.argmax()]
                    except Exception as e:
                        
                        
                        if len(g.ndata['y'].shape) > 1:
                            coarsened_y[new_id] = torch.zeros(g.ndata['y'].shape[1], device=device)
                        else:
                            coarsened_y[new_id] = torch.zeros(1, device=device, dtype=g.ndata['y'].dtype)
            
            
            coarsened_g.ndata['x'] = coarsened_original_features
            
            coarsened_g.ndata['hidden'] = coarsened_hidden_features
            if coarsened_y is not None:
                coarsened_g.ndata['y'] = coarsened_y
            
            
            for key in g.ndata.keys():
                if key not in ['x', 'y', 'hidden'] and key not in coarsened_g.ndata:
                    if torch.is_tensor(g.ndata[key]):
                        if g.ndata[key].dtype == torch.bool:
                            coarsened_g.ndata[key] = torch.zeros(len(node_id_map), dtype=torch.bool, device=device)
                        else:
                            shape = [len(node_id_map)]
                            if len(g.ndata[key].shape) > 1:
                                shape.extend(g.ndata[key].shape[1:])
                            coarsened_g.ndata[key] = torch.zeros(shape, dtype=g.ndata[key].dtype, device=device)
                            
                        
                        for old_id, new_id in node_id_map.items():
                            mapped_nodes = (node_mapping == old_id).nonzero().view(-1)
                            
                            if g.ndata[key].dtype == torch.bool:
                                
                                coarsened_g.ndata[key][new_id] = g.ndata[key][mapped_nodes].any()
                            else:
                                
                                try:
                                    coarsened_g.ndata[key][new_id] = g.ndata[key][mapped_nodes].float().mean(dim=0)
                                except:

                                    if len(mapped_nodes) > 0:
                                        coarsened_g.ndata[key][new_id] = g.ndata[key][mapped_nodes[0]]


            

            num_coarsened_nodes = len(node_id_map)

            C = sp.eye(num_coarsened_nodes, format='csr')
            
            if isinstance(c2n, dict) and len(c2n) > 0:

                final_c2n = {}
                for new_coarse_idx, orig_nodes in new_c2n.items():
                    original_nodes = []
                    for node in orig_nodes:

                        if node < n2c.numel():
                            old_coarse_idx = n2c[node].item()
                            if old_coarse_idx in c2n:

                                original_nodes.extend(c2n[old_coarse_idx])
                            else:

                                original_nodes.append(node)
                        else:

                            original_nodes.append(node)
                    

                    final_c2n[new_coarse_idx] = list(set(original_nodes))
                

                c2n = final_c2n
            else:

                c2n = new_c2n
            

            
            return coarsened_g, C, c2n, new_n2c
        except Exception as e:

            import traceback
            traceback.print_exc()  


            default_c2n = {i: [i] for i in range(num_nodes)}
            default_n2c = torch.arange(num_nodes, device=device)
            default_C = sp.eye(num_nodes, format='csr')
            return g, default_C, default_c2n, default_n2c
    except Exception as e:
        import traceback
        traceback.print_exc()  
        
        try:
            num_nodes = g.number_of_nodes()
        except:
            num_nodes = 100
            
        default_c2n = {i: [i] for i in range(num_nodes)}
        default_n2c = torch.arange(num_nodes, device=device)
        default_C = sp.eye(num_nodes, format='csr')
        return g, default_C, default_c2n, default_n2c 