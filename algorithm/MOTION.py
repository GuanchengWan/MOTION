from algorithm.Base import BaseClient,BaseServer
import numpy as np
import pygsp as gsp
from pygsp import graphs, filters, reduction
import scipy as sp
from scipy import sparse
import torch
import matplotlib
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from sortedcontainers import SortedList
from algorithm.utils.coarsening_utils import coarsen
import time
from algorithm.utils.util import *
import pickle
import torch.nn.functional as F
import copy
# from torch.nn import gcl
from algorithm.utils.gcl_methods import *
import dgl
from torch_geometric.data import Data
from torch_geometric.utils import coalesce
from algorithm.utils.node_moe_utils import node_coarsening_with_moe


class MOTIONServer(BaseServer):
    def __init__(self, args, clients,model, data,logger):
        super().__init__(args, clients,model, data,logger)
        self.acc_matrix = torch.zeros((self.num_tasks, self.num_tasks)).to(self.device)

        self.pcb_ratio = args.pcb_ratio if hasattr(args, 'pcb_ratio') else 0.1
        self.pcb_min_ratio = args.pcb_min_ratio if hasattr(args, 'pcb_min_ratio') else 0.0001
        self.pcb_max_ratio = args.pcb_max_ratio if hasattr(args, 'pcb_max_ratio') else 0.0001
    
    def normalize(self, x, dim=0):
        min_values, _ = torch.min(x, dim=dim, keepdim=True)
        max_values, _ = torch.max(x, dim=dim, keepdim=True)
        denominator = torch.clamp(max_values - min_values, min=1e-12)
        y = (x - min_values) / denominator
        return y
    
    def clamp(self, x, min_ratio=0, max_ratio=0):
        if len(x.size()) == 1:
            d = x.size(0)
            sorted_x, _ = torch.sort(x)
            min_val = sorted_x[int(d * min_ratio)]
            max_val = sorted_x[int(d * (1-max_ratio)-1)]
        else:
            d = x.size(1)
            sorted_x, _ = torch.sort(x, dim=1)
            min_val = sorted_x[:, int(d * min_ratio)].unsqueeze(1)
            max_val = sorted_x[:, int(d * (1-max_ratio)-1)].unsqueeze(1)
        clamped_x = torch.clamp(x, min_val, max_val)
        return clamped_x
    
    def act(self, x):
        return torch.tanh(x)  
    
    def pcb_merge(self, task_vectors, pcb_ratio=None):

        if pcb_ratio is None:
            pcb_ratio = self.pcb_ratio
            
        all_checks = task_vectors.clone()
        n, d = all_checks.shape   
        
        all_checks_abs = self.clamp(torch.abs(all_checks), 
                                    min_ratio=self.pcb_min_ratio, 
                                    max_ratio=self.pcb_max_ratio)
        clamped_all_checks = torch.sign(all_checks) * all_checks_abs
        
        self_pcb = self.normalize(all_checks_abs, 1)**2
        self_pcb_act = torch.exp(n * self_pcb)
        
        cross_pcb = all_checks * torch.sum(all_checks, dim=0)
        cross_pcb_act = self.act(cross_pcb)
        
        task_pcb = self_pcb_act * cross_pcb_act

        scale = self.normalize(self.clamp(task_pcb, 1-pcb_ratio, 0), dim=1)
        
        merged_tv = torch.sum(clamped_all_checks * scale, dim=0) / torch.clamp(torch.sum(scale, dim=0), min=1e-12)
        
        return merged_tv, clamped_all_checks, scale
    
    def run(self):
        for task_id in range(self.num_tasks):

            for client_id, client in enumerate(self.clients):
                if task_id < len(client.data["task"]):
                    task = client.data["task"][task_id]
                    train_nodes = task["train_mask"].sum().item()
                    val_nodes = task["val_mask"].sum().item()
                    test_nodes = task["test_mask"].sum().item()
            for round in range(self.num_rounds):
                print("round "+str(round+1)+":")
                self.logger.write_round(round+1)
                self.sample()
                self.communicate()
                self.participating_clients = []
                for cid in self.sampled_clients:
                    if task_id < len(self.clients[cid].data["task"]):
                        self.participating_clients.append(cid)
                                                
                print("cid : ", end='')
                for cid in self.participating_clients:
                    print(cid, end=' ')
                    self.clients[cid].train(task_id)

                if(len(self.participating_clients)>0):
                    self.global_evaluate(task_id, round)
                    self.aggregate(task_id)
                    self.local_validate(task_id)
                    if round == self.num_rounds - 1:  
                        for eval_task_id in range(task_id + 1):
                            self.local_evaluate(eval_task_id, task_id)
                        self.calculate_AA_AF(task_id)
                    else:
                        self.local_evaluate(task_id, task_id)
                    
                    if hasattr(self.data, "task") and task_id < len(self.data.task):
                        if round == self.num_rounds - 1:  
                            self.global_evaluate(task_id, round)
                        else:

                            self.global_evaluate(task_id, None)

    def aggregate(self, task_id=None):
        if task_id is not None and len(self.participating_clients) > 0:
            task_total_samples = 0
            client_task_samples = {}
            
            for cid in self.participating_clients:
                task = self.clients[cid].data["task"][task_id]
                task_samples = task["train_mask"].sum().item()
                client_task_samples[cid] = task_samples
                task_total_samples += task_samples
                
            task_vectors = []
            for cid in self.participating_clients:
                client_task_vector = []
                for client_param, global_param in zip(self.clients[cid].model.parameters(), self.model.parameters()):
                    client_task_vector.append((client_param - global_param).view(-1))
                flat_task_vector = torch.cat(client_task_vector)
                task_vectors.append(flat_task_vector)
            
            flat_task_checks = torch.stack(task_vectors)
            n, d = flat_task_checks.shape
            
            weighted_task_vectors = []
            for i, cid in enumerate(self.participating_clients):
                weight = client_task_samples[cid] / task_total_samples
                weighted_task_vectors.append(flat_task_checks[i] * weight)

            weighted_flat_task_checks = torch.stack(weighted_task_vectors)

            merged_tv, _, scales = self.pcb_merge(weighted_flat_task_checks)
            
            start_idx = 0
            for global_param in self.model.parameters():
                # print(global_param.shape)
                param_size = global_param.numel()
                # print(merged_tv[start_idx:start_idx + param_size].view(global_param.shape).detach().cpu().numpy())
                param_update = merged_tv[start_idx:start_idx + param_size].view(global_param.shape)
                global_param.data.add_(param_update)
                start_idx += param_size

    
    def global_evaluate(self, task_id=None, curr_round=None):

        if task_id is None:
            self.model.eval()
            with torch.no_grad():
                task_data = self.task_data(0, self.data, {"train_mask": torch.zeros_like(self.data.test_mask), "val_mask": torch.zeros_like(self.data.test_mask), "test_mask": self.data.test_mask})
                g = pyg_to_dgl(task_data)
                out = self.model(g)
                out = F.log_softmax(out, 1)
                loss = F.nll_loss(out[self.data.test_mask], self.data.y[self.data.test_mask])
                pred = out[self.data.test_mask].max(dim=1)[1]
                acc = pred.eq(self.data.y[self.data.test_mask]).sum().item() / self.data.test_mask.sum().item()
                print("test_loss : "+format(loss.item(), '.4f'))
                self.logger.write_test_loss(loss.item())
                print("test_acc : "+format(acc, '.4f'))
                self.logger.write_test_acc(acc)
            return
            
        self.model.eval()
        global_acc_values = []
        
        for eval_task_id in range(task_id + 1):
            with torch.no_grad():
                if hasattr(self.data, "task") and eval_task_id < len(self.data.task):
                    task = self.data.task[eval_task_id]
                    task_data = self.task_data(eval_task_id, self.data, task)
                    g = pyg_to_dgl(task_data)
                    
                    
                    out = self.model(g)
                    out = F.log_softmax(out, 1)
                    
                    test_mask = task["test_mask"]
                    if test_mask.sum() == 0:
                        print(f"Warning: Empty test mask for global task {eval_task_id}")
                        continue
                    
                    y_test = self.data.y[test_mask]
                    unique_classes, class_counts = torch.unique(y_test, return_counts=True)
                    
                    loss = F.nll_loss(out[test_mask], y_test)
                    pred = out[test_mask].max(dim=1)[1]
                    correct = (pred == y_test).sum()
                    nodes = y_test.shape[0]
                    
                    if nodes == 0:
                        print(f"Warning: No test samples for global task {eval_task_id}")
                        continue
                    
                    acc = (correct/nodes).item()
                    if torch.isnan(torch.tensor(acc)):
                        print(f"Warning: Accuracy is NaN for global task {eval_task_id}")
                        continue
                    
                    print(f"Global task {eval_task_id} - total:{nodes}, correct:{correct}, acc:{acc:.4f}, loss:{loss.item():.4f}")
                    
                    self.acc_matrix[task_id, eval_task_id] = acc
                    global_acc_values.append(acc)
                else:
                    print(f"Warning: No task data available for global task {eval_task_id}")
                    print(f"Falling back to local_evaluate for task {eval_task_id}")
                    task_acc = self.local_evaluate(eval_task_id, task_id)
                    global_acc_values.append(task_acc)
                    
        
        if curr_round is not None and curr_round == self.num_rounds - 1:

            self.calculate_AA_AF(task_id)
            
        if global_acc_values:
            avg_acc = sum(global_acc_values) / len(global_acc_values)
            self.logger.write_test_acc(avg_acc)

    def calculate_AA_AF(self, task_id):
        if task_id == 0:
            aa = self.acc_matrix[task_id, 0].item()
            af = -1  
        else:

            aa = 0
            for i in range(task_id + 1):
                aa += self.acc_matrix[task_id, i].item()
            aa /= (task_id + 1)
            

            af = 0
            for i in range(task_id):  
                forgetting = self.acc_matrix[i, i].item() - self.acc_matrix[task_id, i].item()
                af += forgetting
            af /= task_id  
        
        print(f"[Task {task_id} Finished] AA: {aa:.4f}, AF: {af:.4f}")
        print("Accuracy Matrix:")
        print(self.acc_matrix[:task_id+1, :task_id+1])
        

        self.logger.write_aa(aa)
        if af != -1:
            self.logger.write_af(af)

    def local_evaluate(self, eval_task_id, curr_task_id=None):
        clients_test_loss = []
        clients_test_acc = []
        total_correct = 0
        total_nodes = 0
        self.model.eval()
        with torch.no_grad():
            for client_id, client in enumerate(self.clients):
                if client_id not in self.participating_clients:
                    continue
                with torch.no_grad():
                    task = client.data["task"][eval_task_id]
                    test_mask = task.get('test_mask', None)
                    
                    task_data = self.task_data(eval_task_id, client.data, task)
                    g = pyg_to_dgl(task_data)
                        
                    out = self.model(g)
                    out = F.log_softmax(out,1)
                    
                    y_test = g.ndata['y'][test_mask]
                    

                    if len(y_test.shape) > 1 and y_test.shape[1] > 1:  
                        y_test_indices = torch.max(y_test, 1)[1]
                    else:
                        y_test_indices = y_test
                        
                    unique_classes, class_counts = torch.unique(y_test_indices, return_counts=True)
                    
                    if len(y_test.shape) > 1 and y_test.shape[1] > 1:  
                        y_test_indices = torch.max(y_test, 1)[1]
                        loss = F.nll_loss(out[test_mask], y_test_indices)
                    else:
                        loss = F.nll_loss(out[test_mask], y_test)
                    

                    pred = out[test_mask].max(dim=1)[1]
                    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                        y_test_indices = torch.max(y_test, 1)[1]
                        correct = (pred == y_test_indices).sum()
                    else:
                        correct = (pred == y_test).sum()
                    
                    nodes = test_mask.sum().item()
                    
                    total_correct += correct.item()
                    total_nodes += nodes
                    
                    if nodes == 0:
                        print(f"Warning: No test samples for client {client_id}")
                        continue
                    
                    acc = (correct/nodes).item()
                    if torch.isnan(torch.tensor(acc)):
                        print(f"Warning: Accuracy is NaN for client {client_id}")
                        continue
                    
                    clients_test_loss.append(loss.item())
                    clients_test_acc.append(acc)
            
        mean_test_acc = total_correct / total_nodes if total_nodes > 0 else 0.0
        
        if curr_task_id is not None:
            self.acc_matrix[curr_task_id, eval_task_id] = mean_test_acc
            
        mean_val_loss = np.mean(clients_test_loss)
        std_val_loss = np.std(clients_test_loss)
        mean_val_acc = np.mean(clients_test_acc)
        std_val_acc = np.std(clients_test_acc)
        
        print("mean_test_loss :"+format(mean_val_loss, '.4f'))
        self.logger.write_mean_val_loss(mean_val_loss)
        print("mean_test_acc :"+format(mean_val_acc, '.4f'))
        self.logger.write_mean_val_acc(mean_val_acc)
        
        return mean_test_acc

    def local_validate(self, task_id):
        clients_val_loss = []
        clients_val_acc = []
        self.model.eval()
        with torch.no_grad():
            for client_id, client in enumerate(self.clients):
                if client_id not in self.participating_clients:
                    continue
                with torch.no_grad():
                    task = client.data["task"][task_id]
                    val_mask = task.get('val_mask', None)

                    task_data = self.task_data(task_id, client.data, task)
                    g = pyg_to_dgl(task_data)

                    out = self.model(g)
                    out = F.log_softmax(out,1)
                    

                    y_val = g.ndata['y'][val_mask]
                    
                    if len(y_val.shape) > 1 and y_val.shape[1] > 1:  
                        y_val_indices = torch.max(y_val, 1)[1]
                    else:
                        y_val_indices = y_val
                        
                    unique_classes, class_counts = torch.unique(y_val_indices, return_counts=True)
                    
                    if len(y_val.shape) > 1 and y_val.shape[1] > 1:  
                        y_val_indices = torch.max(y_val, 1)[1]
                        loss = F.nll_loss(out[val_mask], y_val_indices)
                    else:
                        loss = F.nll_loss(out[val_mask], y_val)
                    
                    pred = out[val_mask].max(dim=1)[1]
                    if len(y_val.shape) > 1 and y_val.shape[1] > 1:
                        y_val_indices = torch.max(y_val, 1)[1]
                        correct = (pred == y_val_indices).sum()
                    else:
                        correct = (pred == y_val).sum()
                    
                    total = val_mask.sum().item()
                    
                    if total == 0:
                        print(f"Warning: No validation samples for client {client_id}")
                        continue
                    
                    acc = (correct/total).item()
                    print(f"Client {client_id} - total:{total}, correct:{correct}, acc:{acc:.4f}, loss:{loss.item():.4f}")
                    clients_val_loss.append(loss.item())
                    clients_val_acc.append(acc)
            
        if not clients_val_loss or not clients_val_acc:
            print("Warning: No valid validation results collected from any client")
            return
            
        mean_val_loss = np.mean(clients_val_loss)
        std_val_loss = np.std(clients_val_loss)
        mean_val_acc = np.mean(clients_val_acc)
        std_val_acc = np.std(clients_val_acc)
        
        print("mean_val_loss :"+format(mean_val_loss, '.4f'))
        self.logger.write_mean_val_loss(mean_val_loss)
        print("std_val_loss :"+format(std_val_loss, '.4f'))
        print("mean_val_acc :"+format(mean_val_acc, '.4f'))
        self.logger.write_mean_val_acc(mean_val_acc)
        print("std_val_acc :"+format(std_val_acc, '.4f'))

class MOTIONClient(BaseClient):
    def __init__(self, args, model, data):
        super().__init__(args, model, data)
        self.model = model
        self.data = data
        self.args = args
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.combined_g_list = []
        self.method = args.method
        self.num_epochs = args.num_epochs
        if self.method == "DYGRA":
            self.gcl_method = DYGRA_reservior
            
        self.gcl = self.gcl_method(self.model, self.optimizer, args.num_classes, args.buffer_size, self.args)    
        
        self.node_reduction_rate = args.node_reduction_rate if hasattr(args, 'node_reduction_rate') else 0.5
        self.nfp = args.nfp if hasattr(args, 'nfp') else True
        self.use_moe = args.use_moe if hasattr(args, 'use_moe') else False
        self.k_list = torch.tensor(args.k_list, device=self.device) if hasattr(args, 'k_list') else torch.tensor([0.2, 0.4, 0.6, 0.8], device=self.device)
        

        self.use_node_positional = args.use_node_positional if hasattr(args, 'use_node_positional') else True
        self.use_node_mmd = args.use_node_mmd if hasattr(args, 'use_node_mmd') else True
        self.use_node_mahalanobis = args.use_node_mahalanobis if hasattr(args, 'use_node_mahalanobis') else True
        self.similarity_threshold = args.similarity_threshold if hasattr(args, 'similarity_threshold') else 0.7
        

        self.global_node_idxs = {}  
        self.prev_node_idxs = set()  
        

    def train(self, task_id):
        task = self.data["task"][task_id]
        task_data = self.task_data(task_id, self.data, task)
        g = pyg_to_dgl(task_data)

        num_nodes = g.number_of_nodes()
        
        initial_nodes_count = len(self.prev_node_idxs)
        
        node_idxs = torch.zeros(num_nodes, dtype=torch.long)
        
        new_nodes_mask = torch.zeros(num_nodes, dtype=torch.bool)
        new_nodes_count = 0
        
        for i in range(num_nodes):
            node_feature = g.ndata['x'][i].cpu().numpy()
            feature_hash = hash(node_feature.tobytes())
            node_key = (feature_hash, task_id)
            
            if node_key not in self.global_node_idxs:
                self.global_node_idxs[node_key] = len(self.global_node_idxs)
                new_nodes_mask[i] = True
                new_nodes_count += 1
            
            node_idxs[i] = self.global_node_idxs[node_key]
        
        for i in range(num_nodes):
            if new_nodes_mask[i]:
                node_feature = g.ndata['x'][i].cpu().numpy()
                feature_hash = hash(node_feature.tobytes())
                self.prev_node_idxs.add((feature_hash, task_id))
        
        g.ndata['node_idxs'] = node_idxs.to(self.device)
        g.ndata['new_nodes_mask'] = new_nodes_mask.to(self.device)
    
        for key, value in task.items():
            if isinstance(value, torch.Tensor) and key.endswith("_mask"):
                if key == 'val_mask':
                    g.ndata['valid_mask'] = value
                else:
                    g.ndata[key] = value
        
        if self.args.nfp: 
            er_buffer = self.gcl.update_er_buffer(g)
        else:
            er_buffer = []
            
        if task_id == 0:
            combined_g, self.c2n, self.n2c = combine_graph(g, device=self.device)
        else:
            combined_g, self.c2n, self.n2c = combine_graph(g, self.coarsened_g, self.C, self.c2n, self.n2c, self.device)
        
        replay_nodes = self.n2c[torch.tensor(er_buffer)]
        self.combined_g_list.append(combined_g)
        
        features = combined_g.ndata['x']
        labels = torch.max(combined_g.ndata['y'], 1).indices
        loss_func = nn.CrossEntropyLoss()
        
        valid_mask = combined_g.ndata['valid_mask']
        best_model_path = 'best_model_stat_dict'
        best_bac = 0
        print(f"num_epochs:{self.num_epochs}")
        for epoch in range(self.num_epochs):
            self.gcl.observe(self.combined_g_list, task_id, loss_func)
            valid_bac, valid_f1, valid_acc = evaluate(self.gcl, copy.deepcopy(combined_g), features, labels, valid_mask)
            if valid_bac > best_bac:
                best_bac = valid_bac
                torch.save(self.model.state_dict(), best_model_path)

        self.gcl.net.load_state_dict(torch.load(best_model_path,weights_only=False))
        
        self.gcl.net.return_hidden = True
        combined_g_copy = copy.deepcopy(combined_g)
        combined_g_copy.add_edges(combined_g_copy.nodes(), combined_g_copy.nodes())
        node_hidden_features = self.gcl.net(combined_g_copy, combined_g_copy.ndata['x']).detach()
        self.gcl.net.return_hidden = False
        
        
        if self.use_moe:
            
            self.coarsened_g, self.C, self.c2n, self.n2c = node_coarsening_with_moe(
                g=combined_g,
                node_features=node_hidden_features,  
                node_reduction=self.node_reduction_rate,
                replay_nodes=replay_nodes,
                device=self.device,
                k_list=self.k_list,
                c2n=self.c2n if hasattr(self, 'c2n') else None,
                n2c=self.n2c if hasattr(self, 'n2c') else None,
                use_node_positional=self.use_node_positional,
                use_node_mmd=self.use_node_mmd,
                use_node_mahalanobis=self.use_node_mahalanobis,
                similarity_threshold=self.similarity_threshold
            )

        else:
            self.coarsened_g, self.C, self.c2n, self.n2c = graph_coarsening(
                g=combined_g, 
                node_hidden_features=node_hidden_features, 
                c2n=self.c2n if hasattr(self, 'c2n') else {}, 
                n2c=self.n2c if hasattr(self, 'n2c') else torch.zeros(combined_g.number_of_nodes(), dtype=torch.long, device=self.device), 
                train_ratio=0.6, 
                reduction=self.node_reduction_rate, 
                replay_nodes=replay_nodes, 
                device=self.device
            )
            
        
        print(f"Successfully completed training for task {task_id}")
        



