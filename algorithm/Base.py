import random

import numpy as np
import torch.nn.functional as F
import torch
import copy
from torch_geometric.data import Data
from torch_geometric.utils import coalesce

class BaseServer:
    def __init__(self, args, clients, model, data, logger):
        self.logger = logger
        self.sampled_clients = None
        self.clients = clients
        self.model = model
        self.cl_sample_rate = args.cl_sample_rate
        self.num_rounds = args.num_rounds
        self.num_epochs = args.num_epochs
        self.data = data
        self.args = args
        self.num_total_samples = sum([client.num_samples for client in self.clients])
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.num_tasks = args.num_tasks
        self.acc_matrix = torch.zeros((self.num_tasks, self.num_tasks)).to(self.device)
        
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
                        
                avg_train_loss = 0

                task_total_samples = 0
                for cid in self.participating_clients:
                    task = self.clients[cid].data["task"][task_id]
                    task_total_samples += task["train_mask"].sum().item()
                for cid in self.participating_clients:
                    print(cid, end=' ')
                    for epoch in range(self.num_epochs):
                        self.clients[cid].round = round
                        loss = self.clients[cid].train(task_id)

                        task = self.clients[cid].data["task"][task_id]
                        task_samples = task["train_mask"].sum().item()
                        avg_train_loss += loss * task_samples / task_total_samples

                if(len(self.participating_clients)>0):
                    self.aggregate(task_id)
                    self.local_validate(task_id)

                    if round == self.num_rounds - 1:
                        for eval_task_id in range(task_id + 1):
                            self.local_evaluate(eval_task_id, task_id)
                        self.calculate_AA_AF(task_id)
                    else:
                        self.local_evaluate(task_id, task_id)
                    self.global_evaluate(task_id, round)

    def communicate(self):
        for cid in self.sampled_clients:
            # self.clients[cid].model = copy.deepcopy(self.model)
            for client_param, server_param in zip(self.clients[cid].model.parameters(), self.model.parameters()):
                client_param.data.copy_(server_param.data)

    def sample(self):
        num_sample_clients = int(len(self.clients) * self.cl_sample_rate)
        sampled_clients = random.sample(range(len(self.clients)), num_sample_clients)
        self.sampled_clients = sampled_clients

    def aggregate(self, task_id=None):
        if task_id is not None and len(self.participating_clients) > 0:
            task_total_samples = 0
            client_task_samples = {}
            
            for cid in self.participating_clients:
                task = self.clients[cid].data["task"][task_id]
                task_samples = task["train_mask"].sum().item()
                client_task_samples[cid] = task_samples
                task_total_samples += task_samples
            
            for i, cid in enumerate(self.participating_clients):
                w = client_task_samples[cid] / task_total_samples
                for client_param, global_param in zip(self.clients[cid].model.parameters(), self.model.parameters()):
                    if i == 0:
                        global_param.data.copy_(w * client_param)
                    else:
                        global_param.data += w * client_param
        else:
            num_total_samples = sum([self.clients[cid].num_samples for cid in self.sampled_clients])
            for i, cid in enumerate(self.sampled_clients):
                w = self.clients[cid].num_samples / num_total_samples
                for client_param, global_param in zip(self.clients[cid].model.parameters(), self.model.parameters()):
                    if i == 0:
                        global_param.data.copy_(w * client_param)
                    else:
                        global_param.data += w * client_param

    def global_evaluate(self, task_id=None, curr_round=None):

        if task_id is None:
            self.model.eval()
            with torch.no_grad():
                embedding, out = self.model(self.data)
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
                    
                    
                    embedding, out = self.model(task_data)
                    
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
                    task_data = self.task_data(eval_task_id, client.data, task)
                    embedding, out = self.model(task_data)
                        
                    test_mask = task["test_mask"]
                    if test_mask.sum() == 0:
                        print(f"Warning: Empty test mask for client {client_id}")
                        continue
                    

                    y_test = client.data.y[test_mask]
                    unique_classes, class_counts = torch.unique(y_test, return_counts=True)
                    loss = F.nll_loss(out[test_mask], y_test)
                    pred = out[test_mask].max(dim=1)[1]
                    correct = (pred == y_test).sum()
                    nodes = y_test.shape[0]
                    
                    total_correct += correct.item()
                    total_nodes += nodes
                    
                    if nodes == 0:
                        print(f"Warning: No test samples for client {client_id}")
                        continue
                        
                    acc = (correct/nodes).item()
                    if torch.isnan(torch.tensor(acc)):
                        print(f"Warning: Accuracy is NaN for client {client_id}")
                        continue
                        
                    print(f"Client {client_id} - total:{nodes}, correct:{correct}, acc:{acc:.4f}, loss:{loss.item():.4f}")
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
        print("std_test_loss :"+format(std_val_loss, '.4f'))
        print("mean_test_acc :"+format(mean_val_acc, '.4f'))
        self.logger.write_mean_val_acc(mean_val_acc)
        print("std_test_acc :"+format(std_val_acc, '.4f'))
        
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
                    task_data = self.task_data(task_id,client.data,task)
                    embedding,out = self.model(task_data)
                        
                    val_mask = task["val_mask"]
                    if val_mask.sum() == 0:
                        print(f"Warning: Empty validation mask for client {client_id}")
                        continue
                        

                    y_val = client.data.y[val_mask]
                    unique_classes, class_counts = torch.unique(y_val, return_counts=True)
                    
                    loss = F.nll_loss(out[val_mask], y_val)

                    pred = out[val_mask].max(dim=1)[1]
                    correct = (pred == y_val).sum()
                    total = y_val.shape[0]
                    
                    if total == 0:
                        print(f"Warning: No validation samples for client {client_id}")
                        continue
                        
                    acc = (correct/total).item()
                    if torch.isnan(torch.tensor(acc)):
                        print(f"Warning: Accuracy is NaN for client {client_id}")
                        continue
                        
                    print(f"Client {client_id} - total:{total}, correct:{correct}, acc:{acc:.4f}, loss:{loss.item():.4f}")
                    clients_val_loss.append(loss.item())
                    clients_val_acc.append(acc)
                    
        if not clients_val_loss or not clients_val_acc:
            print("Warning: No valid results collected from any client")
            return
            
        mean_val_loss = np.mean(clients_val_loss)
        std_val_loss = np.std(clients_val_loss)
        mean_val_acc = np.mean(clients_val_acc)
        std_val_acc = np.std(clients_val_acc)
        
        print("mean_val_loss :"+format(mean_val_loss, '.4f'))
        self.logger.write_mean_val_loss(mean_val_loss)
        print("mean_val_acc :"+format(mean_val_acc, '.4f'))
        self.logger.write_mean_val_acc(mean_val_acc)


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
        
    def task_data(self, task_id, whole_data, task):
        handled = task["train_mask"] | task["val_mask"] | task["test_mask"]
        masked_edge_index = self.edge_masking(whole_data.edge_index, handled=handled, device=self.device)
        task_data = Data(x=whole_data.x, edge_index=masked_edge_index, y=whole_data.y)
        return task_data
    
    def edge_masking(self,edge_index, handled, device):
        num_nodes = edge_index.max().item()+1
        node_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
        for node in handled:
            node_mask[node] = True
        mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        edge_index = edge_index[:, mask].to(device)
        self_loop_indices = torch.tensor([[node, node] for node in handled], dtype=torch.long).t().to(device)
        edge_index = torch.cat([edge_index, self_loop_indices], dim=1).to(device)
        edge_index = coalesce(edge_index)
        return edge_index

class BaseClient:
    def __init__(self, args, model, data):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.data = data
        self.loss_fn = F.cross_entropy
        self.num_samples = len(data.x)
        self.args = args
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")

    def train(self,task_id):
        task = self.data["task"][task_id]
        task_data = self.task_data(task_id,self.data,task)
        self.model.train()
        self.optimizer.zero_grad()
        
        train_mask = task['train_mask']
        if train_mask.sum() == 0:
            print(f"Warning: Empty train mask for task_id {task_id}")
            return 0.0
            
        y_train = self.data.y[train_mask]
        unique_classes, class_counts = torch.unique(y_train, return_counts=True)
        class_distribution = dict(zip(unique_classes.tolist(), class_counts.tolist()))
    
        embedding,out = self.model(task_data)
        
        out = torch.clamp(out, min=-1e6, max=1e6)
        
        loss = self.loss_fn(out[train_mask], self.data.y[train_mask])
            
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        return loss.item()
    
    def task_data(self, task_id, whole_data, task):
        handled = task["train_mask"] | task["val_mask"] | task["test_mask"]
        masked_edge_index = self.edge_masking(whole_data.edge_index, handled=handled, device=self.device)
        task_data = Data(x=whole_data.x, edge_index=masked_edge_index, y=whole_data.y)
        return task_data
    
    def edge_masking(self,edge_index, handled, device):
        num_nodes = edge_index.max().item()+1
        node_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
        for node in handled:
            node_mask[node] = True
        mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        edge_index = edge_index[:, mask].to(device)
        self_loop_indices = torch.tensor([[node, node] for node in handled], dtype=torch.long).t().to(device)
        edge_index = torch.cat([edge_index, self_loop_indices], dim=1).to(device)
        edge_index = coalesce(edge_index)
        return edge_index