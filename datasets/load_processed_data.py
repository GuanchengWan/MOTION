import os

import torch
import numpy as np
from datasets.partition import *
from datasets.processing import *
import random
from collections import Counter
def get_folders(path):
    folders = []
    files = os.listdir(path)
    for file in files:
        if os.path.isdir(os.path.join(path, file)):
            folders.append(file)
    return folders


def load_processed_data(args, data):
    current_path = os.path.abspath(__file__)
    processed_data_path = os.path.join(os.path.dirname(current_path), 'processed_data')

    if args.skew_type == "label_skew":
        cur_split_name = args.task + "_" + args.skew_type + "_" + args.dataset + "_" + str(args.num_clients) \
                        + "_clients_seed_" + str(args.seed) + "_alpha_" + str(args.dirichlet_alpha) + "_num_classes_per_task_"+str(args.num_classes_per_task)
    else:
        cur_split_name = args.task + "_" + args.skew_type + "_" + args.dataset + "_" + str(args.num_clients) \
                        + "_clients_seed_" + str(args.seed)+ "num_classes_per_task_"+str(args.num_classes_per_task)

    cur_split_path = os.path.join(processed_data_path, cur_split_name)

    print("splitting data...")
    if not os.path.exists(cur_split_path):
        if args.evaluation_mode == "global_model_on_independent_data":
            client_data, server_data = split_train_val_test_inductive(data, args.train_val_test_split)
            clients_nodes = []
            if args.skew_type == "label_skew":
                clients_nodes = dirichlet_partitioner(client_data, args.num_clients, args.dirichlet_alpha,
                                                      args.least_samples, args.dirichlet_try_cnt)
            elif args.skew_type == "domain_skew":
                clients_nodes = louvain_partitioner(client_data, args.num_clients)

            clients_data = [get_subgraph_by_node(client_data, clients_nodes[i]) for i in range(len(clients_nodes))]
            clients_data = [split_train_val_test(clients_data[i], args.train_val_test_split) for i in
                            range(len(clients_data))]
            if args.dataset_split_metric == "inductive":
                clients_data = [split_data_inductive(data) for data in clients_data]
            for client_id in range(args.num_clients):
                task_file= class_incremental(data=clients_data[client_id], 
                                            classes_per_task=args.num_classes_per_task, 
                                            train_split=args.train_val_test_split[0], 
                                            val_split=args.train_val_test_split[1], 
                                            test_split=args.train_val_test_split[2], 
                                            shuffle_task=args.shuffle_task)
                clients_data[client_id]["task"] = task_file

            clients = []
            max_tasks = max([len(clients_data[i]["task"]) for i in range(args.num_clients)])
            args.num_tasks = max_tasks

            server_data.test_mask = torch.ones(server_data.num_nodes, dtype=torch.bool)
            server_tasks = create_server_tasks_by_distribution(clients_data, server_data, args.num_tasks)
            server_data["task"] = server_tasks

            clients_data = clients_data
            server_data = server_data

        else:
            clients_nodes = []
            if args.skew_type == "label_skew":
                clients_nodes = dirichlet_partitioner(data, args.num_clients, args.dirichlet_alpha,
                                                      args.least_samples, args.dirichlet_try_cnt)
            elif args.skew_type == "domain_skew":
                clients_nodes = louvain_partitioner(data, args.num_clients)
            
            clients_data = [get_subgraph_by_node(data, clients_nodes[i]) for i in range(len(clients_nodes))]
            clients_data = [split_train_val_test(clients_data[i], args.train_val_test_split) for i in
                            range(len(clients_data))]

            if args.dataset_split_metric == "inductive":
                clients_data = [split_data_inductive(data) for data in clients_data]
            for client_id in range(args.num_clients):
                task_file= class_incremental(data=clients_data[client_id], 
                                            classes_per_task=args.num_classes_per_task, 
                                            train_split=args.train_val_test_split[0], 
                                            val_split=args.train_val_test_split[1], 
                                            test_split=args.train_val_test_split[2], 
                                            shuffle_task=args.shuffle_task)
                clients_data[client_id]["task"] = task_file
            clients = []

            max_tasks = max([len(clients_data[i]["task"]) for i in range(args.num_clients)])
            args.num_tasks = max_tasks
            
            server_nodes = []
            for client_data in clients_data:
                for node in range(len(client_data.global_map)):
                    if client_data.test_mask[node]:
                        server_nodes.append(client_data.global_map[node])
            server_data = get_subgraph_by_node(data, server_nodes)
            server_data.test_mask = torch.ones(len(server_nodes), dtype=torch.bool)
            server_data = server_data
            
            
            max_tasks = max([len(clients_data[i]["task"]) for i in range(args.num_clients)])
            args.num_tasks = max_tasks
            
            server_tasks = create_server_tasks_by_distribution(clients_data, server_data, args.num_tasks)
            server_data["task"] = server_tasks

            clients_data = clients_data

        os.mkdir(cur_split_path)
        for i, client_data in enumerate(clients_data):
            torch.save(client_data, os.path.join(cur_split_path, "client_" + str(i)+".pt"))
        torch.save(server_data, os.path.join(cur_split_path, "server.pt"))

        print("data saved as " + cur_split_name)
        return clients_data, server_data
    else:
        print("loading data from /" + cur_split_name)
        clients_data = []
        print(args.shuffle_task)
        for i in range(args.num_clients):
            client_data = torch.load(os.path.join(cur_split_path, "client_" + str(i) +".pt"), weights_only=False)
            clients_data.append(client_data)
        max_tasks = max([len(clients_data[i]["task"]) for i in range(args.num_clients)])
        args.num_tasks = max_tasks
        # print(f"max is {args.num_tasks}&{max_tasks}")
        server_data = torch.load(os.path.join(cur_split_path, "server.pt"), weights_only=False)
        return clients_data, server_data

def class_incremental(data, classes_per_task, train_split, val_split, test_split, shuffle_task=False):
    num_nodes = data.x.shape[0]
    num_classes = data.y.max().item() + 1
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    for class_i in range(num_classes):
        class_i_node_mask = data.y == class_i
        num_class_i_nodes = class_i_node_mask.sum().item()
        
        class_i_node_list = torch.where(class_i_node_mask)[0].numpy()
        np.random.shuffle(class_i_node_list)

        train_end = int(train_split * num_class_i_nodes)
        val_end = int((train_split + val_split) * num_class_i_nodes)
        
        train_indices = class_i_node_list[:train_end]
        val_indices = class_i_node_list[train_end:val_end]
        test_indices = class_i_node_list[val_end:num_class_i_nodes]
        
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
    num_tasks = (num_classes + classes_per_task - 1) // classes_per_task
    label_to_task = {}
    drop_signal = False
    
    shuffled_classes = list(range(num_classes))
    if shuffle_task:
        random.shuffle(shuffled_classes)
    
    for task_id in range(num_tasks):
        left = task_id * classes_per_task
        right = min((task_id + 1) * classes_per_task, num_classes)
        if right < (task_id + 1) * classes_per_task:
            drop_signal = True        
    
        for ptr in range(left, right):
            label_to_task[shuffled_classes[ptr]] = task_id
    
    if drop_signal:
        num_tasks -= 1
    tasks = [{"train_mask": torch.zeros_like(train_mask).bool(), 
              "val_mask": torch.zeros_like(val_mask).bool(), 
              "test_mask": torch.zeros_like(test_mask).bool()} for _ in range(num_tasks)] 

    for class_i in range(num_classes):
        class_i_train = train_mask & (data.y == class_i)
        class_i_val = val_mask & (data.y == class_i)
        class_i_test = test_mask & (data.y == class_i)
        task_i = label_to_task[class_i]
        if task_i == num_tasks:
            continue
        tasks[task_i]["train_mask"] = tasks[task_i]["train_mask"] | class_i_train
        tasks[task_i]["val_mask"] = tasks[task_i]["val_mask"] | class_i_val
        tasks[task_i]["test_mask"] = tasks[task_i]["test_mask"] | class_i_test

    np.random.shuffle(tasks)
    
    return tasks

def create_server_tasks_by_distribution(clients_data, server_data, num_tasks):
    import torch
    import numpy as np
    from collections import Counter
    
    server_tasks = []
    
    server_nodes = list(range(server_data.num_nodes))
    server_labels = server_data.y.cpu().numpy()
    
    all_classes = np.unique(server_labels)
    
    num_clients = len(clients_data)
    
    total_classes = set()
    for client_data in clients_data:
        client_labels = client_data.y.cpu().numpy()
        total_classes.update(set(np.unique(client_labels)))
    
    client_task_classes = {}
    for client_id, client_data in enumerate(clients_data):
        client_task_classes[client_id] = {}
        for task_id in range(min(num_tasks, len(client_data["task"]))):
            task = client_data["task"][task_id]
            task_nodes = torch.where(task["train_mask"] | task["val_mask"] | task["test_mask"])[0]
            task_labels = [client_data.y[node_idx].item() for node_idx in task_nodes]
            client_task_classes[client_id][task_id] = Counter(task_labels)
    
    allocated_nodes = set()
    
    for task_id in range(num_tasks):
        task_class_distribution = {} 
        
        for client_id in client_task_classes:
            if task_id in client_task_classes[client_id]:
                for class_label, count in client_task_classes[client_id][task_id].items():
                    if class_label not in task_class_distribution:
                        task_class_distribution[class_label] = [0, 0]
                    task_class_distribution[class_label][0] += 1  
                    task_class_distribution[class_label][1] += count 
        
        if not task_class_distribution:
            print(f"Warning: No class distribution information for task {task_id}")
            server_tasks.append({
                "train_mask": torch.zeros(server_data.num_nodes, dtype=torch.bool),
                "val_mask": torch.zeros(server_data.num_nodes, dtype=torch.bool),
                "test_mask": torch.zeros(server_data.num_nodes, dtype=torch.bool)
            })
            continue
            
        
        
        test_mask = torch.zeros(server_data.num_nodes, dtype=torch.bool)
        
        class_alloc_plan = {}
        min_nodes_per_class = 5 
        
        for class_label, [client_count, _] in task_class_distribution.items():
            available_nodes = [node for node in server_nodes 
                              if server_labels[node] == class_label and node not in allocated_nodes]
            
            if not available_nodes:
                print(f"Warning: No available nodes for class {class_label} in task {task_id}")
                continue
            
            allocation_ratio = client_count / num_clients
            
            total_class_nodes = sum(1 for node in server_nodes if server_labels[node] == class_label)
            
            nodes_to_allocate = max(
                min_nodes_per_class,
                int(total_class_nodes * allocation_ratio)
            )
            
            nodes_to_allocate = min(nodes_to_allocate, len(available_nodes))
            
            class_alloc_plan[class_label] = {
                "available": len(available_nodes),
                "to_allocate": nodes_to_allocate,
                "client_ratio": f"{client_count}/{num_clients}"
            }
        
        for class_label, info in class_alloc_plan.items():
            nodes_to_allocate = info["to_allocate"]
            if nodes_to_allocate <= 0:
                continue
                
            available_nodes = [node for node in server_nodes 
                              if server_labels[node] == class_label and node not in allocated_nodes]
            
            import random
            random.seed(task_id * 100 + class_label) 
            selected_nodes = random.sample(available_nodes, nodes_to_allocate)
            
            test_mask[selected_nodes] = True
            
            allocated_nodes.update(selected_nodes)
            

        task = {
            "train_mask": torch.zeros(server_data.num_nodes, dtype=torch.bool),  
            "val_mask": torch.zeros(server_data.num_nodes, dtype=torch.bool),    
            "test_mask": test_mask  
        }
        
        server_tasks.append(task)
        
        if len(allocated_nodes) == len(server_nodes):
            print(f"Warning: All server nodes have been allocated after task {task_id}")
            for i in range(task_id+1, num_tasks):
                server_tasks.append({
                    "train_mask": torch.zeros(server_data.num_nodes, dtype=torch.bool),
                    "val_mask": torch.zeros(server_data.num_nodes, dtype=torch.bool),
                    "test_mask": torch.zeros(server_data.num_nodes, dtype=torch.bool)
                })
            break
    
    while len(server_tasks) < num_tasks:
        server_tasks.append({
            "train_mask": torch.zeros(server_data.num_nodes, dtype=torch.bool),
            "val_mask": torch.zeros(server_data.num_nodes, dtype=torch.bool),
            "test_mask": torch.zeros(server_data.num_nodes, dtype=torch.bool)
        })
    
    server_data.test_mask = torch.ones(server_data.num_nodes, dtype=torch.bool)
    
    for task_id, task in enumerate(server_tasks):
        test_count = task["test_mask"].sum().item()

        
        if test_count > 0:
            task_mask = task["test_mask"]
            task_nodes = torch.where(task_mask)[0]
            task_labels = [server_labels[node] for node in task_nodes]
            
            class_counts = Counter(task_labels)
    
    return server_tasks
