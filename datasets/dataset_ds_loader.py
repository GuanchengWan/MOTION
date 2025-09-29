from datasets.dataset_loader import load_dataset
import torch
from torch_geometric.data import Data
from datasets.partition import louvain_partitioner
import numpy as np


def load_ds_dataset(args, data):
    if args.specified_domain_skew_task == "ogbn_arxiv_years":

        node_years = data.node_year.view(-1).tolist()
        year_dict = {}
        for node_id, year in enumerate(node_years):
            if year >= 2000:
                if year not in year_dict:
                    year_dict[year] = []
                year_dict[year].append(node_id)
        clients_nodes = []
        for value in year_dict.values():
            clients_nodes.append(value)

        if args.num_clients < 21:
            while len(clients_nodes) > args.num_clients:
                client_data_sizes = [len(clients_nodes[i]) for i in range(len(clients_nodes))]
                min_id = np.argmin(client_data_sizes)
                min_len_client = clients_nodes.pop(min_id)
                client_data_sizes.pop(min_id)
                min_id2 = np.argmin(client_data_sizes)
                clients_nodes[min_id2].extend(min_len_client)

        clients_data = partition_by_node(data, clients_nodes)

        while len(clients_data) < args.num_clients:
            client_data_sizes = [len(clients_data[i]) for i in range(len(clients_data))]
            max_id = np.argmax(client_data_sizes)
            max_len_client = clients_data.pop(max_id)
            append_clients = louvain_partitioner(max_len_client, 2)
            clients_data.extend(append_clients)

        return clients_data


def partition_by_node(data, clients_nodes):
    clients_data = []

    for nodes in clients_nodes:
        node_map = {node: i for i, node in enumerate(nodes)}

        sub_edge_index = []
        for i in range(data.edge_index.size(1)):
            if data.edge_index[0, i].item() in node_map and data.edge_index[1, i].item() in node_map:
                sub_edge_index.append([node_map[data.edge_index[0, i].item()], node_map[data.edge_index[1, i].item()]])
        sub_edge_index = torch.tensor(sub_edge_index, dtype=torch.long).t().contiguous()

        sub_x = data.x[nodes]
        sub_y = data.y[nodes]
        sub_data = Data(x=sub_x, edge_index=sub_edge_index, y=sub_y)

        if hasattr(data, "train_mask"):
            train_mask = torch.zeros(len(nodes), dtype=torch.bool)
            for node in nodes:
                train_mask[node_map[node]] = data.train_mask[node]
            sub_data.train_mask = train_mask

        if hasattr(data, "val_mask"):
            val_mask = torch.zeros(len(nodes), dtype=torch.bool)
            for node in nodes:
                val_mask[node_map[node]] = data.val_mask[node]
            sub_data.val_mask = val_mask

        if hasattr(data, "test_mask"):
            test_mask = torch.zeros(len(nodes), dtype=torch.bool)
            for node in nodes:
                test_mask[node_map[node]] = data.test_mask[node]
            sub_data.test_mask = test_mask

        clients_data.append(sub_data)

    return clients_data
