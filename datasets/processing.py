import torch
from torch_geometric.data import Data


def split_train_val_test_inductive(data, train_val_test_p):
    p_sum = sum(train_val_test_p)
    proportion = [(p/p_sum) for p in train_val_test_p]

    indices = torch.randperm(data.num_nodes)

    train_size = int(proportion[0] * data.num_nodes)
    val_size = int(proportion[1] * data.num_nodes)
    test_size = data.num_nodes - train_size - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    test_indices_list = test_indices.tolist()
    test_node_map = {node: i for i, node in enumerate(test_indices_list)}

    test_edge_index = []
    for i in range(data.edge_index.size(1)):
        if data.edge_index[0, i].item() in test_node_map and data.edge_index[1, i].item() in test_node_map:
            test_edge_index.append([test_node_map[data.edge_index[0, i].item()], test_node_map[data.edge_index[1, i].item()]])
    test_edge_index = torch.tensor(test_edge_index, dtype=torch.long).t().contiguous()

    test_sub_x = data.x[test_indices]
    test_sub_y = data.y[test_indices]
    test_data = Data(x=test_sub_x, edge_index=test_edge_index, y=test_sub_y)
    test_mask_new = torch.ones(test_data.num_nodes, dtype=torch.bool)
    test_data.test_mask = test_mask_new

    train_indices_list = train_indices.tolist()
    val_indices_list = val_indices.tolist()
    train_indices_list.extend(val_indices_list)
    train_val_indices = train_indices_list
    train_val_node_map = {node: i for i, node in enumerate(train_val_indices)}

    train_val_edge_index = []
    for i in range(data.edge_index.size(1)):
        if data.edge_index[0, i].item() in train_val_node_map and data.edge_index[1, i].item() in train_val_node_map:
            train_val_edge_index.append([train_val_node_map[data.edge_index[0, i].item()], train_val_node_map[data.edge_index[1, i].item()]])
    train_val_edge_index = torch.tensor(train_val_edge_index, dtype=torch.long).t().contiguous()

    train_val_sub_x = data.x[train_val_indices]
    train_val_sub_y = data.y[train_val_indices]
    train_val_data = Data(x=train_val_sub_x, edge_index=train_val_edge_index, y=train_val_sub_y)

    if hasattr(data, "train_mask"):
        train_mask = torch.zeros(len(train_val_indices), dtype=torch.bool)
        train_mask1 = data.train_mask.tolist()
        for node in train_val_indices:
            # train_mask[node_map[node]] = data.train_mask[node].item()
            train_mask[train_val_node_map[node]] = train_mask1[node]
        train_val_data.train_mask = train_mask

    if hasattr(data, "val_mask"):
        val_mask = torch.zeros(len(train_val_indices), dtype=torch.bool)
        val_mask1 = data.val_mask.tolist()
        for node in train_val_indices:
            val_mask[train_val_node_map[node]] = val_mask1[node]
        train_val_data.val_mask = val_mask

    return test_data, train_val_data


def split_train_val(data, train_val_p):
    p_sum = sum(train_val_p)
    proportion = [(p/p_sum) for p in train_val_p]

    indices = torch.randperm(data.num_nodes)

    train_size = int(proportion[0] * data.num_nodes)
    val_size = data.num_nodes - train_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]


    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True

    data.train_mask = train_mask
    data.val_mask = val_mask

    return data


def split_train_val_test(data, train_val_test_p):
    p_sum = sum(train_val_test_p)
    proportion = [(p/p_sum) for p in train_val_test_p]

    indices = torch.randperm(data.num_nodes)

    train_size = int(proportion[0] * data.num_nodes)
    val_size = int(proportion[1] * data.num_nodes)
    test_size = data.num_nodes - train_size - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]


    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def split_data_inductive(data):
    train_nodes = []
    if not hasattr(data, "train_mask"):
        print("data is not split before inductive split")
        return data

    if not hasattr(data, "num_nodes"):
        print("data dont have attribute num_nodes when processing inductive split")
        return data

    for i in range(0, data.num_nodes):
        if data.train_mask[i]:
            train_nodes.append(i)

    new_edge_list = []
    for edge_id in data.edge_index:
        src = data.edge_index[0, edge_id].item()
        tgt = data.edge_index[1, edge_id].item()
        if src in train_nodes and tgt in train_nodes:
            new_edge_list.append((src, tgt))

    new_edge_index = torch.tensor(new_edge_list, dtype=torch.long)
    data.edge_index = new_edge_index
    return data


