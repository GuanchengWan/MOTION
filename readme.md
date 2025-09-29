
# Description

This repository provides the full implementation of MOTION (Multi-Sculpt EvOluTIONary Coarsening for Federated Continual Graph Learning), the first framework for incremental, privacy-preserving learning on dynamic, distributed graphs. It includes:

- **Client-side G-TMSC module**: graph topology-preserving multi-expert coarsening that fuses structural metrics via similarity-guided scoring to retain key subgraph patterns.  
- **Server-side G-EPAE module**: graph-aware evolving parameter adaptive engine that builds a topology-sensitive compatibility matrix to weight and integrate client updates, reducing aggregation conflicts.  
- **Benchmark evaluation**: scripts to reproduce experiments on five real-world graph datasets (Cora, CiteSeer, PubMed, Amazon-Photo, CoAuthor-CS), demonstrating up to 30 % average accuracy gain over FedAvg and negative average forgetting rates.  
- **Dependencies & usage**: PyTorch, PyG, CUDA; ready-to-run training and evaluation pipelines, with configurable hyperparameters for reduction rate and expert selection.


# How to run?

```bash
python main.py \
  --fed_algorithm MOTION \
  --dataset cora \
  --model GAT \
  --num_clients 2 \
  --num_rounds 1 \
  --skew_type label_skew \
  --num_classes_per_task 1 \
  --num_classes 8 \
  --device_id 0 \
  --dirichlet_alpha 1.0 \
  --seed 0
```
Arguments:
--fed_algorithm (str): federated learning algorithm to use (e.g. MOTION).

--dataset (str): graph dataset name (cora, citeseer, pubmed, etc.).

--model (str): GNN backbone (GAT).

--num_clients (int): number of simulated federated clients.

--num_rounds (int): total communication rounds.

--skew_type (str): data partition strategy (label_skew).

--num_classes_per_task (int): number of classes assigned to each client’s local task.

--num_classes (int): total number of classes in the dataset.

--device_id (int): CUDA device index.

--dirichlet_alpha (float): concentration parameter for Dirichlet non-IID split (higher → more uniform).

--seed (int): random seed for reproducibility.