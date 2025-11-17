<div align="center">

# 🚀 MOTION

### **Multi-Sculpt Evolutionary Coarsening for Federated Continual Graph Learning**

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

**Authors:** Guancheng Wan+, Fengyuan Ran +, Ruikang Zhang, Wenke Huang, Xuankun Rong, Guibin Zhang, Yuxin Wu, Bo Du, Mang Ye

<p align="center">
  <img src="image.png" alt="MOTION Framework Overview" width="800"/>
</p>

---

## 🔥 News

<div align="center">

🎉 **MOTION is accepted by NeurIPS 2025!** 🎉

---

## 📋 Table of Contents

- [✨ Overview](#-overview)
- [💥 Highlights](#-highlights)
- [⚒️ Installation](#️-installation)
- [🚀 Quick Start](#-quick-start)
- [⚙️ Key Hyperparameters](#️-key-hyperparameters)
- [📖 Project Layout](#-project-layout)
- [📌 Citation](#-citation)

## ✨ Overview

MOTION tackles **Federated Continual Graph Learning (FCGL)**, enabling incremental learning over dynamically evolving graphs distributed across multiple clients while preserving privacy.

### 🎯 Core Challenges

- **Topology Preservation**: Maintaining graph structure across tasks on clients during coarsening
- **Aggregation Conflicts**: Reducing server-side conflicts when combining client updates across evolving tasks

### 🏗️ Key Components

| Component        | Description                                                                                                                                                               |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **G-TMSC** | **Graph Topology-preserving Multi-Sculpt Coarsening** <br />• Similarity-guided multi-expert coarsening <br />• Preserves critical subgraph structures on clients |
| **G-EPAE** | **Graph-Aware Evolving Parameter Adaptive Engine** <br />• Topology-sensitive compatibility analysis <br />• Adaptive aggregation to minimize conflicting updates |

> 💡 **Result**: Enhanced stability and generalization for FCGL while respecting federated privacy constraints.

## 💥 Highlights

<div align="center">

| ✨ Feature                             | 📝 Description                                                           |
| -------------------------------------- | ------------------------------------------------------------------------ |
| **Topology Preservation**        | Maintains local graph structure during coarsening operations             |
| **Smart Aggregation**            | Adaptive, topology-aware server aggregation minimizes interference       |
| **Modular Architecture**         | Plug-in design for GNN backbones (GAT, GCN, etc.) and datasets           |
| **Federated Continual Learning** | Specialized for node classification in continual, federated environments |

</div>

## ⚒️ Installation

### 📋 System Requirements

| Component                   | Version | Notes                                  |
| --------------------------- | ------- | -------------------------------------- |
| **Python**            | 3.9+    | Required                               |
| **PyTorch**           | Latest  | GPU recommended for better performance |
| **PyTorch Geometric** | Latest  | Essential for graph neural networks    |
| **CUDA**              | 11.0+   | For GPU acceleration                   |

### 🚀 Quick Install

After installing PyTorch and PyTorch Geometric for your CUDA version:

```bash
pip install -U numpy scipy scikit-learn networkx tqdm ogb
```

## 🚀 Quick Start

Get started with MOTION in just a few minutes! Here's a simple federated continual learning experiment on the Cora dataset.

### 🎯 Basic Example

Run a small-scale experiment with 2 clients and GAT backbone:

```bash
python main.py \
  --fed_algorithm MOTION \
  --dataset cora \
  --model GAT \
  --num_clients 2 \
  --num_rounds 1 \
  --skew_type label_skew \
  --num_classes_per_task 1 \
  --num_classes 7 \
  --device_id 0 \
  --dirichlet_alpha 1.0 \
  --seed 0
```

### 💡 Important Notes

<div align="center">

| ⚙️ Configuration         | 📝 Details                                                                                                                                |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Custom Options**   | Configure additional parameters via `args.py`<br />• `reduction_rate`, `expert_select`<br />• `node_reduction_rate`, `k_list` |
| **Dataset Location** | Datasets stored in `datasets/raw_data/` by default <br />• Override with `--dataset_dir` flag                                        |

</div>

## ⚙️ Key Hyperparameters

### 📊 Core Configuration

| Parameter             | Options                                 | Description                        |
| --------------------- | --------------------------------------- | ---------------------------------- |
| **dataset**     | `cora`, `citeseer`, `pubmed`, ... | Target dataset for experiments     |
| **model**       | `GAT` (default), `GCN`, ...         | GNN backbone architecture          |
| **num_clients** | Integer                                 | Number of federated clients        |
| **num_rounds**  | Integer                                 | Communication rounds in federation |
| **skew_type**   | `label_skew`, `domain_skew`, ...    | Data partition strategy            |

### 🎯 Continual Learning Setup

| Parameter                      | Description                                                     |
| ------------------------------ | --------------------------------------------------------------- |
| **num_tasks**            | Total number of sequential tasks                                |
| **num_classes**          | Total number of classes across all tasks                        |
| **num_classes_per_task** | Classes learned per task                                        |
| **dirichlet_alpha**      | Controls label distribution non-IIDness (higher = more uniform) |

### 🧠 Model Architecture

| Parameter               | Description                 |
| ----------------------- | --------------------------- |
| **hidden_dim**    | Hidden layer dimensionality |
| **num_layers**    | Number of GNN layers        |
| **dropout**       | Dropout probability         |
| **learning_rate** | Optimization learning rate  |
| **weight_decay**  | L2 regularization strength  |

### 🔧 MOTION-Specific Parameters

| Parameter                     | Description                      |
| ----------------------------- | -------------------------------- |
| **reduction_rate**      | Graph coarsening reduction rate  |
| **expert_select**       | Expert selection strategy        |
| **node_reduction_rate** | Node-level reduction rate        |
| **k_list**              | List of k-hop neighborhood sizes |

### 📁 File System Configuration

<div align="center">

| Path                | Default                 | Override          |
| ------------------- | ----------------------- | ----------------- |
| **Datasets**  | `datasets/raw_data/`  | `--dataset_dir` |
| **Logs**      | `logs/`               | `--logs_dir`    |
| **Task Type** | `node_classification` | `--task`        |

</div>

## 📖 Project Layout

```
MOTION/
├── main.py                 # 🚀 Main entrypoint and experiment runner
├── args.py                 # ⚙️ Hyperparameter definitions and CLI flags
├── algorithm/              # 🧠 Core MOTION algorithm
│   ├── MOTION.py          # Main algorithm implementation
│   ├── Base.py            # Base federated learning framework
│   └── utils/             # Algorithm utilities
├── backbone/               # 🏗️ GNN backbone implementations
│   ├── GAT.py             # Graph Attention Network
│   └── ...                # Other GNN architectures
├── datasets/               # 📊 Dataset handling
│   ├── raw_data/          # Raw dataset files
│   ├── loaders/           # Data loading utilities
│   └── partitioners/      # Federated data partitioning
├── tasks/                  # 🎯 Task implementations
│   ├── node_classification_task.py
│   └── base_task.py       # Base task framework
├── utils/                  # 🔧 Utilities
│   ├── logging.py         # Experiment logging
│   ├── seeds.py           # Random seed management
│   └── taskflow.py        # Experiment orchestration
└── logs/                   # 📝 Experiment outputs and checkpoints
```

### 📁 Key Directories Overview

<div align="center">

| Directory                | Purpose                       | Key Files                    |
| ------------------------ | ----------------------------- | ---------------------------- |
| **`algorithm/`** | Core MOTION implementation    | `MOTION.py`, `Base.py`   |
| **`backbone/`**  | GNN model architectures       | `GAT.py`, `GCN.py`, etc. |
| **`datasets/`**  | Data loading and partitioning | Loaders, partitioners        |
| **`tasks/`**     | Learning task definitions     | Classification, regression   |
| **`utils/`**     | Helper utilities              | Logging, orchestration       |
| **`logs/`**      | Experiment outputs            | Checkpoints, results         |

</div>

## 📌 Citation

If you use MOTION in your research, please cite our paper:

```bibtex
@inproceedings{MOTION_NeurIPS25,
  title={{MOTION}: Multi-Sculpt Evolutionary Coarsening for Federated Continual Graph Learning},
  author={Wan, Guancheng and Ran, Fengyuan and Zhang, Ruikang and Huang, Wenke and Rong, Xuankun and Zhang, Guibin and Wu, Yuxin and Du, Bo and Ye, Mang},
  booktitle={NeurIPS},
  year={2025}
}
```



*Thank you for using MOTION! 🚀*
