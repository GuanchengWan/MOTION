import argparse
import os

supported_datasets = ['cora', 'citeseer', 'pubmed', 'cs', 'physics', 'computers', 'photo',
                        'texas', 'wisconsin', 'cornell', 'squirrel', 'chameleon', 'crocodile', 'actor',
                        'twitch', 'fb100', 'Penn94', 'deezer', 'year', 'snap-patents', 'pokec', 'yelpchi', 'gamer',
                        'ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'genius',
                        'roman_empire', 'amazon_ratings', 'minesweeper', 'tolokers', 'questions']

supported_specified_task = ['cora']

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="cora")
current_path = os.path.abspath(__file__)
dataset_path = os.path.join(os.path.dirname(current_path), 'datasets')
root_dir = os.path.join(dataset_path, 'raw_data')
if not os.path.exists(root_dir):
    os.makedirs(root_dir)
parser.add_argument("--dataset_dir", type=str, default=root_dir)

log_path = os.path.join(os.path.dirname(current_path), 'logs')
if not os.path.exists(log_path):
    os.makedirs(log_path)
parser.add_argument("--logs_dir", type=str, default=log_path)

parser.add_argument("--specified_domain_skew_task", type=str, default=None)
parser.add_argument("--task", type=str, default="node_classification")
parser.add_argument("--skew_type", type=str, default="domain_skew")
# parser.add_argument("--train_val_test_split", type=list, default=[0.5, 0.25, 0.25])
# parser.add_argument("--train_val_test_split", type=list, default=[0.09, 0.3, 0.61])

parser.add_argument("--train_val_test_split", type=list, default=[0.2, 0.4, 0.4])
parser.add_argument("--dataset_split_metric", type=str, default="transductive")

parser.add_argument("--num_rounds", type=int, default=1)
parser.add_argument("--num_clients", type=int, default=2)
parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--cl_sample_rate", type=float, default=1.0)
parser.add_argument("--evaluation_mode", type=str, default="global")

parser.add_argument("--fed_algorithm", type=str, default="MOTION")

parser.add_argument("--model", type=str, default="GAT")
parser.add_argument("--hidden_dim", type=int, default=64)
parser.add_argument("--num_layers", type=int, default=3)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--learning_rate", type=float, default=0.005)
parser.add_argument("--weight_decay", type=float, default=4e-4)

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device_id", type=int, default=0)

parser.add_argument("--dirichlet_alpha", type=float, default=1.0)
parser.add_argument("--least_samples", type=int, default=5)
parser.add_argument("--dirichlet_try_cnt", type=int, default=1000)


parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--lr_g", type=float, default=1e-2)
parser.add_argument("--decay", type=float, default=0.3)
parser.add_argument("--num_tasks", type=int, default=5)
parser.add_argument("--num_classes", type=int, default=7)
parser.add_argument("--beta", type=float, default=0.01)
parser.add_argument("--reduction_rate", type=float, default=0.5)
parser.add_argument("--nfp", type=bool, default=True)
parser.add_argument("--buffer_size", type=int, default=200)
parser.add_argument('--method', type=str, default="DYGRA")
parser.add_argument("--num_classes_per_task", type=int, default=1)
parser.add_argument("--shuffle_task", type=bool, default=False)

parser.add_argument("--use_moe", type=bool, default=True, help="Whether to use MoE for coarsening")
parser.add_argument("--k_list", type=list, default=[0.2, 0.4, 0.6, 0.8], help="Sparsity ratios for MoE experts")
parser.add_argument("--hidden_spl", type=int, default=64, help="Hidden size for MoE SpLearner")
parser.add_argument("--num_layers_spl", type=int, default=2, help="Number of layers for MoE SpLearner")
parser.add_argument("--expert_select", type=int, default=2, help="Number of experts to select for each node")
parser.add_argument("--lam", type=float, default=1.0, help="Lambda parameter for combining expert outputs")
parser.add_argument("--node_reduction_rate", type=float, default=0.5, help="Reduction rate for node coarsening (0.0-1.0)")
parser.add_argument("--similarity_threshold", type=float, default=0.7, help="Similarity threshold for merging nodes")
parser.add_argument("--use_node_positional", type=bool, default=True, help="Whether to use positional features for nodes")
parser.add_argument("--use_node_mmd", type=bool, default=True, help="Whether to use MMD distance for node similarity")
parser.add_argument("--use_node_mahalanobis", type=bool, default=True, help="Whether to use Mahalanobis distance")
parser.add_argument("--hop_sizes", type=list, default=[1, 2], help="Hop sizes for subgraph decomposition")

args = parser.parse_args()
