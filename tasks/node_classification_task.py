from tasks.base_task import BaseTask
from datasets.dataset_loader import load_dataset
from datasets.dataset_ds_loader import load_ds_dataset
from backbone import get_model
from algorithm import get_server, get_client
from datasets.partition import *
from datasets.processing import *
import numpy as np
from utils.logger import DefaultLogger
import time
from datasets.load_processed_data import *


class NodeClassificationTask(BaseTask):
    def __init__(self, args):
        super(NodeClassificationTask, self).__init__(args)

        print(args)
        self.server_data = None
        self.clients_data = None
        self.server = None
        self.clients = []
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.dataset = load_dataset(args.train_val_test_split, args.dataset_dir, args.dataset)
        self.input_dim = self.dataset.num_node_features
        listy = self.dataset.y.tolist()
        out_dim = len(np.unique(listy))
        self.num_classes = out_dim

    def process_data(self):
        clients_data, server_data = load_processed_data(self.args, self.dataset)
        self.server_data = server_data
        self.clients_data = clients_data

    def init_server_client(self):
        clients = []
        for cid in range(self.args.num_clients):
            client_model = get_model(self.args.model, self.input_dim, self.args.hidden_dim, self.num_classes,
                                     self.args.num_layers, self.args.dropout)
            client_model = client_model.to(self.device)
            client_data = self.clients_data[cid]
            client_data = client_data.to(self.device)
            client = get_client(self.args.fed_algorithm, self.args, client_model, client_data)
            clients.append(client)

        logger = DefaultLogger(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()) + '-' + self.args.task + '-' +
                               self.args.dataset + '-' + self.args.fed_algorithm, self.args.logs_dir)

        server_model = get_model(self.args.model, self.input_dim, self.args.hidden_dim, self.num_classes, self.args.num_layers, self.args.dropout)
        server_model = server_model.to(self.device)
        self.server_data = self.server_data.to(self.device)
        self.server = get_server(self.args.fed_algorithm, self.args, clients, server_model, self.server_data, logger)
        self.clients = clients

    def run(self):
        self.process_data()
        self.init_server_client()
        self.server.run()
