import os
import importlib


def get_all_algorithms():
    return [model.split('.')[0] for model in os.listdir('algorithm')
            if not model.find('__') > -1 and 'py' in model]


fed_servers = {}
fed_clients = {}
for algorithm in get_all_algorithms():
    mod = importlib.import_module('algorithm.' + algorithm)
    server_class_name = algorithm + 'Server'
    client_class_name = algorithm + 'Client'
    # class_name = {x.lower(): x for x in mod.__dir__()}[model.replace('_', '')]
    fed_servers[algorithm] = getattr(mod, server_class_name)
    fed_clients[algorithm] = getattr(mod, client_class_name)


def get_server(model_name, args, clients, model, data, logger):
    return fed_servers[model_name](args, clients, model, data, logger)


def get_client(model_name, args, model, data):
    return fed_clients[model_name](args, model, data)


def load_client(args, model, data):
    if args.algorithm == "fedavg":
        from algorithm.Base import BaseClient
        from algorithm.FedAvg import FedAvgClient
        return FedAvgClient(args, model, data)


def load_server(args, clients, model, data, logger):
    if args.algorithm == "fedavg":
        from algorithm.Base import BaseServer
        from algorithm.FedAvg import FedAvgServer
        return FedAvgServer(args, clients, model, data, logger)

