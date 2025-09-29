import os
import importlib


def get_all_models():
    return [model.split('.')[0] for model in os.listdir('backbone')
            if not model.find('__') > -1 and 'py' in model]


names = {}
for model in get_all_models():
    mod = importlib.import_module('backbone.' + model)
    class_name = model
    # class_name = {x.lower(): x for x in mod.__dir__()}[model.replace('_', '')]
    names[model] = getattr(mod, class_name)


def get_model(model_name, input_dim, hidden_dim, output_dim, num_layers, dropout):
    if model_name not in names:
        raise ValueError(f"Model {model_name} not found. Available models: {list(names.keys())}")
    if model_name == 'GAT':
        return names[model_name](input_dim, hidden_dim, output_dim, num_heads=8)


