import os
import math
import torch
import numpy as np
from tqdm.auto import tqdm
from copy import deepcopy
from inspect import getmembers, isfunction
from metric_calculators import get_metric_fns
import random


class FractionalDataloader:
    def __init__(self, dataloader, fraction, seed=None):
        self.dataloader_numel = len(dataloader.dataset)
        self.numel = int(fraction * self.dataloader_numel)

        self.batch_size = self.dataloader_numel / len(dataloader)
        self.num_batches = int(math.ceil(self.numel / self.batch_size))
        self.dataloader = dataloader
        self.dataset = self.dataloader.dataset
        self.seed = seed
    
    def __iter__(self):
        cur_elems = 0
        if self.seed is not None:
            self.dataloader.dataset.set_seed(self.seed)
            torch.manual_seed(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)
        it = iter(self.dataloader)
        while cur_elems < self.numel:
            try:
                x, y = next(it)
                cur_elems += x.shape[0]
                yield x, y
            except StopIteration:
                it = iter(self.dataloader)
                
        
    def __len__(self):
        return self.num_batches


def prepare_data(config, device='cuda'):
    """ Load all dataloaders required for experiment. """
    if isinstance(config, list):
        return [prepare_data(c, device) for c in config]
    
    dataset_name = config['name']
    
    import my_datasets.configs as config_module
    data_config = deepcopy(getattr(config_module, dataset_name))
    data_config.update(config)
    data_config['device'] = device

    if data_config['type'] == 'books':
        from my_datasets.books import prepare_train_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = None
    elif data_config['type'] == 'glue':
        from my_datasets.glue import prepare_train_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = None
    else:
        raise NotImplementedError(config['type'])
    
    if 'train_fraction' in data_config:
        for k, v in dict(train_loaders.items()).items():
            if k == 'splits':
                train_loaders[k] = [FractionalDataloader(x, data_config['train_fraction']) for x in v]
            elif not isinstance(v, list) and not isinstance(v, torch.Tensor):
                train_loaders[k] = FractionalDataloader(v, data_config['train_fraction'])

    return {
        'train': train_loaders,
        'test': test_loaders
    }

def prepare_bert(config, device, repair=False, classifier=False):
    from transformers import BertForMaskedLM, BertForSequenceClassification
    bases = []
    config_example = None
    for i, base_path in tqdm(enumerate(config['bases']), desc="Preparing Models"):
        if classifier:
            base_model = BertForSequenceClassification.from_pretrained(base_path)
        else:
            base_model = BertForMaskedLM.from_pretrained(base_path)
        config_example = base_model.config
        bases.append(base_model)
    if repair != False:
        from models.bert_bn import BertBNForMaskedLM
        new_model = BertBNForMaskedLM(config_example, rescale=False)
    elif classifier == True:
        new_model = deepcopy(base_model)
    else:
        new_model = BertForMaskedLM(config_example)
    return {'bases': bases, 'new': new_model}

def prepare_models(config, device='cuda', repair=False, classifier=False):
    """ Load all pretrained models in config. """
    if config['name'].startswith('bert'):
        return prepare_bert(config, device, repair=repair, classifier=classifier)
    else:
        # can add more models here
        raise NotImplementedError(config['name'])


def prepare_graph(config, classifier=False):
    """ Get graph class of experiment models in config. """
    if config['name'].startswith('bert'):
        import graphs.transformer_enc_graph as graph_module
        model_name = 'bert'
        graph = getattr(graph_module, model_name)
    else:
        raise NotImplementedError(config['name'])
    return graph


def get_merging_fn(name):
    """ Get alignment function from name. """
    import matching_functions
    matching_fns = dict([(k, v) for (k, v) in getmembers(matching_functions, isfunction) if 'match_tensors' in k])
    return matching_fns[name]


def prepare_experiment_config(config, type='vis'):
    """ Load all functions/classes/models requested in config to experiment config dict. """

    data = prepare_data(config['dataset'], device=config['device'])
    if config['eval_type'] == 'logits':
        config['model']['output_dim'] = len(data['test']['class_names'])
    else:
        config['model']['output_dim'] = 512
    new_config = {
        'graph': prepare_graph(config['model']),
        'data': data,
        'models': prepare_models(config['model'], device=config['device']),
        'merging_fn': get_merging_fn(config['merging_fn']),
        'metric_fns': get_metric_fns(config['merging_metrics']),
    }
    # Add outstanding elements
    for key in config:
        if key not in new_config:
            new_config[key] = config[key]
    return new_config

def prepare_lang_config(config, type='vis', repair=False, classifier=False):
    """ Load all functions/classes/models requested in config to experiment config dict. """
    data = prepare_data(config['dataset'], device=config['device'])
    new_config = {
        'graph': prepare_graph(config['model'], classifier=classifier),
        'data': data,
        'models': prepare_models(config['model'], device=config['device'], repair=repair, classifier=classifier),
        'merging_fn': get_merging_fn(config['merging_fn']),
        'metric_fns': get_metric_fns(config['merging_metrics']),
    }
    # Add outstanding elements
    for key in config:
        if key not in new_config:
            new_config[key] = config[key]
    return new_config


def get_config_from_name(name, device=None):
    """ Load config based on its name. """
    out = deepcopy(getattr(__import__('configs.' + name), name).config)
    if device is None and 'device' not in out:
        out['device'] = 'cuda'
    elif device is not None:
        out['device'] = device
    return out


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def contains_name(layer_name, node_list):
    for node in node_list:
        if node in layer_name:
            return True
    return False

def find_pairs(str_splits):
    pairs = []
    for i, str_split_i in enumerate(str_splits):
        try:
            split_i = set([int(k) for k in str_split_i.split('_')])
        except:
            continue
        for str_split_j in str_splits[i+1:]:
            try:
                split_j = set([int(k) for k in str_split_j.split('_')])
            except:
                continue
            if len(split_i.intersection(split_j)) == 0:
                pairs.append((str_split_i, str_split_j))
    return pairs


def split_str_to_ints(split):
    return [int(i) for i in split.split('_')]


def is_valid_pair(model_dir, pair, model_type):
    paths = os.listdir(os.path.join(model_dir, pair[0]))
    flag = True
    for path in paths:
        if f'{model_type}_v0.pth.tar' not in path:
            flag = False
    return flag


def inject_pair_language(config, pair):
    config['model']['bases'] = [os.path.join(config['model']['dir'], pair_item) for pair_item in pair]
    return config 
