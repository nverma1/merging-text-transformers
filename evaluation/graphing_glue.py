import os
import json
import argparse

import torch
import numpy as np

from tqdm import tqdm
from copy import deepcopy
from eval_glue import get_metrics
from safetensors.torch import load
from transformers import BertForSequenceClassification


pairs_0 = [1,2,3,4,1,2,3,1,2,1]
pairs_1 = [2,3,4,5,3,4,5,4,5,5]

task_to_outputs = {
    'cola': 2,
    'mnli': 3,
    'mrpc': 2,
    'qnli': 2,
    'qqp': 2,
    'rte': 2,
    'sst2': 2,
    'stsb': 1,
    'wnli': 2
}

lambdas = np.arange(21) / 20

def get_merged_state_dict(state_dict_1, state_dict_2, w=0.5,):
        """
        Post transformations, obtain state dictionary for merged model by linearly interpolating between 
        transformed models in each graph. By default all parameters are averaged, but if given an interp_w 
        weight, will be weightedly averaged instead.
        - interp_w (Optional): If None, all parameters of each model is averaged for merge. Otherwise, 
        interp_w is a list of len(num_models_to_merge), with weights bearing the importance of incorporating 
        features from each model into the merged result.
        Returns: state dict of merged model.
        """
        state_dict = {}
        merged_state_dict = deepcopy(state_dict_1)
        keys = list(state_dict_1.keys())
        try:
            for key in keys:
                if key in merged_state_dict:
                    param = state_dict_1[key]
                    if param.shape == merged_state_dict[key].shape:
                        new_value = state_dict_1[key] * w + state_dict_2[key] * (1-w)
                    state_dict[key] = new_value
        except RuntimeError as e:
            if 'size' not in str(e):
                raise e
        return state_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--originals', action='store_true', required=False, default=False)
    parser.add_argument('--task',required=True)
    parser.add_argument('--path', required=False)
    parser.add_argument('--merge-type', required=False)
    parser.add_argument('--outfile')
    parser.add_argument('--train-frac', required=False, default=1.0, type=float)
    parser.add_argument('--cache-dir', required=False)

    args = parser.parse_args()
    task = args.task
    cache_dir = args.cache_dir

    ROOT_PATH = f'models/'
    vanilla_lambda_lists = {}
    placeholder_model = BertForSequenceClassification.from_pretrained('google/multiberts-seed_0', num_labels=task_to_outputs[task])

    # loop through 10 experiments 
    for i in range(10):

        # load models 0 and 1
        if args.originals == True:
            model0_path = f'{ROOT_PATH}/trained/multiberts_new/{task}/seed_{pairs_0[i]}/model.safetensors'
            model1_path = f'{ROOT_PATH}/trained/multiberts_new/{task}/seed_{pairs_1[i]}/model.safetensors'
            model0_load = open(model0_path, 'rb')
            model1_load = open(model1_path, 'rb')
            model0_statedict = load(model0_load.read())
            model1_statedict = load(model1_load.read())
        else:
            new_path = os.path.join(ROOT_PATH, args.path, 'individual_models')

            model_file0 = f'match_tensors_permute_{args.merge_type}_0_{args.task}_seed_{pairs_0[i]}_b8_{args.task}{args.train_frac}_{pairs_0[i]}_{pairs_1[i]}.pt'
            model_file1 = f'match_tensors_permute_{args.merge_type}_1_{args.task}_seed_{pairs_1[i]}_b8_{args.task}{args.train_frac}_{pairs_0[i]}_{pairs_1[i]}.pt'
            model_path0 = os.path.join(new_path, model_file0)
            model_path1 = os.path.join(new_path, model_file1)
            model0_statedict = torch.load(model_path0)
            model1_statedict = torch.load(model_path1)

        vanilla_lambda_lists[i] = {}
        if task == 'mnli':
            vanilla_lambda_lists[i][0] =[]
            vanilla_lambda_lists[i][1] = []
        else:
            vanilla_lambda_lists[i][0] = []

        # loop through interpolation lambdas and get loss for each
        for l in tqdm(lambdas):
            merged_statedict = get_merged_state_dict(model0_statedict, model1_statedict, w=l)
            placeholder_model.load_state_dict(merged_statedict)
            metrics = get_metrics(args.task, placeholder_model, cache_dir)
            for j, metric in enumerate(metrics):
                vanilla_lambda_lists[i][j].append(metric['eval_loss'])

    if os.path.exists('results/glue') == False:
        os.makedirs('results/glue')
    
    with open(f'results/glue/{args.outfile}.json', 'w+') as out:
        json.dump(vanilla_lambda_lists, out)
    

            
if __name__ == "__main__":
    main()

