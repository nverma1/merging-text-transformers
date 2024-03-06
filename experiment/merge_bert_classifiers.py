import os
import argparse
import json
import random
import torch
import numpy as np

from copy import deepcopy
from tqdm.auto import tqdm

from utils import *
from model_merger import ModelMerge

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


save_dir_root = 'models'

def run_auxiliary_experiment(merging_fn, merge_type, experiment_config, pairs, device, args):
    for pair in tqdm(pairs, desc='Evaluating Pairs...'):
        experiment_config = inject_pair_language(experiment_config, pair)

        
        if args.task != 'mlm':
            config = prepare_lang_config(experiment_config, type='lang', classifier=True)
        else:
            config = prepare_lang_config(experiment_config, type='lang', classifier=False)
        train_loaders = [config['data'][i]['train']['full'] for i in range(len(config['data']))]
        base_models = config['models']['bases']
        
        Grapher = config['graph']
        if args.task != 'mlm':
            # set classifier to true
            graphs = [Grapher(deepcopy(base_model), merge_type=merge_type, qk=args.qk, classifier=True).graphify() for base_model in base_models]
        else:
            # no classifier head, lm head
            graphs = [Grapher(deepcopy(base_model), merge_type=merge_type, qk=args.qk, classifier=False).graphify() for base_model in base_models]

        bsz = experiment_config['dataset'][0]['batch_size']
        train_frac = experiment_config['dataset'][0]['train_fraction'] 
        model1 = raw_config['model_names']['model1'].replace('/', '_')
        model2 = raw_config['model_names']['model2'].replace('/', '_')

        Merge = ModelMerge(*graphs, device=device)

        unmerge, cost_dict = Merge.transform(
            deepcopy(config['models']['new']), 
            train_loaders, 
            sentence_level=args.sentence_level,
            special_toks=args.special_toks,
            transform_fn=get_merging_fn(merging_fn), 
            metric_classes=config['metric_fns'],
            permute_heads=args.permute_heads,
            ignore_heads=args.ignore_heads,
            save_both=args.save_both,
            merge_cls=args.merge_cls,
            no_absval=args.no_absval,
            saved_features=args.saved_feature_path,
            res_type=args.res_type,
        )

        if args.sentence_level:
            param_tail = f'b{bsz}_{args.task}{train_frac}_sent'
        else:
            param_tail = f'b{bsz}_{args.task}{train_frac}'
        
        param_tail+=f'_{args.seed0}_{args.seed1}'

        save_dir = os.path.join(save_dir_root, args.task, args.exp_name)
        if os.path.exists(save_dir) == False:
            os.makedirs(save_dir)
        if os.path.exists(os.path.join(save_dir, 'individual_models')) == False:
            os.makedirs(os.path.join(save_dir, 'individual_models'))
        

        with open(f'{save_dir}/test_{merging_fn}_{merge_type}_{model1}_{model2}_{param_tail}.args', 'w+') as f_args:
            f_args.write(str(vars(args)) + '\n')
            f_args.write(str(experiment_config))

        if args.save_both:
            torch.save(Merge.merged_model1.state_dict(), 
                   f'{save_dir}/test_{merging_fn}_{merge_type}_{model1}_{model2}_0_{param_tail}.pt')
            torch.save(Merge.merged_model2.state_dict(), 
                   f'{save_dir}/test_{merging_fn}_{merge_type}_{model1}_{model2}_1_{param_tail}.pt')
        else:
            torch.save(Merge.merged_model, 
                   f'{save_dir}/test_{merging_fn}_{merge_type}_{model1}_{model2}_{param_tail}.pt')
        torch.save(Merge.graphs[0].model.state_dict(), f'{save_dir}/individual_models/{merging_fn}_{merge_type}_0_{model1}_{param_tail}.pt')
        torch.save(Merge.graphs[1].model.state_dict(), f'{save_dir}/individual_models/{merging_fn}_{merge_type}_1_{model2}_{param_tail}.pt')

        if unmerge != None:
            if os.path.exists(os.path.join(save_dir, 'unmerge')) == False:
                os.makedirs(os.path.join(save_dir, 'unmerge'))
            torch.save(unmerge, f'{save_dir}/unmerge/unmerge_mat_{args.seed0}_{args.seed1}.pt')
        
        if os.path.exists(os.path.join(save_dir, 'costs')) == False:
            os.makedirs(os.path.join(save_dir, 'costs'))
        with open(f'{save_dir}/costs/costs_{args.seed0}_{args.seed1}.pt', 'w+') as costs_out:
            json.dump(cost_dict, costs_out)

        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",required=False,default='bert_books'
    )
    parser.add_argument(
        '--task',default='mlm'
    )
    parser.add_argument(
        "--seed0",default=0,type=int
    )
    parser.add_argument(
        "--seed1",default=1,type=int
    )
    parser.add_argument(
        "--train-frac",type=float, required=False
    )
    parser.add_argument(
        "--bsz",type=int, required=False,default=8
    )
    parser.add_argument(
        "--special-toks",required=False,action='store_true'
    )
    parser.add_argument(
        "--permute-heads",required=False,action='store_true'
    )
    parser.add_argument(
        "--ignore-heads",required=False,action='store_true'
    )
    parser.add_argument(
        "--exp-name",
    )
    parser.add_argument(
        "--merge-type", # one of ff_only, res_only, ff+res, ff+attn, attn_only, res+attn, all
    )
    parser.add_argument(
        '--qk',required=False,action='store_true'
    )
    parser.add_argument(
        '--save-both',action='store_true'
    )
    parser.add_argument(
        '--merging-fn',default='match_tensors_permute'
    )
    parser.add_argument(
        '--merge-cls',action='store_true'
    )
    parser.add_argument(
        '--sentence-level',required=False
    )
    parser.add_argument(
        '--no-absval',action='store_true'
    )
    parser.add_argument(
        '--res-type',required=False,default='first' # one of first, last, sep, all 
    )
    parser.add_argument(
        '--saved-feature-path',required=False
    )
    args = parser.parse_args()
    

    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
   
    raw_config = get_config_from_name(args.cfg, device=device)
    model_dir = raw_config['model']['dir']
    model_name = raw_config['model']['name']

    if args.task != 'mlm':
        raw_config['dataset'][0]['task'] = args.task
        raw_config['model_names']['model1'] = f'{args.task}/seed_{args.seed0}'
        raw_config['model_names']['model2'] = f'{args.task}/seed_{args.seed1}'
    else:
        raw_config['model_names']['model1'] = f'multiberts-seed_{args.seed0}'
        raw_config['model_names']['model2'] = f'multiberts-seed_{args.seed1}'


    if args.bsz:
        for i in range(len(raw_config['dataset'])):
            raw_config['dataset'][i]['batch_size'] = args.bsz 
    if args.train_frac:
        raw_config['dataset'][0]['train_fraction'] = args.train_frac
    

    run_pairs = [(raw_config['model_names']['model1'], raw_config['model_names']['model2'])]
    

    print(raw_config['model_names']['model1'])


    with torch.no_grad():
        node_results = run_auxiliary_experiment(
            merging_fn=args.merging_fn, 
            merge_type=args.merge_type,
            experiment_config=raw_config, 
            pairs=run_pairs, 
            device=device, 
            args=args
        )