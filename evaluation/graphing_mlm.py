import os 
import json
import argparse

import math
import numpy as np
import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from copy import deepcopy
from datasets import Dataset
from transformers import BertTokenizer, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling, Trainer


pairs_0 = [1,2,3,4,1,2,3,1,2,1]
pairs_1 = [2,3,4,5,3,4,5,4,5,5]

lambdas = np.arange(21) / 20


class PermAllTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.out_proj = kwargs.pop('out_proj')
        super().__init__(*args, **kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False):

        # this circumvents issues with tied weights via saving out_proj
        def mini_forward(data):
            x = model(**data, output_hidden_states=True)['hidden_states'][-1]
            x =model.cls.predictions.transform.dense(x)
            x =model.cls.predictions.transform.transform_act_fn(x)
            x =model.cls.predictions.transform.LayerNorm(x)
            device =model.cls.predictions.decoder.bias.device
            x = F.linear(x, self.out_proj.to(device)) +model.cls.predictions.decoder.bias
            return x
    
        # forward pass
        logits = mini_forward(inputs)
        labels = inputs.get('labels')
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss()
        mlm_loss = loss_fct(logits.view(-1, model.config.vocab_size), labels.view(-1))
        return (mlm_loss, logits) if return_outputs else mlm_loss

def wikitext_ppl_all(tokenizer, model, out_proj):
    
    wikitext = datasets.load_dataset('wikitext','wikitext-103-raw-v1', split='validation')
    block_size=128

    def preprocess_function(examples):
        return tokenizer([" ".join(x) for x in examples["text"]])
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    tokenized_wikitext = wikitext.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=wikitext.column_names,
    )

    lm_dataset = tokenized_wikitext.map(group_texts, batched=True, num_proc=4)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    trainer=PermAllTrainer(
        model=model,
        eval_dataset=lm_dataset,
        data_collator=data_collator,
        out_proj=out_proj
    )
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    return math.exp(eval_results['eval_loss'])


def wikitext_ppl(tokenizer, model):
    
    wikitext = datasets.load_dataset('wikitext','wikitext-103-raw-v1', split='validation')
    block_size=128

    def preprocess_function(examples):
        return tokenizer([" ".join(x) for x in examples["text"]])
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    tokenized_wikitext = wikitext.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=wikitext.column_names,
    )

    lm_dataset = tokenized_wikitext.map(group_texts, batched=True, num_proc=4)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    trainer=Trainer(
        model=model,
        eval_dataset=lm_dataset,
        data_collator=data_collator
    )
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    return math.exp(eval_results['eval_loss'])



def get_merged_state_dict(state_dict_1, state_dict_2, w=0.5):
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
            # Only catch runtime errors about tensor sizes, we need to be able to add models with diff heads together
            if 'size' not in str(e):
                raise e
        return state_dict


def main():

    vanilla_lambda_lists = {}

    parser = argparse.ArgumentParser()
    parser.add_argument('--originals', action='store_true', required=False, default=False)
    parser.add_argument('--path')
    parser.add_argument('--merge-type')
    parser.add_argument('--outfile')
    parser.add_argument('--train-frac', default=0.1, type=float)
    parser.add_argument('--dataset', default='wikitext', type=str)
    parser.add_argument('--unmerge', action='store_true', required=False, default=False)

    args = parser.parse_args()

    ROOT_PATH = f'models/'
    placeholder_tokenizer = BertTokenizer.from_pretrained('google/multiberts-seed_0')
    placeholder_model = BertForMaskedLM.from_pretrained('google/multiberts-seed_0')

    train_frac = args.train_frac
    for i in range(10):
        if args.originals == True:
            modeldict1 = BertForMaskedLM.from_pretrained(f'google/multiberts-seed_{pairs_0[i]}').state_dict()
            modeldict2 = BertForMaskedLM.from_pretrained(f'google/multiberts-seed_{pairs_1[i]}').state_dict()
        else:
            new_path = os.path.join(ROOT_PATH, args.path, 'individual_models')

            model_file1 = f'match_tensors_permute_{args.merge_type}_0_multiberts-seed_{pairs_0[i]}_b8_mlm{train_frac}_{pairs_0[i]}_{pairs_1[i]}.pt'
            model_file2 = f'match_tensors_permute_{args.merge_type}_1_multiberts-seed_{pairs_1[i]}_b8_mlm{train_frac}_{pairs_0[i]}_{pairs_1[i]}.pt'
            model_path1 = os.path.join(new_path, model_file1)
            model_path2 = os.path.join(new_path, model_file2)
            modeldict1 = torch.load(model_path1)
            modeldict2 = torch.load(model_path2)
            if args.unmerge == True:
                unmerge_file = os.path.join(ROOT_PATH, args.path, 'unmerge', f'unmerge_mat_{pairs_0[i]}_{pairs_1[i]}.pt')
                unmerge_mat = torch.load(unmerge_file)
                
        
        vanilla_lambda_lists[i] = {}
        vanilla_lambda_lists[i][0] = []

        for l in tqdm(lambdas):
            merged_statedict = get_merged_state_dict(modeldict1, modeldict2, w=l)
            placeholder_model.load_state_dict(merged_statedict)
            if args.dataset == 'wikitext':
                if args.unmerge is True:
                    new_proj = l * modeldict1['cls.predictions.decoder.weight'] + (1-l) * modeldict2['cls.predictions.decoder.weight'] @ unmerge_mat
                    vanilla_lambda_lists[i][0].append(wikitext_ppl_all(placeholder_tokenizer, placeholder_model, new_proj))
                else:
                    vanilla_lambda_lists[i][0].append(wikitext_ppl(placeholder_tokenizer, placeholder_model))

    if os.path.exists('results/mlm') == False:
        os.makedirs('results/mlm')

    with open(f'results/mlm/{args.dataset}/{args.outfile}', 'w+') as out:
        json.dump(vanilla_lambda_lists, out)
    
            
if __name__ == "__main__":
    main()

