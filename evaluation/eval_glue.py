#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import argparse

import numpy as np
from datasets import load_dataset, load_metric

from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    EvalPrediction,
    Trainer,
    default_data_collator,
)
import torch

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
}

task_to_outputs = {
    'cola': 2,
    'mnli': 3,
    'mrpc': 2,
    'qnli': 2,
    'qqp': 2,
    'rte': 2,
    'sst2': 2,
    'stsb': 1,
}

def get_metrics(task, model, cache_dir, log=False):

    tokenizer = BertTokenizer.from_pretrained(
        'google/multiberts-seed_0',
        cache_dir=cache_dir,
        use_fast=True,
    )

    # Preprocessing the raw_datasets
    if task == 'mnli':
        validation_name = 'validation_matched'
    else:
        validation_name = 'validation'

    raw_datasets = load_dataset("glue", task, cache_dir=cache_dir, split=validation_name)
    is_regression = (task == "stsb")

    sentence1_key, sentence2_key = task_to_keys[task]

    # Padding strategy
    max_seq_length=128
    padding = 'max_length'

    if max_seq_length > tokenizer.model_max_length:
        print(
            f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        return result

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on dataset",
    )

    eval_dataset = raw_datasets

    # Get the metric function
    metric = load_metric("glue", task)

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        # print(Counter(preds))
        # print(Counter(p.label_ids))
        if task is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if padding == 'max_length':
        data_collator = default_data_collator

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        train_dataset=None,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Evaluation
    print('evaluate')

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [task]
    eval_datasets = [eval_dataset]
    if task == "mnli":
        tasks.append("mnli-mm")
        mismatched_eval =  load_dataset("glue", task, cache_dir=cache_dir, split="validation_mismatched")
        eval_datasets.append(mismatched_eval.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on dataset",
        ))

    metric_list = []
    for eval_dataset, task in zip(eval_datasets, tasks):
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        metrics["eval_samples"] = len(eval_dataset)
        if log:
            trainer.log_metrics("eval", metrics)
        metric_list.append(metrics)
        #trainer.save_metrics("eval", metrics)
    
    return metric_list

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--task',required=True)
    parser.add_argument('--hf-model')
    parser.add_argument('--merged-model')
    parser.add_argument('--merged-model-dict')
    parser.add_argument('--cache-dir', default='/checkpoint/nverma7/cache/')
    parser.add_argument('--tokenizer-name', default='google/multiberts-seed_0')
    parser.add_argument('--loss', action='store_true', required=False)

    args = parser.parse_args()


    
    is_regression = (args.task == "stsb")
    if not is_regression:
        num_labels = task_to_outputs[args.task]
    else:
        num_labels = 1

    # load tokenizer and model
    
    if args.hf_model:
        model = BertForSequenceClassification.from_pretrained(
            args.hf_model,
            cache_dir=args.cache_dir,
        )
    elif args.merged_model:
        model = torch.load(args.merged_model)
    elif args.merged_model_dict:
        model_dict = torch.load(args.merged_model_dict)
        model = BertForSequenceClassification.from_pretrained('google/multiberts-seed_0', num_labels=num_labels)
        model.load_state_dict(model_dict)
        
    
    metrics = get_metrics(args.task, model, args.cache_dir, log=True)



if __name__ == "__main__":
    main()

