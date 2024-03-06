# Merging Text Transformer Models from Different Initializations

This repository contains the code for the paper _[Merging Text Transformer Models from Different Initializations](https://arxiv.org/pdf/2403.00986.pdf)_ by [Neha Verma](https://nverma1.github.io/) and [Maha Elbayad](https://elbayadm.github.io/). 

## Getting Started

### Dependencies

We recommend creating a new virtual environment, and installing the following dependencies:
 
```
pip install -r requirements.txt
```


### Data

Main masked language modeling experiments are run on a subset of the [BooksCorpus](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zhu_Aligning_Books_and_ICCV_2015_paper.pdf), as it was a subset of the original BERT training data. To extract a subset of the data, run the following command:

```
cd my_datasets
python sample_books_corpus.py $PATH_TO_BOOKS_CORPUS
```

## Experiments

### Obtaining Models  

We use several models from the [MultiBERTs](https://openreview.net/pdf?id=K0E_F0gFDgA) reproductions, accessible on [HuggingFace]((https://huggingface.co/) via the ``google/multiberts-seed_i`` paths, where ``i`` is the seed number. We also fine-tune these models on the GLUE benchmark for additional experiments, using the HuggingFace library. To fine-tune these models, run the following command:

```
bash training/finetune_glue.sh $seed_index $task
```
Where task is one of the GLUE tasks in ``[mnli, cola, sst2, stsb, qnli, qqp, rte, mrpc]``, and seed_index is the index of the seed in ``1-5``.

### Merging Models

To merge models, we have the following scripts to produce merged models from our paper:

```
# merge models by feed-forward and/or attention components
bash experiments/run_component_experiments.sh

# merge models using different attention merging algorithms
bash experiments/run_mha_experiments.sh

# merge models using different residual merging algorithms
bash experiments/run_res_experiments.sh

# merge models using different data amounts
bash experiments/run_data_ablation.sh

# merge glue models, trained in previous step
bash experiments/run_glue_experiments.sh

```

Finally, many other merging experiments can be run using ``experiment/merge_bert_classifiers.py``, adding potentially adding configs to the ``configs/`` directory.

### Evaluation

To evaluate glue models before merging, run the following command:
```
python evaluation/graphing_glue.py --task $TASK --originals --outfile vanilla_${TASK}
```
Where task is one of the GLUE tasks in ``[mnli, cola, sst2, stsb, qnli, qqp, rte, mrpc]``

To evaluate glue models after merging, run the folowing:
```
python evaluation/graphing_glue.py --task $TASK --path $PATH_TO_MERGED_MODELS --merge-type $MERGE_TYPE  --outfile $OUTFILE
```
Where task is one of the GLUE tasks in ``[mnli, cola, sst2, stsb, qnli, qqp, rte, mrpc]``, merge-type is one of the merge types in ``[res_only, attn_only, ff_only, ff+attn, res+attn, ff+res, all]``


To evaluate MLM models before merging, run the following command: 
```
python evaluation/graphing_mlm.py --originals --outfile vanilla_mlm
```
To evaluate MLM models after merging, run the following; 
```
python evaluation/graphing_mlm.py --path $PATH_TO_MERGED_MODELS --merge-type $MERGE_TYPE  --outfile $OUTFILE --train-frac $TRAIN_FRAC
```
where merge-type is one of the merge types in ``[res_only, attn_only, ff_only, ff+attn, res+attn, ff+res, all]`` and train-frac is the fraction of the training data used in the MLM experiments.
If a permutation was applied to the output projection, pass the relevant ``--unmerge`` flag to the command.

## Citation

If you use this code, please cite our paper:

```
@article{verma2024merging,
    title={Merging Text Transformer Models from Different Initializations},
    author={Neha Verma and Maha Elbayad},
    journal={arXiv},
    year={2024},
}
```

### Acknowledgements

We would like to acknowledge the authors of the [ZipIt!](https://github.com/gstoica27/ZipIt) codebase, which we use as a starting point for our repository. We also acknowledge the authors of the [BERT-similarity](https://github.com/twinkle0331/BERT-similarity), which we used to help with our GLUE fine-tuning code. 
