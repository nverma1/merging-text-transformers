#!/bin/bash

# this script is to run the experiments by residual merger type. This corresponds to table 2 in our paper 

# all pairs
pairs_0=(1 2 3 4 1 2 3 1 2 1)
pairs_1=(2 3 4 5 3 4 5 4 5 5)

cd ../

train_frac=0.1

# identity residual permutation = vanilla merging. 
for i in 0 1 2 3 4 5 6 7 8 9; do 
    seed0=${pairs_0[$i]}
    seed1=${pairs_1[$i]}
    python -m language_evaluation_scripts.merge_bert_classifiers --cfg bert_books --task mlm --seed0 $seed0 --seed1 $seed1 --train-frac 0.0001 --bsz 8 --special-toks --permute-heads --no-absval --merge-type res_only --merging-fn match_tensors_identity --exp-name vanilla_merge
done

# by residual
for res_type in first last all sep; do 
    for i in 0 1 2 3 4 5 6 7 8 9; do 
        seed0=${pairs_0[$i]}
        seed1=${pairs_1[$i]}

        python -m language_evaluation_scripts.merge_bert_classifiers --cfg bert_books --task mlm --seed0 $seed0 --seed1 $seed1 --train-frac ${train_frac} --bsz 8 --special-toks --no-absval --res-type ${res_type} --permute-heads --exp-name  res_only_frac${train_frac}_res_${res_type} --merge-type res_only --merging-fn match_tensors_permute
    done
done
