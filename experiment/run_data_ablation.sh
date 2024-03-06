#!/bin/bash

# this script is to run the data ablation experiments. It runs through all 10 pairs, and all levels of data fractions we consider. This corresponds to figure 6 in our paper

# all pairs
pairs_0=(1 2 3 4 1 2 3 1 2 1)
pairs_1=(2 3 4 5 3 4 5 4 5 5)

cd ../
 
# data fractions
for train_frac in 0.001 0.005 0.01 0.05 0.1 0.5 1.0; do 
    # pairs
    for i in 0 1 2 3 4 5 6 7 8 9; do
        seed0=${pairs_0[$i]}
        seed1=${pairs_1[$i]}
        python -m experiment.merge_bert_classifiers --cfg bert_books --task mlm --seed0 $seed0 --seed1 $seed1 --train-frac ${train_frac} --bsz 8 --special-toks --permute-heads --no-absval --merge-type ff+attn --merging-fn match_tensors_permute --exp-name perm_ff+attn_frac${train_frac}_recheck 
    done
done