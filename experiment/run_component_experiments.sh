#!/bin/bash

# this script is to run the experiments by components merged. This corresponds to figure 3 and figure 4 in our paper. 

# all pairs
pairs_0=(1 2 3 4 1 2 3 1 2 1)
pairs_1=(2 3 4 5 3 4 5 4 5 5)

cd ../
 

# identity permutation = vanilla merging.
# we use a very small train fraction to speed up the experiments as no data is needed here
for i in 0 1 2 3 4 5 6 7 8 9; do
    seed0=${pairs_0[$i]}
    seed1=${pairs_1[$i]}
    python -m experiment.merge_bert_classifiers --cfg bert_books --task mlm --seed0 $seed0 --seed1 $seed1 --train-frac 0.0001 --bsz 8 --special-toks --permute-heads --no-absval --merge-type ff+attn --merging-fn match_tensors_identity --exp-name vanilla_merge
done

# by component
train_frac=0.1
for type in ff_only attn_only ff+attn; do 
    # pairs
    for i in 0 1 2 3 4 5 6 7 8 9; do
        seed0=${pairs_0[$i]}
        seed1=${pairs_1[$i]}
        python -m experiment.merge_bert_classifiers --cfg bert_books --task mlm --seed0 $seed0 --seed1 $seed1 --train-frac ${train_frac} --bsz 8 --special-toks --permute-heads --no-absval --merge-type $type --merging-fn match_tensors_permute --exp-name perm_${type}_frac${train_frac}
    done
done
