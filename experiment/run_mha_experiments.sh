#!/bin/bash

# this script is to run the experiments by multi-headed attention merger type. This corresponds to table 1 in our paper 

# all pairs
pairs_0=(1 2 3 4 1 2 3 1 2 1)
pairs_1=(2 3 4 5 3 4 5 4 5 5)

cd ../

train_frac=0.1
 
for i in 0 1 2 3 4 5 6 7 8 9; do 
    seed0=${pairs_0[$i]}
    seed1=${pairs_1[$i]}

    # identity MHA permutation = ff_only
    python -m experiment.merge_bert_classifiers --cfg bert_books --task mlm --seed0 $seed0 --seed1 $seed1 --train-frac $train_frac --bsz 8 --special-toks --no-absval --merge-type ff_only --merging-fn match_tensors_identity --exp-name perm_ff_only_frac${train_frac}

    # Monotonic head alignment = --permute-heads off
    python -m experiment.merge_bert_classifiers --cfg bert_books --task mlm --seed0 $seed0 --seed1 $seed1 --train-frac ${train_frac} --bsz 8 --special-toks --no-absval --merge-type ff+attn --merging-fn match_tensors_permute --exp-name perm_ff+attn_frac${train_frac}_noperm

    # Ignore Heads = --ignore-heads on 
    python -m experiment.merge_bert_classifiers --cfg bert_books --task mlm --seed0 $seed0 --seed1 $seed1 --train-frac ${train_frac} --bsz 8 --special-toks --ignore-heads --no-absval --merge-type ff+attn --merging-fn match_tensors_permute --exp-name perm_ff+attn_frac${train_frac}_ignoreperm

    # Permute Heads = --permute-heads on 
    python -m experiment.merge_bert_classifiers --cfg bert_books --task mlm --seed0 $seed0 --seed1 $seed1 --train-frac ${train_frac} --bsz 8 --special-toks --permute-heads --no-absval --merge-type ff+attn --merging-fn match_tensors_permute --exp-name perm_ff+attn_frac${train_frac}
done
