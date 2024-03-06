#!/bin/bash

# this script is to run the glue experiments. This requires fine-tuning the glue models first. This corresponds to table 3 in our paper 

# all pairs
pairs_0=(1 2 3 4 1 2 3 1 2 1)
pairs_1=(2 3 4 5 3 4 5 4 5 5)

cd ../

train_frac=1
type=all


# by residual
for task in mnli qqp qnli sst2 cola stsb mrpc rte; do 
    for i in 0 1 2 3 4 5 6 7 8 9; do 
        seed0=${pairs_0[$i]}
        seed1=${pairs_1[$i]}

        # small data because this is just vanilla merging
        python -m experiment.merge_bert_classifiers --cfg bert_classifier --task $task --seed0 $seed0 --seed1 $seed1 --train-frac 0.001 --bsz 8 --special-toks --permute-heads --no-absval --merge-type $type --merging-fn match_tensors_identity --exp-name ${task}_vanilla_merge

        python -m experiment.merge_bert_classifiers --cfg bert_classifier --task $task --seed0 $seed0 --seed1 $seed1 --train-frac ${train_frac} --bsz 8 --special-toks --no-absval --permute-heads --exp-name  ${task}_permute_${type}_frac${train_frac} --merge-type $type --merging-fn match_tensors_permute
    done
done
