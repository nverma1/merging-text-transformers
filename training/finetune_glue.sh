#!/bin/bash

index=$1

task=$2

if [[ $task == "mrpc" ]]; then
    epoch=5
else
    epoch=3
fi

python3 run_glue.py --model_name_or_path google/multiberts-seed_${index} \
                   --task_name $task \
                   --do_train \
                   --do_eval \
                   --learning_rate 2e-5 \
                   --max_seq_length 128 \
                   --num_train_epochs $epoch \
                   --output_dir models/trained/multiberts/$task/seed_${index} \
                   --load_best_model_at_end \
                   --save_total_limit 1 \
                   --save_strategy "no" \
                   --fp16 > models/trained/multiberts/log/$task/${task}_${index}.txt 2>&1
