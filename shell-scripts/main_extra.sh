#!/bin/bash

## data
data_cais="train_cais.json"
data_ecdt="train_ecdt.json"
data_augment="train.json train_augment.json" # not aug or aug

## pre-training models
pre_training_models=$1
# hfl/chinese-macbert-base hfl/chinese-lert-base -------> undone
# hfl/chinese-pert-base hfl/chinese-bert-wwm-ext hfl/chinese-roberta-wwm-ext


## model architectures
model_architectures="2" # 0-2
# 0 --> SLUTaggingBERT
# 1 --> SLUTaggingBERTCascaded
# 2 --> SLUTaggingBERTMultiHead
# -1 == > BiLSTM 

## hyperparameters
lrs=`seq 1e-5 2e-5 5e-5`

bsz=320
max_epoch=100
dropout=0.2

use_asr="1 0"
train_mix="1 0"

device=$2


for model in $model_architectures; do
    for use_aug in $data_augment; do
        for asr in $use_asr; do
            for mix in $train_mix; do
                for lr in $lrs; do
                    python scripts/slu_bert.py --train_path $use_aug \
                        --train_path_cais $data_cais --train_path_ecdt $data_ecdt \
                        --device $device --lr $lr --max_epoch $max_epoch \
                        --batch_size $bsz --dropout $dropout --architecture ${{model}} \
                        --model_name ${pre_training_models} --use_asr ${{asr}} --train_mix ${{mix}}
                done
            done
        done
    done
done

grep 'FINAL BEST RESUL' exp/* | cut -d':'  -f1,6 | cut -f1 > results.log
cat results.log | sort -t: -k2 -n | tail -1