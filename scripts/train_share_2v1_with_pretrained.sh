#!/bin/bash

# 2v1场景训练脚本 - 使用预训练的敌方模型

# 环境配置
env="MultipleCombat"
scenario="2v1/NoWeapon/Selfplay"
algo="mappo"
exp="2v1_pretrained_opponent"
seed=1

# 预训练敌方模型路径
opponent_model_path="results/MultipleCombat/1v1/NoWeapon/Selfplay/mappo/1v1_selfplay/run1/models/iter_0_actor.pt"

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
echo "Using pre-trained opponent model: ${opponent_model_path}"

CUDA_VISIBLE_DEVICES=0 python train/train_2v1_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --seed ${seed} --n-training-threads 1 --n-rollout-threads 32 --cuda --log-interval 1 --save-interval 1 \
    --num-mini-batch 1 --episode-length 1000 --num-env-steps 20000000 --ppo-epoch 10 --seed 1 \
    --use-selfplay --selfplay-algorithm "fsp" --n-choose-opponents 1 --init-elo 1200 \
    --opponent-model-path ${opponent_model_path} 