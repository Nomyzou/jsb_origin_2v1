#!/bin/sh

# 默认参数
env="SingleCombat"
scenario="1v1/ShootMissile/HierarchySelfplay"
algo="ppo"
exp="v1"
seed=1
gpu_id=0
model_dir="./results/SingleCombat/1v1/ShootMissile/HierarchySelfplay/ppo/v1/wandb/latest-run/files"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            env="$2"
            shift 2
            ;;
        --scenario)
            scenario="$2"
            shift 2
            ;;
        --algo)
            algo="$2"
            shift 2
            ;;
        --exp)
            exp="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --gpu)
            gpu_id="$2"
            shift 2
            ;;
        --model-dir)
            model_dir="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --env ENV_NAME        Environment name (default: SingleCombat)"
            echo "  --scenario SCENARIO   Scenario name (default: 1v1/ShootMissile/HierarchySelfplay)"
            echo "  --algo ALGO           Algorithm name (default: ppo)"
            echo "  --exp EXP             Experiment name (default: v1)"
            echo "  --seed SEED           Random seed (default: 1)"
            echo "  --gpu GPU_ID          GPU ID (default: 0)"
            echo "  --model-dir DIR       Model directory path"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
echo "Using GPU: ${gpu_id}, Model directory: ${model_dir}"

CUDA_VISIBLE_DEVICES=${gpu_id} python render/render_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --seed ${seed} --n-training-threads 1 --n-rollout-threads 4 --cuda \
    --log-interval 1 --save-interval 1 \
    --num-mini-batch 5 --buffer-size 3000 --num-env-steps 1e8 \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-params 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
    --hidden-size "128 128" --act-hidden-size "128 128" --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8 \
    --model-dir "${model_dir}" 