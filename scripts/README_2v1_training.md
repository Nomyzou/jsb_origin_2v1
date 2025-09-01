# 2v1 场景训练指南

## 概述

本目录包含了用于2v1空战场景的训练脚本和配置文件。2v1场景包含2架红方飞机对抗1架蓝方飞机。

## 文件说明

### 训练脚本
- `train_share_2v1_selfplay.sh`: 基础2v1自博弈训练脚本
- `train_share_2v1_with_pretrained.sh`: 使用预训练敌方模型的2v1训练脚本

### 配置文件
- `train/train_2v1_jsbsim.py`: 专门的2v1训练脚本，支持加载预训练模型
- `envs/JSBSim/configs/2v1/NoWeapon/Selfplay.yaml`: 2v1场景配置文件

## 使用方法

### 1. 基础2v1自博弈训练

```bash
cd LAG/scripts
./train_share_2v1_selfplay.sh
```

这将启动一个2v1场景的训练，其中：
- 2架红方飞机（A0100, A0200）由智能体控制
- 1架蓝方飞机（B0100）由自博弈对手控制

### 2. 使用预训练敌方模型训练

```bash
cd LAG/scripts
./train_share_2v1_with_pretrained.sh
```

这将使用预训练的敌方模型进行训练。默认情况下，脚本会尝试加载：
`results/MultipleCombat/1v1/NoWeapon/Selfplay/mappo/1v1_selfplay/run1/models/iter_0_actor.pt`

### 3. 自定义预训练模型路径

您可以修改脚本中的 `opponent_model_path` 变量来使用不同的预训练模型：

```bash
# 在脚本中修改这一行
opponent_model_path="path/to/your/pretrained/model.pt"
```

## 配置说明

### 场景配置 (Selfplay.yaml)

- **任务类型**: `hierarchical_multiplecombat`
- **最大步数**: 1000步（约200秒）
- **飞机配置**:
  - A0100, A0200: 红方F16战斗机
  - B0100: 蓝方F16战斗机
- **初始位置**: 红方飞机在北方，蓝方飞机在南方

### 训练参数

- **算法**: MAPPO (Multi-Agent Proximal Policy Optimization)
- **自博弈算法**: FSP (Fictitious Self-Play)
- **对手数量**: 1个
- **初始ELO**: 1200

## 注意事项

1. **模型兼容性**: 确保预训练模型与当前训练配置兼容（相同的网络架构和输入输出维度）

2. **硬件要求**: 建议使用GPU进行训练，脚本默认使用CUDA_VISIBLE_DEVICES=0

3. **存储空间**: 训练结果将保存在 `results/` 目录下

4. **日志记录**: 训练过程会记录到控制台和wandb（如果启用）

## 故障排除

### 预训练模型加载失败

如果预训练模型加载失败，系统会自动回退到随机初始化的对手模型，并在日志中显示警告信息。

### 内存不足

如果遇到内存不足问题，可以：
- 减少 `n-rollout-threads` 参数
- 减少 `episode-length` 参数
- 使用CPU训练（移除 `--cuda` 参数）

## 扩展

要创建新的2v1场景配置，可以：
1. 复制现有的yaml配置文件
2. 修改飞机配置、初始位置等参数
3. 更新训练脚本中的 `scenario-name` 参数 