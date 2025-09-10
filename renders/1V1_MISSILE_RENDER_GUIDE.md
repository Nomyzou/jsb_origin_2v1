# 1v1导弹对战渲染指南

## 概述

本指南介绍如何使用训练好的导弹对战模型进行渲染，生成可视化的空战轨迹。

## 文件结构

```
scripts/render/
├── render_1v1_missile.py      # 主要的渲染脚本
├── render_1v1_missile.sh      # 简化的shell脚本
└── render_jsbsim.py           # 通用渲染脚本

scripts/results/SingleCombat/1v1/ShootMissile/HierarchySelfplay/ppo/v1/wandb/latest-run/files/
├── actor_96.pt                # 训练好的actor模型
└── critic_latest.pt          # critic模型
```

## 使用方法

### 方法1: 使用Shell脚本（推荐）

```bash
# 激活conda环境
conda activate jsbsim1

# 运行渲染脚本
./scripts/render/render_1v1_missile.sh
```

### 方法2: 直接使用Python脚本

```bash
# 激活conda环境
conda activate jsbsim1

# 运行渲染脚本
python scripts/render/render_1v1_missile.py \
    --model-path "scripts/results/SingleCombat/1v1/ShootMissile/HierarchySelfplay/ppo/v1/wandb/latest-run/files/actor_96.pt" \
    --scenario-name "1v1/ShootMissile/HierarchySelfplay" \
    --render-episodes 3 \
    --episode-length 1000 \
    --output-dir "renders/missile_combat" \
    --num-agents 2
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-path` | `scripts/results/.../actor_96.pt` | 训练好的模型文件路径 |
| `--scenario-name` | `1v1/ShootMissile/HierarchySelfplay` | 场景名称 |
| `--render-episodes` | `5` | 渲染的episode数量 |
| `--episode-length` | `1000` | 每个episode的最大步数 |
| `--output-dir` | `renders/missile_combat` | 输出目录 |
| `--num-agents` | `2` | 智能体数量 |

## 输出文件

渲染完成后，会在输出目录中生成以下文件：

```
renders/missile_combat_YYYYMMDD_HHMMSS/
├── episode_1/
│   ├── JSBSimRecording.txt.acmi    # TacView格式的轨迹文件
│   └── episode_info.txt           # Episode信息
├── episode_2/
│   ├── JSBSimRecording.txt.acmi
│   └── episode_info.txt
└── ...
```

## 查看渲染结果

### 使用TacView（推荐）

1. 下载并安装 [TacView](https://www.tacview.net/)
2. 打开生成的 `.txt.acmi` 文件
3. 可以看到3D空战轨迹，包括：
   - 飞机运动轨迹
   - 导弹发射和飞行轨迹
   - 爆炸效果
   - 实时状态信息

### 使用命令行分析

```bash
# 查看导弹轨迹
grep "^M" renders/missile_combat_*/episode_*/JSBSimRecording.txt.acmi

# 查看爆炸效果
grep "Explosion" renders/missile_combat_*/episode_*/JSBSimRecording.txt.acmi

# 统计导弹数量
grep "^M" renders/missile_combat_*/episode_*/JSBSimRecording.txt.acmi | wc -l
```

## 模型说明

### 训练环境
- **环境**: SingleCombat
- **场景**: 1v1/ShootMissile/HierarchySelfplay
- **算法**: PPO (Proximal Policy Optimization)
- **自对弈**: 使用FSP (Fictitious Self-Play)算法

### 模型特点
- 支持导弹发射和制导
- 使用比例导引法进行导弹制导
- 包含完整的空战策略

## 故障排除

### 常见问题

1. **模型文件不存在**
   ```
   错误: 模型文件不存在: scripts/results/.../actor_96.pt
   ```
   **解决方案**: 检查模型路径是否正确，确保训练已完成

2. **CUDA内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   **解决方案**: 设置 `export CUDA_VISIBLE_DEVICES=0` 或使用CPU模式

3. **环境初始化失败**
   ```
   Can not support the environment
   ```
   **解决方案**: 确保conda环境已激活，检查依赖包是否正确安装

### 调试模式

启用详细日志：
```bash
python scripts/render/render_1v1_missile.py --debug
```

## 自定义配置

### 修改渲染参数

编辑 `render_1v1_missile.sh` 文件：
```bash
# 修改渲染episodes数量
EPISODES=10

# 修改episode长度
EPISODE_LENGTH=2000

# 修改输出目录
OUTPUT_DIR="my_custom_output"
```

### 使用不同的模型

```bash
python scripts/render/render_1v1_missile.py \
    --model-path "path/to/your/model.pt" \
    --render-episodes 1
```

## 性能优化

1. **GPU加速**: 确保CUDA可用，设置正确的GPU设备
2. **内存管理**: 对于长episode，考虑减少batch size
3. **并行渲染**: 可以同时运行多个渲染进程

## 扩展功能

### 批量渲染

```bash
# 渲染多个模型
for model in models/*.pt; do
    python scripts/render/render_1v1_missile.py \
        --model-path "$model" \
        --output-dir "renders/$(basename $model .pt)"
done
```

### 自动分析

```python
# 分析渲染结果
import os
import glob

def analyze_renders(output_dir):
    acmi_files = glob.glob(f"{output_dir}/**/*.acmi", recursive=True)
    for file in acmi_files:
        # 分析导弹命中率、飞行时间等
        pass
```

这个渲染系统为分析训练好的导弹对战模型提供了完整的可视化解决方案。 