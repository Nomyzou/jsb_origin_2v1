# 1v1导弹对战渲染系统 - 完整指南

## 🎯 项目概述

基于训练好的导弹对战模型，创建了完整的渲染系统，可以生成可视化的空战轨迹，包括飞机运动、导弹发射、制导和爆炸效果。

## 📁 文件结构

```
scripts/render/
├── render_1v1_missile.py          # 主要渲染脚本
├── render_1v1_missile.sh          # 简化的shell脚本
├── test_render_setup.py           # 环境测试脚本
└── render_jsbsim.py               # 通用渲染脚本

scripts/results/SingleCombat/1v1/ShootMissile/HierarchySelfplay/ppo/v1/wandb/latest-run/files/
├── actor_96.pt                    # 训练好的actor模型
└── critic_latest.pt              # critic模型

renders/
├── 1V1_MISSILE_RENDER_GUIDE.md   # 详细使用指南
├── MISSILE_RENDER_ANALYSIS.md     # 导弹渲染原理分析
└── missile_combat_*/              # 渲染输出目录
```

## 🚀 快速开始

### 1. 环境检查
```bash
# 激活conda环境
conda activate jsbsim1

# 运行环境测试
python scripts/render/test_render_setup.py
```

### 2. 开始渲染
```bash
# 方法1: 使用shell脚本（推荐）
./scripts/render/render_1v1_missile.sh

# 方法2: 直接使用Python脚本
python scripts/render/render_1v1_missile.py --render-episodes 3
```

## ⚙️ 配置参数

### 核心参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-path` | `scripts/results/.../actor_96.pt` | 训练好的模型路径 |
| `--scenario-name` | `1v1/ShootMissile/HierarchySelfplay` | 场景名称 |
| `--render-episodes` | `5` | 渲染的episode数量 |
| `--episode-length` | `1000` | 每个episode的最大步数 |
| `--output-dir` | `renders/missile_combat` | 输出目录 |

### 高级参数
```bash
# 自定义渲染参数
python scripts/render/render_1v1_missile.py \
    --model-path "path/to/your/model.pt" \
    --render-episodes 10 \
    --episode-length 2000 \
    --output-dir "my_custom_output" \
    --num-agents 2
```

## 📊 输出结果

### 文件结构
```
renders/missile_combat_YYYYMMDD_HHMMSS/
├── episode_1/
│   ├── JSBSimRecording.txt.acmi    # TacView格式轨迹文件
│   └── episode_info.txt           # Episode信息
├── episode_2/
│   ├── JSBSimRecording.txt.acmi
│   └── episode_info.txt
└── ...
```

### ACMI文件格式
```
FileType=text/acmi/tacview
FileVersion=2.1
0,ReferenceTime=2020-04-01T00:00:00Z
#时间戳
A0100,T=lon|lat|alt|roll|pitch|yaw,Name=F16,Color=Red      # 飞机A
B0100,T=lon|lat|alt|roll|pitch|yaw,Name=F16,Color=Blue     # 飞机B
M01001,T=lon|lat|alt|roll|pitch|yaw,Name=AIM-9L,Color=Red  # 导弹
```

## 🎮 查看结果

### 使用TacView（推荐）
1. 下载 [TacView](https://www.tacview.net/)
2. 打开 `.txt.acmi` 文件
3. 查看3D空战轨迹

### 命令行分析
```bash
# 查看导弹轨迹
grep "^M" renders/missile_combat_*/episode_*/JSBSimRecording.txt.acmi

# 查看爆炸效果
grep "Explosion" renders/missile_combat_*/episode_*/JSBSimRecording.txt.acmi

# 统计导弹数量
grep "^M" renders/missile_combat_*/episode_*/JSBSimRecording.txt.acmi | wc -l
```

## 🔧 技术细节

### 导弹渲染机制
1. **创建阶段**: 通过 `MissileSimulator.create()` 创建导弹
2. **飞行阶段**: 输出标准状态信息
3. **爆炸阶段**: 输出爆炸效果和移除导弹
4. **未命中阶段**: 仅移除导弹

### 模型特点
- **算法**: PPO (Proximal Policy Optimization)
- **自对弈**: FSP (Fictitious Self-Play)
- **制导**: 比例导引法
- **导弹**: AIM-9L型号

### 物理参数
```python
# AIM-9L导弹参数
t_max = 60        # 最大飞行时间 (秒)
t_thrust = 3     # 发动机工作时间 (秒)
Rc = 300         # 爆炸半径 (米)
v_min = 150      # 最小速度 (m/s)
K = 3            # 比例导引系数
```

## 🛠️ 故障排除

### 常见问题

1. **模型文件不存在**
   ```bash
   # 检查模型路径
   ls scripts/results/SingleCombat/1v1/ShootMissile/HierarchySelfplay/ppo/v1/wandb/latest-run/files/
   ```

2. **CUDA内存不足**
   ```bash
   # 使用CPU模式
   export CUDA_VISIBLE_DEVICES=""
   python scripts/render/render_1v1_missile.py
   ```

3. **环境初始化失败**
   ```bash
   # 检查conda环境
   conda activate jsbsim1
   python scripts/render/test_render_setup.py
   ```

### 调试模式
```bash
# 启用详细日志
python scripts/render/render_1v1_missile.py --debug
```

## 📈 性能优化

### GPU加速
```bash
# 设置GPU设备
export CUDA_VISIBLE_DEVICES=0
```

### 内存管理
```bash
# 减少episode长度
python scripts/render/render_1v1_missile.py --episode-length 500
```

### 批量渲染
```bash
# 渲染多个episode
python scripts/render/render_1v1_missile.py --render-episodes 10
```

## 🔄 扩展功能

### 自定义模型
```bash
# 使用不同的模型
python scripts/render/render_1v1_missile.py \
    --model-path "path/to/your/model.pt"
```

### 批量处理
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
import glob
import os

def analyze_renders(output_dir):
    acmi_files = glob.glob(f"{output_dir}/**/*.acmi", recursive=True)
    for file in acmi_files:
        # 分析导弹命中率、飞行时间等
        pass
```

## 📚 相关文档

- [导弹渲染原理分析](renders/MISSILE_RENDER_ANALYSIS.md)
- [详细使用指南](renders/1V1_MISSILE_RENDER_GUIDE.md)
- [训练脚本说明](scripts/train_selfplay_shoot.sh)

## 🎯 总结

这个渲染系统为分析训练好的导弹对战模型提供了完整的可视化解决方案，包括：

1. **完整的渲染流程**: 从模型加载到轨迹输出
2. **多种查看方式**: TacView 3D可视化 + 命令行分析
3. **灵活的配置**: 支持自定义参数和批量处理
4. **详细的文档**: 包含使用指南和技术分析
5. **故障排除**: 提供常见问题的解决方案

通过这个系统，可以直观地观察和分析训练好的导弹对战模型的性能和行为。 