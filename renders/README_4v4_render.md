# 4v4 空战场景渲染脚本使用指南

## 概述

本目录包含了用于4v4空战场景的渲染脚本，支持8架飞机（4架红方 vs 4架蓝方）的对抗仿真。

## 文件说明

### 配置文件
- `envs/JSBSim/configs/4v4/NoWeapon/Selfplay.yaml`: 标准4v4场景配置文件（51维观察空间）
- `envs/JSBSim/configs/4v4/NoWeapon/1v1Style.yaml`: 1v1风格4v4场景配置文件（15维观察空间）

### 渲染脚本
- `renders/render_4v4.py`: 基础4v4渲染脚本
- `renders/render_4v4_improved.py`: 改进版4v4渲染脚本（推荐使用）
- `renders/render_4v4_1v1_style.py`: 1v1风格4v4渲染脚本（使用1v1模型）

### 测试脚本
- `renders/test_4v4_env.py`: 标准4v4环境测试脚本
- `renders/test_4v4_1v1_style_env.py`: 1v1风格4v4环境测试脚本

## 4v4场景特点

### 飞机配置
- **红方飞机**: A0100, A0200, A0300, A0400
- **蓝方飞机**: B0100, B0200, B0300, B0400
- **初始位置**: 红方在北方，蓝方在南方
- **初始高度**: 20000英尺
- **初始速度**: 800英尺/秒

### 观察空间类型

#### 1. 标准4v4观察空间（51维）
- **观察维度**: 51维 (9 + (8-1) × 6)
- **自身信息**: 9维（高度、姿态、速度等）
- **相对信息**: 42维（与其他7架飞机的相对位置、角度等）

#### 2. 1v1风格观察空间（15维）
- **观察维度**: 15维 (9 + 6)
- **自身信息**: 9维（高度、姿态、速度等）
- **相对信息**: 6维（与对应编号敌机的相对位置、角度等）
- **对应关系**: A0100↔B0100, A0200↔B0200, A0300↔B0300, A0400↔B0400

### 动作空间
- **动作维度**: 4维离散动作
- **控制面**: 副翼、升降舵、方向舵、油门

## 使用方法

### 1. 标准4v4渲染

```bash
cd LAG
python renders/render_4v4_improved.py
```

### 2. 1v1风格4v4渲染（推荐用于1v1模型）

```bash
cd LAG
python renders/render_4v4_1v1_style.py
```

### 3. 环境测试

```bash
# 测试标准4v4环境
python renders/test_4v4_env.py

# 测试1v1风格4v4环境
python renders/test_4v4_1v1_style_env.py
```

### 4. 自定义配置

修改脚本中的以下参数：

```python
# 模型路径（1v1风格）
ego_run_dir = "scripts/results/SingleCombat/1v1/NoWeapon/HierarchySelfplay/ppo/v1/wandb/latest-run/files"

# 模型索引
ego_policy_index = "latest"  # 使用latest模型

# 渲染设置
render = True  # 是否生成TacView文件
```

## 输出文件

### TacView文件
- **格式**: `.txt.acmi`
- **内容**: 完整的空战轨迹记录
- **用途**: 可在TacView软件中可视化分析

### 控制台输出
- **血量信息**: 每50步显示一次
- **奖励信息**: 实时显示红方团队奖励
- **统计信息**: 最终奖励和平均奖励

## 注意事项

### 1. 模型兼容性
- 确保模型与4v4环境兼容（51维观察空间）
- 检查模型是否支持8架飞机的配置

### 2. 硬件要求
- **GPU**: 推荐使用CUDA加速
- **内存**: 建议8GB以上内存
- **存储**: 确保有足够空间保存TacView文件

### 3. 路径配置
- 修改脚本中的模型路径为实际路径
- 确保配置文件路径正确

### 4. 性能优化
- 使用`deterministic=True`确保可重现性
- 调整日志输出频率以平衡性能和监控需求

## 故障排除

### 1. 环境创建失败
```bash
# 检查配置文件是否存在
ls envs/JSBSim/configs/4v4/NoWeapon/Selfplay.yaml
```

### 2. 模型加载失败
```bash
# 检查模型文件是否存在
ls path/to/your/model/actor_1040.pt
```

### 3. 观察维度不匹配
- 确认环境配置为4v4（8架飞机）
- 检查模型是否针对4v4训练

### 4. 内存不足
- 减少批处理大小
- 使用CPU模式而非GPU模式

## 扩展功能

### 1. 多模型对比
可以修改脚本支持加载不同的模型进行对比：

```python
# 加载多个模型
models = {
    'model_v1': load_model('path/to/model_v1'),
    'model_v2': load_model('path/to/model_v2')
}
```

### 2. 统计分析
添加更详细的统计信息：

```python
# 统计信息
stats = {
    'win_rate': calculate_win_rate(),
    'average_duration': calculate_average_duration(),
    'damage_distribution': calculate_damage_distribution()
}
```

### 3. 实时监控
集成Web界面进行实时监控：

```python
# 使用Flask或Streamlit创建Web界面
from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def dashboard():
    return render_template('dashboard.html', stats=stats)
```

## 技术支持

如果遇到问题，请检查：
1. 环境配置是否正确
2. 模型文件是否存在且兼容
3. 依赖库是否安装完整
4. 硬件资源是否充足

更多信息请参考项目文档或提交Issue。 