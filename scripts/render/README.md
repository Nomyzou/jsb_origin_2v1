# 导弹战斗渲染脚本使用说明

## 概述

本目录包含用于渲染导弹战斗场景的脚本，可以生成ACMI文件供TacView查看。

## 核心脚本

### 1. `render_1v1_missile_simple.py` - 简化版本 ⭐
**简化版导弹战斗渲染脚本**
- 代码简洁易懂，类似不带导弹版本的结构
- 保持完整的导弹功能（强制发射、标准ID格式）
- 适合快速使用和修改

```bash
# 基本使用
python scripts/render/render_1v1_missile_simple.py
```

### 2. `render_1v1_missile_standard.py` - 标准版本 🔧
**标准导弹战斗渲染脚本**
- 使用标准导弹ID格式，确保TacView兼容性
- 支持强制发射导弹功能
- 完整的日志输出和错误处理
- 支持命令行参数配置

```bash
# 基本使用
python scripts/render/render_1v1_missile_standard.py

# 强制发射导弹（第10步）
python scripts/render/render_1v1_missile_standard.py --force-shoot --shoot-step 10

# 自定义参数
python scripts/render/render_1v1_missile_standard.py --render-episodes 3 --episode-length 200
```

### 3. `diagnose_tacview_missiles.py` - 诊断工具 🔧
**ACMI文件分析工具**
- 验证导弹数据完整性
- 提供TacView显示建议
- 统计导弹轨迹信息

```bash
python scripts/render/diagnose_tacview_missiles.py
```

### 4. `render_1v1_missile.py` - 原始版本 📜
**原始导弹战斗渲染脚本**
- 保持原始功能，作为参考
- 主要用于对比和兼容性测试

### 5. `render_jsbsim.py` - 通用渲染 🌐
**通用JSBSim环境渲染**
- 支持多种环境类型
- 用于其他场景的渲染

## 主要参数

- `--render-episodes`: 渲染的episode数量（默认：1）
- `--episode-length`: 每个episode的最大步数（默认：100）
- `--output-dir`: 输出目录（默认：renders/missile_combat_standard）
- `--force-shoot`: 启用强制发射导弹
- `--shoot-step`: 强制发射的步数（默认：10）
- `--seed`: 随机种子（默认：42）

## 输出结构

```
renders/missile_combat_standard/
├── episode_1/
│   └── episode_1.acmi
├── episode_2/
│   └── episode_2.acmi
└── ...
```

## TacView使用

1. **下载TacView**: https://www.tacview.net/
2. **打开ACMI文件**: 双击生成的.acmi文件
3. **检查显示设置**:
   - View -> Objects -> Missiles: 启用
   - View -> Objects -> Explosions: 启用
   - View -> Objects -> Aircraft: 启用

## 故障排除

### 看不到导弹？
1. 运行诊断脚本: `python scripts/render/diagnose_tacview_missiles.py`
2. 检查TacView显示设置
3. 确保导弹对象类型已启用

### 模型加载错误？
1. 检查模型文件路径
2. 确保依赖文件完整
3. 验证环境配置

## 快速开始

```bash
# 1. 生成导弹战斗数据（简化版本）
python scripts/render/render_1v1_missile_simple.py

# 2. 诊断数据完整性
python scripts/render/diagnose_tacview_missiles.py

# 3. 在TacView中查看结果
```

## 相关文档

- [TacView导弹显示问题解决方案](../renders/TACVIEW_MISSILE_DISPLAY_FIX.md)
- [清理总结](CLEANUP_SUMMARY.md) 