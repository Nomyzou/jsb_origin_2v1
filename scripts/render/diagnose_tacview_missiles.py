#!/usr/bin/env python3
"""
TacView导弹显示问题诊断脚本
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def analyze_acmi_file(acmi_file):
    """分析ACMI文件中的导弹数据"""
    print(f"分析文件: {acmi_file}")
    print("=" * 60)
    
    with open(acmi_file, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # 统计信息
    total_lines = len(lines)
    missile_lines = [line for line in lines if line.startswith('M') and not line.startswith('M01001F')]
    explosion_lines = [line for line in lines if 'Explosion' in line]
    aircraft_lines = [line for line in lines if line.startswith('A') or line.startswith('B')]
    
    print(f"总行数: {total_lines}")
    print(f"飞机行数: {len(aircraft_lines)}")
    print(f"导弹行数: {len(missile_lines)}")
    print(f"爆炸行数: {len(explosion_lines)}")
    
    # 显示文件头
    print("\n文件头:")
    for i in range(min(5, len(lines))):
        print(f"  {lines[i]}")
    
    # 显示导弹数据示例
    if missile_lines:
        print(f"\n导弹数据示例 (前5行):")
        for i, line in enumerate(missile_lines[:5]):
            print(f"  {line}")
    
    # 显示爆炸数据
    if explosion_lines:
        print(f"\n爆炸数据:")
        for line in explosion_lines:
            print(f"  {line}")
    
    # 检查导弹ID模式
    missile_ids = set()
    for line in missile_lines:
        if line.strip():
            parts = line.split(',')
            if parts:
                missile_ids.add(parts[0])
    
    print(f"\n导弹ID列表: {sorted(list(missile_ids))}")
    
    # 检查时间步
    timestamps = [line for line in lines if line.startswith('#')]
    print(f"\n时间步数: {len(timestamps)}")
    if timestamps:
        print(f"时间范围: {timestamps[0]} 到 {timestamps[-1]}")

def check_tacview_settings():
    """检查TacView显示设置建议"""
    print("\n" + "=" * 60)
    print("TacView显示设置建议")
    print("=" * 60)
    
    print("1. 确保TacView版本支持ACMI 2.1格式")
    print("2. 在TacView中检查以下设置:")
    print("   - View -> Objects -> Missiles: 确保已启用")
    print("   - View -> Objects -> Explosions: 确保已启用")
    print("   - View -> Objects -> Aircraft: 确保已启用")
    print("3. 检查对象过滤器设置:")
    print("   - 确保没有过滤掉导弹对象")
    print("   - 检查颜色设置是否正确")
    print("4. 尝试不同的显示模式:")
    print("   - 3D视图")
    print("   - 2D俯视图")
    print("   - 时间轴视图")

def main():
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 分析现有的ACMI文件
    acmi_file = "renders/missile_combat_direct/episode_1/episode_1.acmi"
    
    if os.path.exists(acmi_file):
        analyze_acmi_file(acmi_file)
    else:
        print(f"文件不存在: {acmi_file}")
        print("请先运行导弹渲染脚本生成ACMI文件")
        return
    
    # 提供TacView设置建议
    check_tacview_settings()
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("如果ACMI文件中包含导弹数据但TacView中看不到:")
    print("1. 检查TacView的显示设置")
    print("2. 确保导弹对象类型已启用")
    print("3. 尝试重新加载文件")
    print("4. 检查TacView版本是否支持当前格式")

if __name__ == "__main__":
    main() 