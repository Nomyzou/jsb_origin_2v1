#!/usr/bin/env python
"""
飞机初始位置可视化脚本
展示4v4场景中飞机的初始位置分布
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import yaml
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def extract_aircraft_positions(config):
    """提取飞机初始位置"""
    aircraft_configs = config['aircraft_configs']
    positions = {}
    
    for aircraft_id, aircraft_config in aircraft_configs.items():
        init_state = aircraft_config['init_state']
        color = aircraft_config['color']
        
        positions[aircraft_id] = {
            'longitude': init_state['ic_long_gc_deg'],
            'latitude': init_state['ic_lat_geod_deg'],
            'altitude': init_state['ic_h_sl_ft'],
            'heading': init_state['ic_psi_true_deg'],
            'color': color
        }
    
    return positions

def visualize_positions(positions, config_path):
    """可视化飞机初始位置"""
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 颜色映射
    color_map = {'Red': 'red', 'Blue': 'blue'}
    
    # 提取坐标
    red_positions = []
    blue_positions = []
    
    for aircraft_id, pos in positions.items():
        if pos['color'] == 'Red':
            red_positions.append((pos['longitude'], pos['latitude'], pos['altitude'], pos['heading']))
        else:
            blue_positions.append((pos['longitude'], pos['latitude'], pos['altitude'], pos['heading']))
    
    # 转换为numpy数组
    red_positions = np.array(red_positions)
    blue_positions = np.array(blue_positions)
    
    # 1. 水平位置图（经纬度）
    ax1.set_title('飞机初始水平位置分布', fontsize=14, fontweight='bold')
    
    # 绘制红方飞机
    if len(red_positions) > 0:
        ax1.scatter(red_positions[:, 0], red_positions[:, 1], 
                   c='red', s=100, alpha=0.7, label='红方飞机', zorder=3)
        # 添加飞机编号
        for i, (lon, lat, alt, heading) in enumerate(red_positions):
            ax1.annotate(f'A0{i+1}00', (lon, lat), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=10, fontweight='bold', color='darkred')
    
    # 绘制蓝方飞机
    if len(blue_positions) > 0:
        ax1.scatter(blue_positions[:, 0], blue_positions[:, 1], 
                   c='blue', s=100, alpha=0.7, label='蓝方飞机', zorder=3)
        # 添加飞机编号
        for i, (lon, lat, alt, heading) in enumerate(blue_positions):
            ax1.annotate(f'B0{i+1}00', (lon, lat), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=10, fontweight='bold', color='darkblue')
    
    # 绘制航向箭头
    arrow_length = 0.005
    for lon, lat, alt, heading in red_positions:
        dx = arrow_length * np.cos(np.radians(heading))
        dy = arrow_length * np.sin(np.radians(heading))
        ax1.arrow(lon, lat, dx, dy, head_width=0.002, head_length=0.001, 
                 fc='red', ec='red', alpha=0.8)
    
    for lon, lat, alt, heading in blue_positions:
        dx = arrow_length * np.cos(np.radians(heading))
        dy = arrow_length * np.sin(np.radians(heading))
        ax1.arrow(lon, lat, dx, dy, head_width=0.002, head_length=0.001, 
                 fc='blue', ec='blue', alpha=0.8)
    
    ax1.set_xlabel('经度 (度)', fontsize=12)
    ax1.set_ylabel('纬度 (度)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 设置坐标轴范围
    all_lons = [pos['longitude'] for pos in positions.values()]
    all_lats = [pos['latitude'] for pos in positions.values()]
    
    lon_margin = (max(all_lons) - min(all_lons)) * 0.1
    lat_margin = (max(all_lats) - min(all_lats)) * 0.1
    
    ax1.set_xlim(min(all_lons) - lon_margin, max(all_lons) + lon_margin)
    ax1.set_ylim(min(all_lats) - lat_margin, max(all_lats) + lat_margin)
    
    # 2. 高度分布图
    ax2.set_title('飞机初始高度分布', fontsize=14, fontweight='bold')
    
    # 绘制高度柱状图
    aircraft_ids = list(positions.keys())
    altitudes = [positions[aid]['altitude'] for aid in aircraft_ids]
    colors = [color_map[positions[aid]['color']] for aid in aircraft_ids]
    
    bars = ax2.bar(range(len(aircraft_ids)), altitudes, color=colors, alpha=0.7)
    ax2.set_xlabel('飞机编号', fontsize=12)
    ax2.set_ylabel('高度 (英尺)', fontsize=12)
    ax2.set_xticks(range(len(aircraft_ids)))
    ax2.set_xticklabels(aircraft_ids, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 添加高度数值标签
    for i, (bar, alt) in enumerate(zip(bars, altitudes)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{alt:.0f}', ha='center', va='bottom', fontsize=10)
    
    # 添加统计信息
    red_alts = [pos['altitude'] for pos in positions.values() if pos['color'] == 'Red']
    blue_alts = [pos['altitude'] for pos in positions.values() if pos['color'] == 'Blue']
    
    stats_text = f"""
统计信息:
红方平均高度: {np.mean(red_alts):.0f} ft
蓝方平均高度: {np.mean(blue_alts):.0f} ft
总体高度范围: {min(altitudes):.0f} - {max(altitudes):.0f} ft
高度标准差: {np.std(altitudes):.0f} ft
    """
    
    ax2.text(0.02, 0.98, stats_text.strip(), transform=ax2.transAxes, 
             verticalalignment='top', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = config_path.replace('.yaml', '')
    output_path = f"{base_name}_positions_{timestamp}.png"
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"位置分布图已保存到: {output_path}")
    
    # 显示图片
    plt.show()
    
    return fig

def print_position_summary(positions):
    """打印位置摘要"""
    print("\n" + "="*60)
    print("飞机初始位置摘要")
    print("="*60)
    
    red_positions = []
    blue_positions = []
    
    for aircraft_id, pos in positions.items():
        if pos['color'] == 'Red':
            red_positions.append((aircraft_id, pos))
        else:
            blue_positions.append((aircraft_id, pos))
    
    print("\n红方飞机:")
    print("-" * 40)
    for aircraft_id, pos in red_positions:
        print(f"{aircraft_id}: 经度={pos['longitude']:.6f}°, "
              f"纬度={pos['latitude']:.6f}°, "
              f"高度={pos['altitude']:.0f}ft, "
              f"航向={pos['heading']:.1f}°")
    
    print("\n蓝方飞机:")
    print("-" * 40)
    for aircraft_id, pos in blue_positions:
        print(f"{aircraft_id}: 经度={pos['longitude']:.6f}°, "
              f"纬度={pos['latitude']:.6f}°, "
              f"高度={pos['altitude']:.0f}ft, "
              f"航向={pos['heading']:.1f}°")
    
    # 计算统计信息
    all_alts = [pos['altitude'] for pos in positions.values()]
    red_alts = [pos['altitude'] for pos in positions.values() if pos['color'] == 'Red']
    blue_alts = [pos['altitude'] for pos in positions.values() if pos['color'] == 'Blue']
    
    print(f"\n高度统计:")
    print(f"  红方平均高度: {np.mean(red_alts):.0f} ft")
    print(f"  蓝方平均高度: {np.mean(blue_alts):.0f} ft")
    print(f"  总体高度范围: {min(all_alts):.0f} - {max(all_alts):.0f} ft")
    print(f"  高度标准差: {np.std(all_alts):.0f} ft")
    print(f"  高度变化率: {np.std(all_alts)/np.mean(all_alts)*100:.1f}%")

def main():
    """主函数"""
    
    # 配置文件路径
    config_path = "envs/JSBSim/configs/4v4/NoWeapon/1v1Style.yaml"
    
    try:
        # 加载配置
        config = load_config(config_path)
        
        # 提取位置信息
        positions = extract_aircraft_positions(config)
        
        # 打印摘要
        print_position_summary(positions)
        
        # 可视化位置
        fig = visualize_positions(positions, config_path)
        
        print(f"\n✅ 位置分析完成！")
        print(f"配置文件: {config_path}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main() 