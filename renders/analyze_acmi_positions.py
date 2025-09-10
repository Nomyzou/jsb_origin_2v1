#!/usr/bin/env python3
"""
分析ACMI文件中飞机位置分布的脚本
支持分析TacView格式的ACMI文件，提取飞机初始位置和轨迹信息
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import logging

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ACMIAnalyzer:
    """ACMI文件分析器"""
    
    def __init__(self, acmi_file_path):
        self.acmi_file_path = Path(acmi_file_path)
        self.aircraft_data = {}
        self.time_data = []
        
    def parse_acmi_file(self):
        """解析ACMI文件"""
        if not self.acmi_file_path.exists():
            raise FileNotFoundError(f"ACMI文件不存在: {self.acmi_file_path}")
            
        logger.info(f"开始解析ACMI文件: {self.acmi_file_path}")
        
        with open(self.acmi_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 解析文件头信息
        self._parse_header(content)
        
        # 解析飞机定义
        self._parse_aircraft_definitions(content)
        
        # 解析轨迹数据
        self._parse_trajectory_data(content)
        
        logger.info(f"解析完成，发现 {len(self.aircraft_data)} 架飞机")
        
    def _parse_header(self, content):
        """解析文件头信息"""
        # 提取文件版本、时间等信息
        version_match = re.search(r'FileVersion=(\d+)', content)
        if version_match:
            logger.info(f"ACMI文件版本: {version_match.group(1)}")
            
    def _parse_aircraft_definitions(self, content):
        """解析飞机定义部分"""
        # 查找所有飞机定义
        aircraft_pattern = r'(\d+),Type=Air,Name="([^"]+)",Color=(\w+)'
        matches = re.findall(aircraft_pattern, content)
        
        for match in matches:
            obj_id, name, color = match
            self.aircraft_data[obj_id] = {
                'name': name,
                'color': color,
                'positions': [],
                'times': [],
                'altitudes': [],
                'speeds': []
            }
            
        logger.info(f"发现飞机定义: {len(matches)} 架")
        
    def _parse_trajectory_data(self, content):
        """解析轨迹数据"""
        # 解析时间戳
        time_pattern = r'#(\d+\.?\d*)'
        time_matches = re.findall(time_pattern, content)
        self.time_data = [float(t) for t in time_matches]
        
        # 解析位置数据
        position_pattern = r'(\d+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)'
        position_matches = re.findall(position_pattern, content)
        
        for match in position_matches:
            obj_id, lat, lon, alt, heading, speed, time = match
            
            if obj_id in self.aircraft_data:
                try:
                    lat = float(lat)
                    lon = float(lon)
                    alt = float(alt)
                    heading = float(heading)
                    speed = float(speed)
                    
                    self.aircraft_data[obj_id]['positions'].append((lat, lon))
                    self.aircraft_data[obj_id]['altitudes'].append(alt)
                    self.aircraft_data[obj_id]['speeds'].append(speed)
                    
                except ValueError:
                    continue
                    
        logger.info(f"解析轨迹数据点: {len(position_matches)} 个")
        
    def get_initial_positions(self):
        """获取飞机初始位置"""
        initial_positions = {}
        
        for obj_id, data in self.aircraft_data.items():
            if data['positions']:
                lat, lon = data['positions'][0]
                alt = data['altitudes'][0] if data['altitudes'] else 0
                speed = data['speeds'][0] if data['speeds'] else 0
                
                initial_positions[obj_id] = {
                    'name': data['name'],
                    'color': data['color'],
                    'latitude': lat,
                    'longitude': lon,
                    'altitude': alt,
                    'speed': speed
                }
                
        return initial_positions
        
    def analyze_position_distribution(self):
        """分析位置分布"""
        initial_positions = self.get_initial_positions()
        
        if not initial_positions:
            logger.warning("没有找到有效的初始位置数据")
            return None
            
        # 按颜色分组
        red_aircraft = {k: v for k, v in initial_positions.items() if v['color'].upper() == 'RED'}
        blue_aircraft = {k: v for k, v in initial_positions.items() if v['color'].upper() == 'BLUE'}
        
        # 计算统计信息
        all_altitudes = [pos['altitude'] for pos in initial_positions.values()]
        red_altitudes = [pos['altitude'] for pos in red_aircraft.values()]
        blue_altitudes = [pos['altitude'] for pos in blue_aircraft.values()]
        
        stats = {
            'total_aircraft': len(initial_positions),
            'red_aircraft': len(red_aircraft),
            'blue_aircraft': len(blue_aircraft),
            'altitude_stats': {
                'mean': np.mean(all_altitudes),
                'std': np.std(all_altitudes),
                'min': np.min(all_altitudes),
                'max': np.max(all_altitudes),
                'range': np.max(all_altitudes) - np.min(all_altitudes)
            },
            'red_altitude_stats': {
                'mean': np.mean(red_altitudes) if red_altitudes else 0,
                'std': np.std(red_altitudes) if red_altitudes else 0
            },
            'blue_altitude_stats': {
                'mean': np.mean(blue_altitudes) if blue_altitudes else 0,
                'std': np.std(blue_altitudes) if blue_altitudes else 0
            }
        }
        
        return {
            'positions': initial_positions,
            'red_aircraft': red_aircraft,
            'blue_aircraft': blue_aircraft,
            'stats': stats
        }
        
    def visualize_positions(self, output_path=None):
        """可视化飞机位置分布"""
        analysis = self.analyze_position_distribution()
        
        if not analysis:
            logger.error("无法生成可视化，没有有效数据")
            return
            
        positions = analysis['positions']
        red_aircraft = analysis['red_aircraft']
        blue_aircraft = analysis['blue_aircraft']
        stats = analysis['stats']
        
        # 创建图形
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'ACMI文件飞机位置分析 - {self.acmi_file_path.name}', fontsize=16)
        
        # 1. 位置分布图
        ax1.set_title('飞机初始位置分布')
        ax1.set_xlabel('经度 (°)')
        ax1.set_ylabel('纬度 (°)')
        ax1.grid(True, alpha=0.3)
        
        # 绘制红方飞机
        if red_aircraft:
            red_lats = [pos['latitude'] for pos in red_aircraft.values()]
            red_lons = [pos['longitude'] for pos in red_aircraft.values()]
            ax1.scatter(red_lons, red_lats, c='red', s=100, alpha=0.7, label='红方飞机')
            
            # 添加飞机标签
            for obj_id, pos in red_aircraft.items():
                ax1.annotate(pos['name'], (pos['longitude'], pos['latitude']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 绘制蓝方飞机
        if blue_aircraft:
            blue_lats = [pos['latitude'] for pos in blue_aircraft.values()]
            blue_lons = [pos['longitude'] for pos in blue_aircraft.values()]
            ax1.scatter(blue_lons, blue_lats, c='blue', s=100, alpha=0.7, label='蓝方飞机')
            
            # 添加飞机标签
            for obj_id, pos in blue_aircraft.items():
                ax1.annotate(pos['name'], (pos['longitude'], pos['latitude']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax1.legend()
        
        # 2. 高度分布图
        ax2.set_title('飞机高度分布')
        ax2.set_xlabel('飞机')
        ax2.set_ylabel('高度 (ft)')
        
        all_names = [pos['name'] for pos in positions.values()]
        all_altitudes = [pos['altitude'] for pos in positions.values()]
        
        colors = ['red' if pos['color'].upper() == 'RED' else 'blue' for pos in positions.values()]
        bars = ax2.bar(range(len(all_names)), all_altitudes, color=colors, alpha=0.7)
        ax2.set_xticks(range(len(all_names)))
        ax2.set_xticklabels(all_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # 添加高度标签
        for i, (bar, alt) in enumerate(zip(bars, all_altitudes)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'{alt:.0f}', ha='center', va='bottom', fontsize=8)
        
        # 3. 高度统计图
        ax3.set_title('高度统计对比')
        
        categories = ['总体', '红方', '蓝方']
        means = [stats['altitude_stats']['mean'], 
                stats['red_altitude_stats']['mean'], 
                stats['blue_altitude_stats']['mean']]
        stds = [stats['altitude_stats']['std'], 
               stats['red_altitude_stats']['std'], 
               stats['blue_altitude_stats']['std']]
        
        x_pos = np.arange(len(categories))
        bars = ax3.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                      color=['gray', 'red', 'blue'])
        ax3.set_ylabel('高度 (ft)')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(categories)
        ax3.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                    f'{mean:.0f}±{std:.0f}', ha='center', va='bottom', fontsize=10)
        
        # 4. 位置统计表
        ax4.axis('off')
        
        # 创建统计文本
        stats_text = f"""
ACMI文件分析结果
================

文件信息:
- 文件名: {self.acmi_file_path.name}
- 总飞机数: {stats['total_aircraft']}
- 红方飞机: {stats['red_aircraft']}
- 蓝方飞机: {stats['blue_aircraft']}

高度统计:
- 平均高度: {stats['altitude_stats']['mean']:.0f} ft
- 高度范围: {stats['altitude_stats']['min']:.0f} - {stats['altitude_stats']['max']:.0f} ft
- 高度标准差: {stats['altitude_stats']['std']:.0f} ft
- 高度变化率: {(stats['altitude_stats']['range']/stats['altitude_stats']['mean']*100):.1f}%

红方高度:
- 平均: {stats['red_altitude_stats']['mean']:.0f} ft
- 标准差: {stats['red_altitude_stats']['std']:.0f} ft

蓝方高度:
- 平均: {stats['blue_altitude_stats']['mean']:.0f} ft
- 标准差: {stats['blue_altitude_stats']['std']:.0f} ft
        """
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # 保存图片
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"acmi_positions_analysis_{timestamp}.png"
            
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"位置分析图已保存到: {output_path}")
        
        plt.show()
        
        return output_path

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='分析ACMI文件中的飞机位置分布')
    parser.add_argument('acmi_file', help='ACMI文件路径')
    parser.add_argument('-o', '--output', help='输出图片路径')
    parser.add_argument('--no-display', action='store_true', help='不显示图形窗口')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = ACMIAnalyzer(args.acmi_file)
    
    try:
        # 解析ACMI文件
        analyzer.parse_acmi_file()
        
        # 获取初始位置
        initial_positions = analyzer.get_initial_positions()
        
        if not initial_positions:
            logger.error("没有找到有效的飞机位置数据")
            return
            
        # 打印位置摘要
        print("\n" + "="*60)
        print("ACMI文件飞机位置摘要")
        print("="*60)
        
        red_aircraft = {k: v for k, v in initial_positions.items() if v['color'].upper() == 'RED'}
        blue_aircraft = {k: v for k, v in initial_positions.items() if v['color'].upper() == 'BLUE'}
        
        print(f"\n红方飞机 ({len(red_aircraft)}架):")
        print("-" * 40)
        for obj_id, pos in red_aircraft.items():
            print(f"{pos['name']}: 经度={pos['longitude']:.6f}°, 纬度={pos['latitude']:.6f}°, "
                  f"高度={pos['altitude']:.0f}ft, 速度={pos['speed']:.0f}kt")
        
        print(f"\n蓝方飞机 ({len(blue_aircraft)}架):")
        print("-" * 40)
        for obj_id, pos in blue_aircraft.items():
            print(f"{pos['name']}: 经度={pos['longitude']:.6f}°, 纬度={pos['latitude']:.6f}°, "
                  f"高度={pos['altitude']:.0f}ft, 速度={pos['speed']:.0f}kt")
        
        # 分析高度分布
        all_altitudes = [pos['altitude'] for pos in initial_positions.values()]
        print(f"\n高度统计:")
        print(f"  红方平均高度: {np.mean([pos['altitude'] for pos in red_aircraft.values()]):.0f} ft")
        print(f"  蓝方平均高度: {np.mean([pos['altitude'] for pos in blue_aircraft.values()]):.0f} ft")
        print(f"  总体高度范围: {np.min(all_altitudes):.0f} - {np.max(all_altitudes):.0f} ft")
        print(f"  高度标准差: {np.std(all_altitudes):.0f} ft")
        print(f"  高度变化率: {(np.max(all_altitudes)-np.min(all_altitudes))/np.mean(all_altitudes)*100:.1f}%")
        
        # 生成可视化
        if not args.no_display:
            output_path = analyzer.visualize_positions(args.output)
            print(f"\n位置分布图已保存到: {output_path}")
        
        print(f"\n配置文件: {args.acmi_file}")
        print("\n✅ 位置分析完成！")
        
    except Exception as e:
        logger.error(f"分析过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main() 