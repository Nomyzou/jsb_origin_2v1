#!/usr/bin/env python3
"""
4v4渲染效果分析脚本
分析不同渲染方法的效果差异，包括轨迹质量、战术多样性等
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from pathlib import Path
import argparse

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from renders.analyze_acmi_positions import ACMIAnalyzer

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RenderEffectAnalyzer:
    """渲染效果分析器"""
    
    def __init__(self):
        self.analysis_results = {}
        
    def analyze_acmi_file(self, acmi_file_path, method_name):
        """分析ACMI文件"""
        logger.info(f"分析 {method_name} 的ACMI文件: {acmi_file_path}")
        
        try:
            analyzer = ACMIAnalyzer(acmi_file_path)
            analyzer.parse_acmi_file()
            
            # 获取初始位置
            initial_positions = analyzer.get_initial_positions()
            
            # 分析位置分布
            analysis = analyzer.analyze_position_distribution()
            
            if analysis:
                self.analysis_results[method_name] = {
                    'initial_positions': initial_positions,
                    'stats': analysis['stats'],
                    'red_aircraft': analysis['red_aircraft'],
                    'blue_aircraft': analysis['blue_aircraft']
                }
                
                logger.info(f"{method_name} 分析完成")
                return True
            else:
                logger.warning(f"{method_name} 分析失败，没有有效数据")
                return False
                
        except Exception as e:
            logger.error(f"{method_name} 分析失败: {e}")
            return False
    
    def compare_methods(self, acmi_files):
        """比较不同渲染方法的效果"""
        logger.info("开始比较不同渲染方法的效果...")
        
        # 分析所有ACMI文件
        for method_name, acmi_file in acmi_files.items():
            if os.path.exists(acmi_file):
                self.analyze_acmi_file(acmi_file, method_name)
            else:
                logger.warning(f"ACMI文件不存在: {acmi_file}")
        
        # 生成对比报告
        self.generate_comparison_report()
        
    def generate_comparison_report(self):
        """生成对比报告"""
        if not self.analysis_results:
            logger.warning("没有分析结果可供比较")
            return
        
        logger.info("生成对比报告...")
        
        # 创建对比图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('4v4渲染方法效果对比分析', fontsize=16)
        
        methods = list(self.analysis_results.keys())
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        # 1. 高度分布对比
        ax1.set_title('不同方法的高度分布对比')
        ax1.set_xlabel('渲染方法')
        ax1.set_ylabel('平均高度 (ft)')
        
        height_means = []
        height_stds = []
        
        for method in methods:
            stats = self.analysis_results[method]['stats']
            height_means.append(stats['altitude_stats']['mean'])
            height_stds.append(stats['altitude_stats']['std'])
        
        bars = ax1.bar(methods, height_means, yerr=height_stds, capsize=5, 
                      color=colors[:len(methods)], alpha=0.7)
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar, mean, std) in enumerate(zip(bars, height_means, height_stds)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                    f'{mean:.0f}±{std:.0f}', ha='center', va='bottom', fontsize=10)
        
        # 2. 高度变化率对比
        ax2.set_title('高度变化率对比')
        ax2.set_xlabel('渲染方法')
        ax2.set_ylabel('高度变化率 (%)')
        
        height_ranges = []
        for method in methods:
            stats = self.analysis_results[method]['stats']
            range_val = stats['altitude_stats']['range']
            mean_val = stats['altitude_stats']['mean']
            change_rate = (range_val / mean_val * 100) if mean_val > 0 else 0
            height_ranges.append(change_rate)
        
        bars = ax2.bar(methods, height_ranges, color=colors[:len(methods)], alpha=0.7)
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar, rate) in enumerate(zip(bars, height_ranges)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 3. 红蓝方高度差异对比
        ax3.set_title('红蓝方高度差异对比')
        ax3.set_xlabel('渲染方法')
        ax3.set_ylabel('红蓝方高度差 (ft)')
        
        height_diffs = []
        for method in methods:
            stats = self.analysis_results[method]['stats']
            red_mean = stats['red_altitude_stats']['mean']
            blue_mean = stats['blue_altitude_stats']['mean']
            height_diffs.append(abs(red_mean - blue_mean))
        
        bars = ax3.bar(methods, height_diffs, color=colors[:len(methods)], alpha=0.7)
        ax3.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar, diff) in enumerate(zip(bars, height_diffs)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{diff:.0f}', ha='center', va='bottom', fontsize=10)
        
        # 4. 位置分布对比
        ax4.set_title('位置分布对比')
        ax4.set_xlabel('经度 (°)')
        ax4.set_ylabel('纬度 (°)')
        ax4.grid(True, alpha=0.3)
        
        # 绘制每种方法的位置分布
        for i, method in enumerate(methods):
            positions = self.analysis_results[method]['initial_positions']
            
            # 红方飞机
            red_positions = [pos for pos in positions.values() if pos['color'].upper() == 'RED']
            if red_positions:
                red_lats = [pos['latitude'] for pos in red_positions]
                red_lons = [pos['longitude'] for pos in red_positions]
                ax4.scatter(red_lons, red_lats, c=colors[i], s=100, alpha=0.7, 
                          marker='o', label=f'{method} (红方)')
            
            # 蓝方飞机
            blue_positions = [pos for pos in positions.values() if pos['color'].upper() == 'BLUE']
            if blue_positions:
                blue_lats = [pos['latitude'] for pos in blue_positions]
                blue_lons = [pos['longitude'] for pos in blue_positions]
                ax4.scatter(blue_lons, blue_lats, c=colors[i], s=100, alpha=0.7, 
                          marker='s', label=f'{method} (蓝方)')
        
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # 保存对比图
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"render_methods_comparison_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"对比图已保存到: {output_path}")
        
        plt.show()
        
        # 生成文本报告
        self.generate_text_report()
        
    def generate_text_report(self):
        """生成文本对比报告"""
        logger.info("生成文本对比报告...")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("4v4渲染方法效果对比分析报告")
        report_lines.append("=" * 80)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        methods = list(self.analysis_results.keys())
        
        # 总体统计
        report_lines.append("总体统计:")
        report_lines.append("-" * 40)
        for method in methods:
            stats = self.analysis_results[method]['stats']
            report_lines.append(f"{method}:")
            report_lines.append(f"  总飞机数: {stats['total_aircraft']}")
            report_lines.append(f"  红方飞机: {stats['red_aircraft']}")
            report_lines.append(f"  蓝方飞机: {stats['blue_aircraft']}")
            report_lines.append("")
        
        # 高度统计对比
        report_lines.append("高度统计对比:")
        report_lines.append("-" * 40)
        for method in methods:
            stats = self.analysis_results[method]['stats']
            report_lines.append(f"{method}:")
            report_lines.append(f"  平均高度: {stats['altitude_stats']['mean']:.0f} ft")
            report_lines.append(f"  高度范围: {stats['altitude_stats']['min']:.0f} - {stats['altitude_stats']['max']:.0f} ft")
            report_lines.append(f"  高度标准差: {stats['altitude_stats']['std']:.0f} ft")
            report_lines.append(f"  高度变化率: {(stats['altitude_stats']['range']/stats['altitude_stats']['mean']*100):.1f}%")
            report_lines.append("")
        
        # 红蓝方对比
        report_lines.append("红蓝方高度对比:")
        report_lines.append("-" * 40)
        for method in methods:
            stats = self.analysis_results[method]['stats']
            report_lines.append(f"{method}:")
            report_lines.append(f"  红方平均高度: {stats['red_altitude_stats']['mean']:.0f} ft")
            report_lines.append(f"  蓝方平均高度: {stats['blue_altitude_stats']['mean']:.0f} ft")
            report_lines.append(f"  高度差异: {abs(stats['red_altitude_stats']['mean'] - stats['blue_altitude_stats']['mean']):.0f} ft")
            report_lines.append("")
        
        # 位置分布分析
        report_lines.append("位置分布分析:")
        report_lines.append("-" * 40)
        for method in methods:
            positions = self.analysis_results[method]['initial_positions']
            report_lines.append(f"{method}:")
            
            # 红方位置
            red_positions = [pos for pos in positions.values() if pos['color'].upper() == 'RED']
            if red_positions:
                red_lats = [pos['latitude'] for pos in red_positions]
                red_lons = [pos['longitude'] for pos in red_positions]
                report_lines.append(f"  红方纬度范围: {min(red_lats):.6f}° - {max(red_lats):.6f}°")
                report_lines.append(f"  红方经度范围: {min(red_lons):.6f}° - {max(red_lons):.6f}°")
            
            # 蓝方位置
            blue_positions = [pos for pos in positions.values() if pos['color'].upper() == 'BLUE']
            if blue_positions:
                blue_lats = [pos['latitude'] for pos in blue_positions]
                blue_lons = [pos['longitude'] for pos in blue_positions]
                report_lines.append(f"  蓝方纬度范围: {min(blue_lats):.6f}° - {max(blue_lats):.6f}°")
                report_lines.append(f"  蓝方经度范围: {min(blue_lons):.6f}° - {max(blue_lons):.6f}°")
            
            report_lines.append("")
        
        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"render_comparison_report_{timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"对比报告已保存到: {report_path}")
        
        # 打印报告
        print('\n'.join(report_lines))

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='分析4v4渲染方法效果')
    parser.add_argument('--acmi-files', nargs='+', help='ACMI文件路径列表')
    parser.add_argument('--method-names', nargs='+', help='对应的方法名称列表')
    parser.add_argument('--auto-scan', action='store_true', help='自动扫描当前目录的ACMI文件')
    
    args = parser.parse_args()
    
    analyzer = RenderEffectAnalyzer()
    
    if args.auto_scan:
        # 自动扫描当前目录的ACMI文件
        acmi_files = {}
        current_dir = Path('.')
        
        for acmi_file in current_dir.glob('*.txt.acmi'):
            # 从文件名推断方法名称
            filename = acmi_file.stem
            if 'individual' in filename:
                method_name = '独立策略网络'
            elif '1v1_style' in filename:
                method_name = '1v1风格'
            elif 'improved' in filename:
                method_name = '改进版'
            else:
                method_name = '标准版'
            
            acmi_files[method_name] = str(acmi_file)
            logger.info(f"发现ACMI文件: {acmi_file} -> {method_name}")
    
    elif args.acmi_files and args.method_names:
        # 使用命令行参数
        if len(args.acmi_files) != len(args.method_names):
            logger.error("ACMI文件数量与方法名称数量不匹配")
            return
        
        acmi_files = dict(zip(args.method_names, args.acmi_files))
    
    else:
        # 使用默认文件
        logger.info("使用默认ACMI文件进行分析...")
        acmi_files = {
            '标准版': '4v4_improved.txt.acmi',
            '1v1风格': '4v4_1v1_style.txt.acmi',
            '独立策略网络': '4v4_individual_policies.txt.acmi'
        }
    
    # 检查文件是否存在
    existing_files = {}
    for method_name, file_path in acmi_files.items():
        if os.path.exists(file_path):
            existing_files[method_name] = file_path
        else:
            logger.warning(f"ACMI文件不存在: {file_path}")
    
    if not existing_files:
        logger.error("没有找到有效的ACMI文件")
        return
    
    # 进行分析
    analyzer.compare_methods(existing_files)
    
    logger.info("✅ 渲染效果分析完成！")

if __name__ == "__main__":
    main() 