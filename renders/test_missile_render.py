#!/usr/bin/env python3
"""
导弹渲染测试脚本
展示导弹在ACMI文件中的渲染原理
"""

import os
import sys
import logging
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.JSBSim.envs import SingleCombatEnv
from envs.JSBSim.core.simulatior import MissileSimulator

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_missile_render():
    """测试导弹渲染功能"""
    
    # 创建1v1导弹任务环境
    logger.info("创建1v1导弹任务环境...")
    env = SingleCombatEnv("1v1/ShootMissile/HierarchySelfplay")
    env.seed(0)
    
    logger.info(f"环境创建成功，智能体数量: {env.num_agents}")
    logger.info(f"任务类型: {env.task.__class__.__name__}")
    
    # 检查导弹配置
    for agent_id, agent in env.agents.items():
        logger.info(f"智能体 {agent_id}: 导弹数量 = {agent.num_missiles}")
    
    # 重置环境
    obs, _ = env.reset()
    
    # 设置ACMI渲染文件
    render_file = f'missile_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt.acmi'
    env.render(mode='txt', filepath=render_file)
    logger.info(f"开始渲染到: {render_file}")
    
    # 模拟导弹发射
    logger.info("模拟导弹发射...")
    
    # 获取第一个智能体作为发射者
    launcher_id = list(env.agents.keys())[0]
    launcher = env.agents[launcher_id]
    
    # 获取目标（敌人）
    target = launcher.enemies[0]
    
    logger.info(f"发射者: {launcher_id} (位置: {launcher.get_position()})")
    logger.info(f"目标: {target.uid} (位置: {target.get_position()})")
    
    # 创建导弹
    missile_uid = f"M{launcher_id[1:]}1"  # 例如: MA1001
    missile = MissileSimulator.create(
        parent=launcher,
        target=target,
        uid=missile_uid,
        missile_model="AIM-9L"
    )
    
    # 将导弹添加到环境的临时模拟器中
    env.add_temp_simulator(missile)
    
    logger.info(f"导弹 {missile_uid} 已创建并添加到环境")
    logger.info(f"导弹初始位置: {missile.get_position()}")
    logger.info(f"导弹初始速度: {missile.get_velocity()}")
    
    # 运行模拟，观察导弹轨迹
    step_count = 0
    max_steps = 100  # 最多运行100步
    
    try:
        while step_count < max_steps:
            step_count += 1
            
            # 渲染当前状态
            env.render(mode='txt', filepath=render_file)
            
            # 运行所有模拟器
            for sim in env._jsbsims.values():
                sim.run()
            for sim in env._tempsims.values():
                sim.run()
            
            # 检查导弹状态
            if missile.is_alive:
                logger.info(f"步骤 {step_count}: 导弹位置 = {missile.get_position()}, 距离目标 = {missile.target_distance:.1f}m")
            elif missile.is_success:
                logger.info(f"步骤 {step_count}: 导弹命中目标!")
                break
            elif missile.is_done:
                logger.info(f"步骤 {step_count}: 导弹未命中目标")
                break
            
            # 检查目标是否被击落
            if not target.is_alive:
                logger.info(f"步骤 {step_count}: 目标被击落!")
                break
                
    except KeyboardInterrupt:
        logger.info("用户中断模拟")
    
    # 输出最终结果
    logger.info(f"模拟结束，总步数: {step_count}")
    logger.info(f"导弹状态: {'命中' if missile.is_success else '未命中' if missile.is_done else '飞行中'}")
    logger.info(f"目标状态: {'存活' if target.is_alive else '被击落'}")
    logger.info(f"ACMI文件已保存: {render_file}")
    
    # 分析ACMI文件内容
    analyze_acmi_file(render_file)

def analyze_acmi_file(acmi_file):
    """分析ACMI文件内容"""
    logger.info(f"分析ACMI文件: {acmi_file}")
    
    if not os.path.exists(acmi_file):
        logger.error(f"ACMI文件不存在: {acmi_file}")
        return
    
    try:
        with open(acmi_file, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
        
        logger.info(f"ACMI文件总行数: {len(lines)}")
        
        # 查找导弹相关的行
        missile_lines = []
        aircraft_lines = []
        explosion_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('M'):  # 导弹行
                missile_lines.append((i, line))
            elif line.startswith('A') or line.startswith('B'):  # 飞机行
                aircraft_lines.append((i, line))
            elif 'Explosion' in line:  # 爆炸行
                explosion_lines.append((i, line))
        
        logger.info(f"导弹相关行数: {len(missile_lines)}")
        logger.info(f"飞机相关行数: {len(aircraft_lines)}")
        logger.info(f"爆炸相关行数: {len(explosion_lines)}")
        
        # 显示前几行导弹数据
        if missile_lines:
            logger.info("导弹数据示例:")
            for i, (line_num, line) in enumerate(missile_lines[:3]):
                logger.info(f"  行 {line_num}: {line}")
        
        # 显示爆炸数据
        if explosion_lines:
            logger.info("爆炸数据示例:")
            for line_num, line in explosion_lines:
                logger.info(f"  行 {line_num}: {line}")
                
    except Exception as e:
        logger.error(f"分析ACMI文件时出错: {e}")

def explain_missile_render():
    """解释导弹渲染原理"""
    logger.info("=" * 60)
    logger.info("导弹渲染原理说明")
    logger.info("=" * 60)
    
    logger.info("1. 导弹创建过程:")
    logger.info("   - 通过 MissileSimulator.create() 创建导弹对象")
    logger.info("   - 继承发射飞机的初始位置、速度和姿态")
    logger.info("   - 设置目标飞机")
    logger.info("   - 添加到环境的临时模拟器列表 (_tempsims)")
    
    logger.info("\n2. 导弹渲染机制:")
    logger.info("   - 导弹作为临时模拟器，在每次渲染时调用 log() 方法")
    logger.info("   - 导弹飞行时: 输出位置、姿态、速度等信息")
    logger.info("   - 导弹命中时: 输出爆炸效果 (Explosion)")
    logger.info("   - 导弹未命中时: 输出消失标记 (-uid)")
    
    logger.info("\n3. ACMI文件格式:")
    logger.info("   - 每行代表一个时间步的物体状态")
    logger.info("   - 格式: uid,T=lon|lat|alt|roll|pitch|yaw,Name=模型名,Color=颜色")
    logger.info("   - 导弹: Name=AIM-9L, Color=发射方颜色")
    logger.info("   - 爆炸: Type=Misc+Explosion, Radius=爆炸半径")
    
    logger.info("\n4. 导弹参数:")
    logger.info("   - 最大飞行时间: 60秒")
    logger.info("   - 发动机工作时间: 3秒")
    logger.info("   - 爆炸半径: 300米")
    logger.info("   - 最小速度: 150 m/s")
    logger.info("   - 制导方式: 比例导引")
    
    logger.info("\n5. 渲染文件位置:")
    logger.info("   - 默认: ./JSBSimRecording.txt.acmi")
    logger.info("   - 可用TacView软件打开查看3D轨迹")

if __name__ == "__main__":
    explain_missile_render()
    test_missile_render() 