import numpy as np
import torch
import os
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from envs.JSBSim.envs import MultipleCombatEnv
from algorithms.ppo.ppo_actor import PPOActor
from envs.JSBSim.utils.situation_assessment import get_situation_adv
from envs.JSBSim.utils.utils import LLA2NEU, get_AO_TA_R
from envs.JSBSim.core.catalog import Catalog as c
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局变量用于记录数据
target_matching_history = []

# 伤害计算相关常量
DAMAGE_DISTANCE_THRESHOLD = 1000.0  # 伤害距离阈值 (米) - 降低到1公里
DAMAGE_ANGLE_THRESHOLD = np.pi/3     # 伤害角度阈值 (60度，弧度) - 降低角度要求
MAX_DAMAGE_PER_STEP = 1.0           # 每步最大伤害值 - 降低到1
DAMAGE_BASE_RATE = 2              # 基础伤害率 - 降低基础伤害

class Args:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = True
    
def _t2n(x):
    return x.detach().cpu().numpy()

def load_model_safely(model_path, policy, device):
    """安全加载模型权重"""
    try:
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            policy.load_state_dict(state_dict)
            logger.info(f"Successfully loaded model from {model_path}")
            return True
        else:
            logger.warning(f"Model file not found: {model_path}")
            return False
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return False

def calculate_damage(ego_feature, enm_feature, distance_threshold=DAMAGE_DISTANCE_THRESHOLD, 
                    angle_threshold=DAMAGE_ANGLE_THRESHOLD, base_rate=DAMAGE_BASE_RATE):
    """
    计算基于角度和距离的伤害值 - 修正版本
    
    Args:
        ego_feature: 己方飞机特征 (north, east, down, vn, ve, vd)
        enm_feature: 敌方飞机特征 (north, east, down, vn, ve, vd)
        distance_threshold: 伤害距离阈值 (米)
        angle_threshold: 伤害角度阈值 (弧度，默认60度)
        base_rate: 基础伤害率
    
    Returns:
        damage: 我方对敌方的伤害值 (0-MAX_DAMAGE_PER_STEP之间)
    """
    try:
        # 计算AO和TA角度以及距离
        AO, TA, R, _ = get_AO_TA_R(ego_feature, enm_feature, return_side=True)
        
        # 检查是否满足伤害条件
        if R > distance_threshold:
            return 0.0
        
        # 正确的伤害计算逻辑：
        # 当AO < angle_threshold时，我方对敌方造成伤害
        # AO越小伤害越大，TA越大伤害越大
        
        if AO < angle_threshold:  # 我方朝向敌机（AO < 60度）
            # AO角度因子：AO越小，伤害越大
            ao_factor = 1.0 - (AO / angle_threshold)
            ao_factor = np.clip(ao_factor, 0, 1)
            
            # TA角度因子：TA越大，伤害越大（敌机越难逃脱）
            # TA从0到π，我们希望在TA较大时伤害更大
            ta_factor = TA / np.pi
            ta_factor = np.clip(ta_factor, 0, 1)
            
            # 综合角度因子
            angle_factor = ao_factor * ta_factor
        else:
            angle_factor = 0.0
        
        # 计算基于距离的伤害 (距离越近伤害越大)
        distance_factor = 3.0 - (R / distance_threshold)
        distance_factor = np.clip(distance_factor, 0, 10)
        
        # 综合伤害计算
        damage = base_rate * angle_factor * distance_factor
        damage = np.clip(damage, 0, MAX_DAMAGE_PER_STEP)
        
        return damage
        
    except Exception as e:
        logger.error(f"Error calculating damage: {e}")
        return 0.0

def apply_damage_system(env, center_lon=120.0, center_lat=60.0, center_alt=0.0):
    """
    应用伤害系统：计算并应用所有飞机之间的伤害
    
    Args:
        env: 环境对象
        center_lon, center_lat, center_alt: 战场中心坐标
    
    Returns:
        damage_log: 伤害日志信息
    """
    try:
        damage_log = []
        state_var = env.task.state_var
        
        # 获取所有飞机ID
        all_agent_ids = list(env.agents.keys())
        red_agent_ids = [aid for aid in all_agent_ids if aid.startswith('A')]
        blue_agent_ids = [aid for aid in all_agent_ids if aid.startswith('B')]
        
        # 初始化伤害累积
        damage_to_red = {red_id: 0.0 for red_id in red_agent_ids}
        damage_to_blue = {blue_id: 0.0 for blue_id in blue_agent_ids}
        
        # 一次遍历：计算所有飞机对彼此的伤害
        for red_id in red_agent_ids:
            if not env.agents[red_id].is_alive:
                continue
                
            red_agent = env.agents[red_id]
            red_state = red_agent.get_property_values(state_var)
            red_cur_ned = LLA2NEU(*red_state[:3], center_lon, center_lat, center_alt)
            red_feature = np.array([*red_cur_ned, *(red_state[6:9])])
            
            for blue_id in blue_agent_ids:
                if not env.agents[blue_id].is_alive:
                    continue
                    
                blue_agent = env.agents[blue_id]
                blue_state = blue_agent.get_property_values(state_var)
                blue_cur_ned = LLA2NEU(*blue_state[:3], center_lon, center_lat, center_alt)
                blue_feature = np.array([*blue_cur_ned, *(blue_state[6:9])])
                
                # 计算距离
                distance = np.linalg.norm([blue_cur_ned[0] - red_cur_ned[0], 
                                         blue_cur_ned[1] - red_cur_ned[1], 
                                         blue_cur_ned[2] - red_cur_ned[2]])
                
                # 计算我方对敌方的伤害
                red_to_blue_damage = calculate_damage(red_feature, blue_feature)
                if red_to_blue_damage > 0:
                    damage_to_blue[blue_id] += red_to_blue_damage
                    damage_log.append({
                        'attacker': red_id,
                        'target': blue_id,
                        'damage': red_to_blue_damage,
                        'type': 'AO_damage',
                        'distance': distance
                    })
                
                # 计算敌方对我方的伤害
                blue_to_red_damage = calculate_damage(blue_feature, red_feature)
                if blue_to_red_damage > 0:
                    damage_to_red[red_id] += blue_to_red_damage
                    damage_log.append({
                        'attacker': blue_id,
                        'target': red_id,
                        'damage': blue_to_red_damage,
                        'type': 'TA_damage',
                        'distance': distance
                    })
        
        # 应用伤害到所有飞机
        for red_id, damage in damage_to_red.items():
            if damage > 0 and env.agents[red_id].is_alive:
                env.agents[red_id].bloods = max(0, env.agents[red_id].bloods - damage)
                if env.agents[red_id].bloods <= 0:
                    env.agents[red_id].is_alive = False
                    logger.info(f"{red_id} 被击落! 总伤害: {damage:.4f}")
        
        for blue_id, damage in damage_to_blue.items():
            if damage > 0 and env.agents[blue_id].is_alive:
                env.agents[blue_id].bloods = max(0, env.agents[blue_id].bloods - damage)
                if env.agents[blue_id].bloods <= 0:
                    env.agents[blue_id].is_alive = False
                    logger.info(f"{blue_id} 被击落! 总伤害: {damage:.4f}")
        
        return damage_log
        
    except Exception as e:
        logger.error(f"Error in damage system: {e}")
        return []

def select_best_target(ego_agent_id, env, center_lon=120.0, center_lat=60.0, center_alt=0.0):
    """
    为指定我方飞机选择最优敌方目标 - 简化版本
    通过计算双向态势优势函数，选择优势差值最大的敌方飞机
    允许多架我方飞机选择同一个目标
    """
    try:
        # 获取我方飞机对象
        ego_agent = env.agents[ego_agent_id]
        
        # 获取所有敌方飞机ID
        enemy_ids = []
        for agent_id in env.agents.keys():
            if agent_id.startswith('B') and env.agents[agent_id].is_alive:  # 只选择存活的敌方飞机
                enemy_ids.append(agent_id)
        
        if not enemy_ids:
            logger.warning(f"No alive enemy agents found for {ego_agent_id}")
            return None, None
        
        # 获取状态变量
        state_var = env.task.state_var
        
        # 计算双向态势优势
        advantage_differences = {}
        advantage_details = {}
        for enemy_id in enemy_ids:
            try:
                enemy_agent = env.agents[enemy_id]
                
                # 获取状态值
                ego_state = ego_agent.get_property_values(state_var)
                enemy_state = enemy_agent.get_property_values(state_var)
                
                # 计算我方对敌方的优势（对我方有利的值为正）
                my_advantage = get_situation_adv(
                    ego_state, enemy_state, center_lon, center_lat, center_alt
                )
                
                # 计算敌方对我方的优势（对敌方有利的值为正）
                enemy_advantage = get_situation_adv(
                    enemy_state, ego_state, center_lon, center_lat, center_alt
                )
                
                # 计算优势差值（我方优势 - 敌方优势）
                advantage_diff = my_advantage - enemy_advantage
                
                # 添加距离惩罚因子，优先选择距离较近的目标
                ego_cur_ned = LLA2NEU(*ego_state[:3], center_lon, center_lat, center_alt)
                enemy_cur_ned = LLA2NEU(*enemy_state[:3], center_lon, center_lat, center_alt)
                distance = np.linalg.norm([enemy_cur_ned[0] - ego_cur_ned[0], 
                                         enemy_cur_ned[1] - ego_cur_ned[1], 
                                         enemy_cur_ned[2] - ego_cur_ned[2]])
                
                # 距离惩罚：距离越远，优势值越低
                distance_penalty = max(0, 1.0 - distance / 10000.0)  # 10公里内无惩罚
                adjusted_advantage_diff = advantage_diff * distance_penalty
                
                advantage_differences[enemy_id] = adjusted_advantage_diff
                advantage_details[enemy_id] = {
                    'my_advantage': my_advantage,
                    'enemy_advantage': enemy_advantage,
                    'advantage_diff': advantage_diff,
                    'adjusted_advantage_diff': adjusted_advantage_diff,
                    'distance': distance
                }
                
                logger.debug(f"{ego_agent_id} vs {enemy_id}: my_adv={my_advantage:.4f}, enemy_adv={enemy_advantage:.4f}, "
                           f"diff={advantage_diff:.4f}, adj_diff={adjusted_advantage_diff:.4f}, dist={distance:.1f}m")
                
            except Exception as e:
                logger.warning(f"Failed to calculate advantage for {ego_agent_id} vs {enemy_id}: {e}")
                advantage_differences[enemy_id] = -np.inf
        
        # 选择优势差值最大的敌方飞机
        if advantage_differences:
            best_enemy = max(advantage_differences.keys(), key=lambda x: advantage_differences[x])
            best_diff = advantage_differences[best_enemy]
            
            logger.info(f"{ego_agent_id} selected target: {best_enemy} "
                       f"(advantage_diff: {advantage_details[best_enemy]['advantage_diff']:.4f}, "
                       f"adjusted_diff: {best_diff:.4f}, "
                       f"distance: {advantage_details[best_enemy]['distance']:.1f}m)")
            return best_enemy, advantage_details[best_enemy]
        else:
            logger.warning(f"No valid targets found for {ego_agent_id}")
            return None, None
            
    except Exception as e:
        logger.error(f"Error in target selection for {ego_agent_id}: {e}")
        return None, None

def update_observation_with_target(obs, env, step_count, center_lon=120.0, center_lat=60.0, center_alt=0.0):
    """
    更新观察数据，为每架我方飞机选择最优目标并更新敌方信息 - 简化版本
    每步都重新选择最优目标，允许多架飞机选择同一目标
    """
    try:
        # 获取我方飞机ID
        friendly_ids = [agent_id for agent_id in env.agents.keys() if agent_id.startswith('A') and env.agents[agent_id].is_alive]
        
        # 为每架我方飞机选择最优目标，允许多架飞机选择同一目标
        target_mapping = {}
        current_step_data = {
            'step': step_count,
            'timestamp': time.time(),
            'matches': []
        }
        
        # 为每架飞机独立选择最优目标
        for friendly_id in friendly_ids:
            target_id, advantage_info = select_best_target(friendly_id, env, center_lon, center_lat, center_alt)
            if target_id:
                target_mapping[friendly_id] = target_id
                
                # 记录匹配和优势信息
                match_data = {
                    'friendly_id': friendly_id,
                    'target_id': target_id,
                    'my_advantage': advantage_info['my_advantage'],
                    'enemy_advantage': advantage_info['enemy_advantage'],
                    'advantage_diff': advantage_info['advantage_diff'],
                    'adjusted_advantage_diff': advantage_info['adjusted_advantage_diff'],
                    'distance': advantage_info['distance']
                }
                current_step_data['matches'].append(match_data)
                
                logger.debug(f"{friendly_id} 选择目标: {target_id} (优势差值: {advantage_info['adjusted_advantage_diff']:.4f})")
        
        # 添加到全局历史记录
        if current_step_data['matches']:
            target_matching_history.append(current_step_data)
        
        # 更新观察数据
        updated_obs = obs.copy()
        
        for i, agent_id in enumerate(env.agents.keys()):
            if agent_id.startswith('A') and agent_id in target_mapping:
                # 获取目标敌方飞机
                target_id = target_mapping[agent_id]
                
                # 获取我方和目标的状态
                ego_agent = env.agents[agent_id]
                target_agent = env.agents[target_id]
                
                ego_state = np.array(ego_agent.get_property_values(env.task.state_var))
                target_state = np.array(target_agent.get_property_values(env.task.state_var))
                
                # 计算相对信息 (6维)
                ego_cur_ned = LLA2NEU(*ego_state[:3], center_lon, center_lat, center_alt)
                target_cur_ned = LLA2NEU(*target_state[:3], center_lon, center_lat, center_alt)
                
                ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])
                target_feature = np.array([*target_cur_ned, *(target_state[6:9])])
                
                AO, TA, R, side_flag = get_AO_TA_R(ego_feature, target_feature, return_side=True)
                
                # 构建正确的相对信息
                relative_info = np.array([
                    (target_state[9] - ego_state[9]) / 340,      # delta_v_body_x (unit: mh)
                    (target_state[2] - ego_state[2]) / 1000,    # delta_altitude (unit: km)
                    AO,                                          # ego_AO (unit: rad)
                    TA,                                          # ego_TA (unit: rad)
                    R / 10000,                                   # relative_distance (unit: 10km)
                    side_flag                                    # side_flag
                ])
                
                # 更新我方飞机的观察数据
                updated_obs[i] = np.concatenate([
                    obs[i][:9],  # 保持我方信息不变
                    relative_info  # 使用正确计算的相对信息
                ])
                
                logger.debug(f"Updated {agent_id} observation with target {target_id}")
        
        return updated_obs
        
    except Exception as e:
        logger.error(f"Error updating observations with targets: {e}")
        return obs

def plot_advantage_analysis():
    """绘制优势函数分析图表"""
    if not target_matching_history:
        logger.warning("No target matching history to plot")
        return
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表 - 2x3布局，每个子图代表一架我方飞机
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('6v6 分层1v1风格 - 各飞机优势函数值变化', fontsize=16, fontweight='bold')
    
    # 我方飞机ID列表
    friendly_ids = ['A0100', 'A0200', 'A0300', 'A0400', 'A0500', 'A0600']
    
    # 为每个我方飞机创建子图
    for i, friendly_id in enumerate(friendly_ids):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # 收集该飞机的所有优势数据
        my_advantages = []
        enemy_advantages = []
        steps = []
        
        # 从历史记录中提取该飞机的数据
        for step_data in target_matching_history:
            for match in step_data['matches']:
                if match['friendly_id'] == friendly_id:
                    my_advantages.append(match['my_advantage'])
                    enemy_advantages.append(match['enemy_advantage'])
                    steps.append(step_data['step'])
                    break
        
        if steps:  # 如果有数据才绘图
            # 绘制我方优势值（红色线）
            ax.plot(steps, my_advantages, color='red', linewidth=2, label='我方优势值', marker='o', markersize=3)
            
            # 绘制敌方优势值（蓝色线）
            ax.plot(steps, enemy_advantages, color='blue', linewidth=2, label='敌方优势值', marker='s', markersize=3)
            
            ax.set_title(f'{friendly_id} 优势函数值变化')
            ax.set_xlabel('步数')
            ax.set_ylabel('优势函数值')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 添加零线作为参考
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        else:
            ax.text(0.5, 0.5, f'{friendly_id}\n无数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{friendly_id}')
    
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"advantage_analysis_6v6_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    logger.info(f"优势函数分析图表已保存为: {plot_filename}")
    
    # 显示图表
    plt.show()
    
    # 打印简要统计信息
    print("\n=== 优势函数分析统计 ===")
    print(f"总步数: {len(target_matching_history)}")
    
    for friendly_id in friendly_ids:
        my_advs = []
        enemy_advs = []
        for step_data in target_matching_history:
            for match in step_data['matches']:
                if match['friendly_id'] == friendly_id:
                    my_advs.append(match['my_advantage'])
                    enemy_advs.append(match['enemy_advantage'])
                    break
        
        if my_advs:
            print(f"{friendly_id} - 我方优势值: 平均={np.mean(my_advs):.4f}, 敌方优势值: 平均={np.mean(enemy_advs):.4f}")
def plot_target_assignment():
    """绘制目标分配柱状图"""
    if not target_matching_history:
        logger.warning("No target matching history to plot")
        return
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表 - 2x3布局，每个子图代表一架我方飞机
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('6v6 分层1v1风格 - 各飞机目标分配情况', fontsize=16, fontweight='bold')
    
    # 我方飞机ID列表
    friendly_ids = ['A0100', 'A0200', 'A0300', 'A0400', 'A0500', 'A0600']
    
    # 为每个我方飞机创建子图
    for i, friendly_id in enumerate(friendly_ids):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # 收集该飞机的目标分配数据
        steps = []
        target_ids = []
        
        # 从历史记录中提取该飞机的目标分配数据
        for step_data in target_matching_history:
            for match in step_data['matches']:
                if match['friendly_id'] == friendly_id:
                    steps.append(step_data['step'])
                    target_ids.append(match['target_id'])
                    break
        
        if steps:  # 如果有数据才绘图
            # 将目标ID转换为数字便于绘图
            target_mapping = {'B0100': 1, 'B0200': 2, 'B0300': 3, 'B0400': 4, 'B0500': 5, 'B0600': 6}
            target_numbers = [target_mapping.get(target_id, 0) for target_id in target_ids]
            
            # 创建柱状图
            bars = ax.bar(steps, target_numbers, width=0.8, alpha=0.7, 
                         color=['red', 'blue', 'green', 'orange', 'purple', 'brown'][i], edgecolor='black', linewidth=0.5)
            
            ax.set_title(f'{friendly_id} 目标分配情况')
            ax.set_xlabel('步数')
            ax.set_ylabel('目标敌机编号')
            ax.set_ylim(0.5, 6.5)
            ax.set_yticks([1, 2, 3, 4, 5, 6])
            ax.set_yticklabels(['B0100', 'B0200', 'B0300', 'B0400', 'B0500', 'B0600'])
            ax.grid(True, alpha=0.3, axis='y')
            
            # 添加目标分配统计信息
            target_counts = {}
            for target_id in target_ids:
                target_counts[target_id] = target_counts.get(target_id, 0) + 1
            
            # 在子图右上角添加统计信息
            stats_text = f"目标分配统计:\n"
            for target_id, count in sorted(target_counts.items()):
                percentage = (count / len(target_ids)) * 100
                stats_text += f"{target_id}: {count}次 ({percentage:.1f}%)\n"
            
            ax.text(0.98, 0.98, stats_text.strip(), transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   fontsize=8)
            
        else:
            ax.text(0.5, 0.5, f'{friendly_id}\n无数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{friendly_id}')
    
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"target_assignment_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    logger.info(f"目标分配柱状图已保存为: {plot_filename}")
    
    # 显示图表
    plt.show()
    
    # 打印目标分配统计信息
    print("\n=== 目标分配统计 ===")
    print(f"总步数: {len(target_matching_history)}")
    
    for friendly_id in friendly_ids:
        target_counts = {}
        for step_data in target_matching_history:
            for match in step_data['matches']:
                if match['friendly_id'] == friendly_id:
                    target_id = match['target_id']
                    target_counts[target_id] = target_counts.get(target_id, 0) + 1
                    break
        
        if target_counts:
            print(f"{friendly_id} 目标分配:")
            for target_id, count in sorted(target_counts.items()):
                percentage = (count / sum(target_counts.values())) * 100
                print(f"  {target_id}: {count}次 ({percentage:.1f}%)")
        else:
            print(f"{friendly_id}: 无目标分配数据")
def main():
    # 6v6 分层1v1风格配置
    num_agents = 12  # 6架红方 + 6架蓝方
    render = True
    ego_policy_index = "1040"  # 使用1040模型
    enm_policy_index = "440"
    
    # 分层1v1模型路径 - 使用您提供的路径
    ego_run_dir = "scripts/results/SingleCombat/1v1/NoWeapon/HierarchySelfplay/ppo/v1/wandb/latest-run/files"
    enm_run_dir = ego_run_dir  # 使用同一个模型作为双方
    
    # 如果路径不存在，使用默认路径
    if not os.path.exists(ego_run_dir):
        logger.warning(f"Model path not found: {ego_run_dir}")
        ego_run_dir = "results/SingleCombat/1v1/NoWeapon/HierarchySelfplay/ppo/v1/wandb/latest-run/files"
    
    experiment_name = "6v6_hierarchy_1v1_blood"
    
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    experiment_name_with_timestamp = f"{experiment_name}_{timestamp}"
    
    # 创建6v6 分层1v1风格环境
    logger.info("Creating 6v6 hierarchical 1v1 style environment...")
    try:
        # 使用6v6的配置文件
        env = MultipleCombatEnv("6v6/NoWeapon/Hierarchy1v1Style")
        env.seed(0)
        logger.info(f"Environment created successfully. Num agents: {env.num_agents}")
        logger.info(f"Observation space: {env.observation_space}")
        logger.info(f"Action space: {env.action_space}")
        
        # 验证观察空间维度
        expected_obs_length = 15  # 9 + 6
        actual_obs_length = env.observation_space.shape[0]
        logger.info(f"Expected observation length: {expected_obs_length}")
        logger.info(f"Actual observation length: {actual_obs_length}")
        
        if actual_obs_length != expected_obs_length:
            logger.error(f"Observation space mismatch! Expected {expected_obs_length}, got {actual_obs_length}")
            return
            
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        return
    
    args = Args()
    
    # 创建策略网络
    logger.info("Creating policy networks...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        ego_policy = PPOActor(args, env.observation_space, env.action_space, device=device)
        enm_policy = PPOActor(args, env.observation_space, env.action_space, device=device)
        ego_policy.eval()
        enm_policy.eval()
        
        # 加载模型权重
        ego_model_path = os.path.join(ego_run_dir, f"actor_{ego_policy_index}.pt")
        enm_model_path = os.path.join(enm_run_dir, f"actor_{enm_policy_index}.pt")
        
        ego_loaded = load_model_safely(ego_model_path, ego_policy, device)
        enm_loaded = load_model_safely(enm_model_path, enm_policy, device)
        
        if not ego_loaded or not enm_loaded:
            logger.warning("Using random initialized policies")
            
    except Exception as e:
        logger.error(f"Failed to create policy networks: {e}")
        return
    
    logger.info("Starting 6v6 hierarchical 1v1 style render...")
    obs, _ = env.reset()
    
    # 初始目标选择
    obs = update_observation_with_target(obs, env, 0)
    
    if render:
        render_file = f'{experiment_name_with_timestamp}.txt.acmi'
        env.render(mode='txt', filepath=render_file)
        logger.info(f"Rendering to: {render_file}")
    
    # RNN状态初始化 - 修复维度问题
    ego_rnn_states = np.zeros((num_agents // 2, 1, 128), dtype=np.float32)  # (6, 1, 128)
    enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)  # (6, 1, 128)
    masks = np.ones((num_agents // 2, 1))  # 6架飞机的掩码
    
    # 观察数据切片：前6架为红方，后6架为蓝方
    enm_obs = obs[num_agents // 2:, :]  # 蓝方观察 (6, 15)
    ego_obs = obs[:num_agents // 2, :]  # 红方观察 (6, 15)
    
    episode_rewards = np.zeros((num_agents // 2, 1))
    step_count = 0
    
    try:
        while True:
            step_count += 1
            start = time.time()
            
            # 红方策略网络推理
            ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
            ego_actions = _t2n(ego_actions)
            ego_rnn_states = _t2n(ego_rnn_states)
            
            # 蓝方策略网络推理
            enm_actions, _, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states, masks, deterministic=True)
            enm_actions = _t2n(enm_actions)
            enm_rnn_states = _t2n(enm_rnn_states)
            
            # 合并动作
            actions = np.concatenate((ego_actions, enm_actions), axis=0)
            
            # 环境步进
            obs, _, rewards, dones, infos = env.step(actions)
            
            # 应用伤害系统
            damage_log = apply_damage_system(env)
            
            # 如果有伤害发生，打印详细信息
            if damage_log:
                logger.info(f"Step {step_count} - 伤害事件:")
                for damage_event in damage_log:
                    logger.info(f"  {damage_event['attacker']} -> {damage_event['target']}: "
                              f"伤害={damage_event['damage']:.4f}, "
                              f"距离={damage_event['distance']:.1f}m, "
                              f"类型={damage_event['type']}")
            
            # 更新目标选择
            obs = update_observation_with_target(obs, env, step_count)
            
            # 计算红方总奖励
            red_rewards = rewards[:num_agents // 2, ...]
            episode_rewards += red_rewards
            
            if render:
                env.render(mode='txt', filepath=render_file)
            
            if dones.all():
                logger.info(f"Episode finished at step {step_count}")
                logger.info(f"Episode info: {infos}")
                break
            
            # 检查是否所有飞机都被击落
            alive_agents = sum(1 for agent in env.agents.values() if agent.is_alive)
            if alive_agents == 0:
                logger.info(f"All aircraft destroyed at step {step_count}")
                break
            
            # 打印血量信息
            bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
            if step_count % 50 == 0:  # 每50步打印一次
                logger.info(f"Step: {step_count}")
                logger.info("飞机状态:")
                for agent_id in env.agents.keys():
                    agent = env.agents[agent_id]
                    status = "存活" if agent.is_alive else "击落"
                    logger.info(f"  {agent_id}: 血量={agent.bloods:.2f}, 状态={status}")
                logger.info(f"Red team rewards: {red_rewards.flatten()}")
                
                # 打印对应关系
                logger.info("Corresponding pairs:")
                for i in range(6):
                    red_id = f"A0{i+1}00"
                    blue_id = f"B0{i+1}00"
                    logger.info(f"  {red_id} <-> {blue_id}")
            
            # 更新观察数据
            enm_obs = obs[num_agents // 2:, ...]
            ego_obs = obs[:num_agents // 2, ...]
            
    except KeyboardInterrupt:
        logger.info("Render interrupted by user")
    except Exception as e:
        logger.error(f"Error during render: {e}")
    
    # 输出最终结果
    logger.info(f"Final episode rewards: {episode_rewards.flatten()}")
    logger.info(f"Average episode reward: {np.mean(episode_rewards):.4f}")
    logger.info(f"Total steps: {step_count}")
    
    # 输出最终飞机状态
    logger.info("Final aircraft status:")
    red_survivors = 0
    blue_survivors = 0
    for agent_id, agent in env.agents.items():
        status = "存活" if agent.is_alive else "击落"
        team = "红方" if agent_id.startswith('A') else "蓝方"
        if agent.is_alive:
            if agent_id.startswith('A'):
                red_survivors += 1
            else:
                blue_survivors += 1
        logger.info(f"  {agent_id} ({team}): 血量={agent.bloods:.2f}, 状态={status}")
    
    logger.info(f"红方幸存者: {red_survivors}/6, 蓝方幸存者: {blue_survivors}/6")
    if red_survivors > blue_survivors:
        logger.info("红方获胜!")
    elif blue_survivors > red_survivors:
        logger.info("蓝方获胜!")
    else:
        logger.info("平局!")
    
    # 验证对应关系
    logger.info("Verifying corresponding enemy relationships:")
    for agent_id in env.agents.keys():
        if agent_id.startswith('A'):
            corresponding_enemy = 'B' + agent_id[1:]
            logger.info(f"  {agent_id} -> {corresponding_enemy}")
        elif agent_id.startswith('B'):
            corresponding_enemy = 'A' + agent_id[1:]
            logger.info(f"  {agent_id} -> {corresponding_enemy}")
    
    # 绘制优势函数分析图表
    logger.info("Generating advantage function analysis plots...")
    plot_advantage_analysis()
    plot_target_assignment()

if __name__ == "__main__":
    main() 