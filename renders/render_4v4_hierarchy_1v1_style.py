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
'''无伤害机制'''
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局变量用于记录数据
target_matching_history = []

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

def select_best_target(ego_agent_id, env, center_lon=120.0, center_lat=60.0, center_alt=0.0):
    """
    为指定我方飞机选择最优敌方目标
    通过计算双向态势优势函数，选择优势差值最大的敌方飞机
    """
    try:
        # 获取我方飞机对象
        ego_agent = env.agents[ego_agent_id]
        
        # 获取所有敌方飞机ID
        enemy_ids = []
        for agent_id in env.agents.keys():
            if agent_id.startswith('B'):  # 蓝方为敌方
                enemy_ids.append(agent_id)
        
        if not enemy_ids:
            logger.warning(f"No enemy agents found for {ego_agent_id}")
            return None
        
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
                advantage_differences[enemy_id] = advantage_diff
                advantage_details[enemy_id] = {
                    'my_advantage': my_advantage,
                    'enemy_advantage': enemy_advantage,
                    'advantage_diff': advantage_diff
                }
                
                logger.debug(f"{ego_agent_id} vs {enemy_id}: my_adv={my_advantage:.4f}, enemy_adv={enemy_advantage:.4f}, diff={advantage_diff:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to calculate advantage for {ego_agent_id} vs {enemy_id}: {e}")
                advantage_differences[enemy_id] = -np.inf
        
        # 选择优势差值最大的敌方飞机
        if advantage_differences:
            best_enemy = max(advantage_differences.keys(), key=lambda x: advantage_differences[x])
            best_diff = advantage_differences[best_enemy]
            
            logger.info(f"{ego_agent_id} selected target: {best_enemy} (advantage_diff: {best_diff:.4f})")
            return best_enemy, advantage_details[best_enemy]
        else:
            logger.warning(f"No valid targets found for {ego_agent_id}")
            return None, None
            
    except Exception as e:
        logger.error(f"Error in target selection for {ego_agent_id}: {e}")
        return None, None

def update_observation_with_target(obs, env, step_count, center_lon=120.0, center_lat=60.0, center_alt=0.0):
    """
    更新观察数据，为每架我方飞机选择最优目标并更新敌方信息
    """
    try:
        # 获取我方飞机ID
        friendly_ids = [agent_id for agent_id in env.agents.keys() if agent_id.startswith('A')]
        
        # 为每架我方飞机选择目标
        target_mapping = {}
        current_step_data = {
            'step': step_count,
            'timestamp': time.time(),
            'matches': []
        }
        
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
                    'advantage_diff': advantage_info['advantage_diff']
                }
                current_step_data['matches'].append(match_data)
        
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
    
    # 创建图表 - 2x2布局，每个子图代表一架我方飞机
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('4v4 分层1v1风格 - 各飞机优势函数值变化', fontsize=16, fontweight='bold')
    
    # 我方飞机ID列表
    friendly_ids = ['A0100', 'A0200', 'A0300', 'A0400']
    
    # 为每个我方飞机创建子图
    for i, friendly_id in enumerate(friendly_ids):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # 收集该飞机的所有优势数据
        my_advantages = []
        enemy_advantages = []
        my_avg_advantages = []  # 每一步的平均优势值
        enemy_avg_advantages = []  # 每一步的平均优势值
        steps = []
        
        # 从历史记录中提取该飞机的数据
        for step_data in target_matching_history:
            # 找到当前飞机的匹配数据
            current_match = None
            for match in step_data['matches']:
                if match['friendly_id'] == friendly_id:
                    current_match = match
                    break
            
            if current_match:
                my_advantages.append(current_match['my_advantage'])
                enemy_advantages.append(current_match['enemy_advantage'])
                steps.append(step_data['step'])
                
                # 计算当前步骤中所有我方飞机对所有敌方飞机的平均优势值
                all_my_advantages_this_step = []
                all_enemy_advantages_this_step = []
                
                for match in step_data['matches']:
                    all_my_advantages_this_step.append(match['my_advantage'])
                    all_enemy_advantages_this_step.append(match['enemy_advantage'])
                
                # 计算平均值
                if all_my_advantages_this_step:
                    my_avg_advantages.append(np.mean(all_my_advantages_this_step))
                    enemy_avg_advantages.append(np.mean(all_enemy_advantages_this_step))
        
        if steps:  # 如果有数据才绘图
            # 绘制我方优势值（红色线）
            ax.plot(steps, my_advantages, color='red', linewidth=2, label='我方优势值', marker='o', markersize=3)
            
            # 绘制敌方优势值（蓝色线）
            ax.plot(steps, enemy_advantages, color='blue', linewidth=2, label='敌方优势值', marker='s', markersize=3)
            
            # 绘制我方平均优势值（橙色虚线）
            ax.plot(steps, my_avg_advantages, color='orange', linestyle='--', linewidth=1.5, 
                   label='我方平均优势值', marker='^', markersize=2, alpha=0.7)
            
            # 绘制敌方平均优势值（浅蓝色虚线）
            ax.plot(steps, enemy_avg_advantages, color='lightblue', linestyle='--', linewidth=1.5, 
                   label='敌方平均优势值', marker='v', markersize=2, alpha=0.7)
            
            ax.set_title(f'{friendly_id} 优势函数值变化')
            ax.set_xlabel('步数')
            ax.set_ylabel('优势函数值')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # 添加零线作为参考
            ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)
        else:
            ax.text(0.5, 0.5, f'{friendly_id}\n无数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{friendly_id}')
    
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"advantage_analysis_{timestamp}.png"
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

def main():
    # 4v4 分层1v1风格配置
    num_agents = 8  # 4架红方 + 4架蓝方
    render = True
    ego_policy_index = "latest"  # 使用latest模型
    enm_policy_index = "latest"
    
    # 分层1v1模型路径 - 使用您提供的路径
    ego_run_dir = "scripts/results/SingleCombat/1v1/NoWeapon/HierarchySelfplay/ppo/v1/wandb/latest-run/files"
    enm_run_dir = ego_run_dir  # 使用同一个模型作为双方
    
    # 如果路径不存在，使用默认路径
    if not os.path.exists(ego_run_dir):
        logger.warning(f"Model path not found: {ego_run_dir}")
        ego_run_dir = "results/SingleCombat/1v1/NoWeapon/HierarchySelfplay/ppo/v1/wandb/latest-run/files"
    
    experiment_name = "4v4_hierarchy_1v1_style"
    
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name_with_timestamp = f"{experiment_name}_{timestamp}"
    
    # 创建4v4 分层1v1风格环境
    logger.info("Creating 4v4 hierarchical 1v1 style environment...")
    try:
        env = MultipleCombatEnv("4v4/NoWeapon/Hierarchy1v1Style")
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
    
    logger.info("Starting 4v4 hierarchical 1v1 style render...")
    obs, _ = env.reset()
    
    # 初始目标选择
    obs = update_observation_with_target(obs, env, 0)
    
    if render:
        render_file = f'{experiment_name_with_timestamp}.txt.acmi'
        env.render(mode='txt', filepath=render_file)
        logger.info(f"Rendering to: {render_file}")
    
    # RNN状态初始化 - 修复维度问题
    ego_rnn_states = np.zeros((num_agents // 2, 1, 128), dtype=np.float32)  # (4, 1, 128)
    enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)  # (4, 1, 128)
    masks = np.ones((num_agents // 2, 1))  # 4架飞机的掩码
    
    # 观察数据切片：前4架为红方，后4架为蓝方
    enm_obs = obs[num_agents // 2:, :]  # 蓝方观察 (4, 15)
    ego_obs = obs[:num_agents // 2, :]  # 红方观察 (4, 15)
    
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
            
            # 打印血量信息
            bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
            if step_count % 50 == 0:  # 每50步打印一次
                logger.info(f"Step: {step_count}, Bloods: {bloods}")
                logger.info(f"Red team rewards: {red_rewards.flatten()}")
                
                # 打印对应关系
                logger.info("Corresponding pairs:")
                for i in range(4):
                    red_id = f"A0{i+1}00"
                    blue_id = f"B0{i+1}00"
                    logger.info(f"  {red_id} <-> {blue_id}")
                
                # 打印当前目标选择
                logger.info("Current target selections:")
                for agent_id in env.agents.keys():
                    if agent_id.startswith('A'):
                        # 这里可以添加目标选择的显示逻辑
                        pass
            
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

if __name__ == "__main__":
    main() 