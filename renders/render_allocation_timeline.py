import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from datetime import datetime
import time
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from envs.JSBSim.envs import MultipleCombatEnv
from algorithms.ppo.ppo_actor import PPOActor
from envs.JSBSim.utils.situation_assessment import get_situation_adv
from envs.JSBSim.utils.utils import LLA2NEU, get_AO_TA_R
from envs.JSBSim.core.catalog import Catalog as c

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

class AllocationTracker:
    """跟踪飞机分配情况"""
    def __init__(self):
        self.allocation_history = {}  # {step: {friendly_id: enemy_id}}
        self.advantage_history = {}   # {step: {agent_id: advantage_value}}
        self.friendly_ids = ['A0100', 'A0200', 'A0300', 'A0400']
        self.enemy_ids = ['B0100', 'B0200', 'B0300', 'B0400']
        
    def update_allocation(self, step, env):
        """更新当前步骤的分配情况"""
        current_allocation = {}
        current_advantages = {}
        
        for friendly_id in self.friendly_ids:
            if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                # 使用select_best_target函数获取真实的目标和优势函数值
                target_enemy, advantages = select_best_target_with_advantages(friendly_id, env)
                current_allocation[friendly_id] = target_enemy
                
                # 存储优势函数值
                if advantages:
                    current_advantages.update(advantages)
        
        self.allocation_history[step] = current_allocation
        self.advantage_history[step] = current_advantages

def select_best_target_with_advantages(ego_agent_id, env, center_lon=120.0, center_lat=60.0, center_alt=0.0):
    """
    选择最佳目标并返回优势函数值
    
    Args:
        ego_agent_id: 我方飞机ID
        env: 环境对象
        center_lon, center_lat, center_alt: 中心点坐标
        
    Returns:
        tuple: (best_enemy_id, advantages_dict)
        - best_enemy_id: 选择的最佳敌方飞机ID
        - advantages_dict: {agent_id: advantage_value} 所有相关飞机的优势函数值
    """
    if ego_agent_id not in env.agents:
        logger.warning(f"Agent {ego_agent_id} not found in environment")
        return None, {}
    
    ego_agent = env.agents[ego_agent_id]
    if not ego_agent.is_alive:
        logger.warning(f"Agent {ego_agent_id} is not alive")
        return None, {}
    
    # 获取敌方飞机列表
    enemy_ids = [aid for aid in env.agents.keys() if aid.startswith('B') and env.agents[aid].is_alive]
    if not enemy_ids:
        logger.warning(f"No enemy agents found for {ego_agent_id}")
        return None, {}
    
    # 定义状态变量
    state_var = ['position/longitude-deg', 'position/latitude-deg', 'position/altitude-ft',
                'velocities/airspeed-kt', 'velocities/vertical-speed-fps', 'attitude/roll-rad',
                'attitude/pitch-rad', 'attitude/heading-rad']
    
    # 计算双向态势优势
    advantage_differences = {}
    advantages_dict = {}
    
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
            
            # 存储优势函数值
            advantages_dict[ego_agent_id] = my_advantage
            advantages_dict[enemy_id] = enemy_advantage
            
            logger.debug(f"{ego_agent_id} vs {enemy_id}: my_adv={my_advantage:.4f}, enemy_adv={enemy_advantage:.4f}, diff={advantage_diff:.4f}")
            
        except Exception as e:
            logger.warning(f"Failed to calculate advantage for {ego_agent_id} vs {enemy_id}: {e}")
            advantage_differences[enemy_id] = -np.inf
    
    # 选择优势差值最大的敌方飞机
    if advantage_differences:
        best_enemy = max(advantage_differences.keys(), key=lambda x: advantage_differences[x])
        best_diff = advantage_differences[best_enemy]
        
        logger.info(f"{ego_agent_id} selected target: {best_enemy} (advantage_diff: {best_diff:.4f})")
        return best_enemy, advantages_dict
    else:
        logger.warning(f"No valid targets found for {ego_agent_id}")
        return None, advantages_dict
    
    def plot_allocation_timeline(self, save_path=None):
        """绘制分配时间图 - 4个子图显示优势函数"""
        if not self.allocation_history:
            logger.warning("No allocation history to plot")
            return
        
        # 创建2x2的子图布局
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()  # 将2x2数组展平为1维
        
        # 获取时间范围
        steps = sorted(self.allocation_history.keys())
        max_step = max(steps)
        
        # 为每个我方飞机绘制子图
        for i, friendly_id in enumerate(self.friendly_ids):
            ax = axes[i]
            
            # 获取该飞机的优势函数历史
            friendly_advantages = []
            target_advantages = []
            time_steps = []
            
            for step in steps:
                if friendly_id in self.allocation_history[step]:
                    current_allocation = self.allocation_history[step][friendly_id]
                    if current_allocation and step in self.advantage_history:
                        # 获取真实的优势函数值
                        friendly_adv = self.advantage_history[step].get(friendly_id, 0.0)
                        target_adv = self.advantage_history[step].get(current_allocation, 0.0)
                        
                        friendly_advantages.append(friendly_adv)
                        target_advantages.append(target_adv)
                        time_steps.append(step)
            
            if time_steps:
                # 绘制我方优势函数（红色线）
                ax.plot(time_steps, friendly_advantages, color='red', linewidth=2, 
                       label=f'{friendly_id} 优势函数', marker='o', markersize=4)
                
                # 绘制目标敌方优势函数（蓝色线）
                ax.plot(time_steps, target_advantages, color='blue', linewidth=2, 
                       label=f'目标敌方优势函数', marker='s', markersize=4)
            
            # 设置子图标题和标签
            ax.set_title(f'{friendly_id} 优势函数对比', fontsize=14, fontweight='bold')
            ax.set_xlabel('时间步', fontsize=12)
            ax.set_ylabel('优势函数值', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            
            # 添加当前目标信息
            if steps and friendly_id in self.allocation_history.get(steps[-1], {}):
                current_target = self.allocation_history[steps[-1]][friendly_id]
                if current_target:
                    ax.text(0.02, 0.98, f'当前目标: {current_target}', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                           fontsize=10)
        
        # 设置总标题
        fig.suptitle('飞机优势函数时间图\n每个子图显示一架我方飞机的优势函数与目标敌方优势函数对比', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Allocation timeline saved to {save_path}")
        
        plt.show()

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
                
                logger.debug(f"{ego_agent_id} vs {enemy_id}: my_adv={my_advantage:.4f}, enemy_adv={enemy_advantage:.4f}, diff={advantage_diff:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to calculate advantage for {ego_agent_id} vs {enemy_id}: {e}")
                advantage_differences[enemy_id] = -np.inf
        
        # 选择优势差值最大的敌方飞机
        if advantage_differences:
            best_enemy = max(advantage_differences.keys(), key=lambda x: advantage_differences[x])
            best_diff = advantage_differences[best_enemy]
            
            logger.info(f"{ego_agent_id} selected target: {best_enemy} (advantage_diff: {best_diff:.4f})")
            return best_enemy
        else:
            logger.warning(f"No valid targets found for {ego_agent_id}")
            return None
            
    except Exception as e:
        logger.error(f"Error in target selection for {ego_agent_id}: {e}")
        return None

def update_observation_with_target(obs, env, center_lon=120.0, center_lat=60.0, center_alt=0.0):
    """
    更新观察数据，为每架我方飞机选择最优目标并更新敌方信息
    """
    try:
        # 获取我方飞机ID
        friendly_ids = [agent_id for agent_id in env.agents.keys() if agent_id.startswith('A')]
        
        # 为每架我方飞机选择目标
        target_mapping = {}
        for friendly_id in friendly_ids:
            target_id = select_best_target(friendly_id, env, center_lon, center_lat, center_alt)
            if target_id:
                target_mapping[friendly_id] = target_id
        
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

def main():
    # 4v4 分层1v1风格配置
    num_agents = 8  # 4架红方 + 4架蓝方
    ego_policy_index = "latest"  # 使用latest模型
    enm_policy_index = "440"
    
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
    
    # 创建分配跟踪器
    allocation_tracker = AllocationTracker()
    
    logger.info("Starting 4v4 hierarchical 1v1 style render with allocation tracking...")
    obs, _ = env.reset()
    
    # 初始目标选择
    obs = update_observation_with_target(obs, env)
    
    # 设置acmi文件渲染
    acmi_file = f'{experiment_name_with_timestamp}.txt.acmi'
    env.render(mode='txt', filepath=acmi_file)
    logger.info(f"Rendering to ACMI file: {acmi_file}")
    
    # RNN状态初始化 - 修复维度问题
    ego_rnn_states = np.zeros((num_agents // 2, 1, 128), dtype=np.float32)  # (4, 1, 128)
    enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)  # (4, 1, 128)
    masks = np.ones((num_agents // 2, 1))  # 4架飞机的掩码
    
    # 观察数据切片：前4架为红方，后4架为蓝方
    enm_obs = obs[num_agents // 2:, :]  # 蓝方观察 (4, 15)
    ego_obs = obs[:num_agents // 2, :]  # 红方观察 (4, 15)
    
    episode_rewards = np.zeros((num_agents // 2, 1))
    step_count = 0
    max_steps = 1000  # 最大步数限制
    
    try:
        while step_count < max_steps:
            step_count += 1
            start = time.time()
            
            # 更新分配情况
            allocation_tracker.update_allocation(step_count, env)
            
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
            obs = update_observation_with_target(obs, env)
            
            # 计算红方总奖励
            red_rewards = rewards[:num_agents // 2, ...]
            episode_rewards += red_rewards
            
            # 渲染到acmi文件
            env.render(mode='txt', filepath=acmi_file)
            
            # 检查是否结束
            if dones.all():
                logger.info(f"Episode finished at step {step_count}")
                logger.info(f"Episode info: {infos}")
                break
            
            # 每100步打印一次进度
            if step_count % 100 == 0:
                logger.info(f"Step {step_count}/{max_steps}")
                
                # 打印血量信息
                bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
                logger.info(f"Bloods: {bloods}")
                logger.info(f"Red team rewards: {red_rewards.flatten()}")
                
                # 打印当前目标选择
                logger.info("Current target selections:")
                for agent_id in env.agents.keys():
                    if agent_id.startswith('A'):
                        # 这里可以添加目标选择的显示逻辑
                        pass
            
            # 更新观察数据
            enm_obs = obs[num_agents // 2:, ...]
            ego_obs = obs[:num_agents // 2, ...]
        
        # 绘制分配时间图
        output_path = f"allocation_timeline_{timestamp}.png"
        allocation_tracker.plot_allocation_timeline(save_path=output_path)
        
        # 输出最终结果
        logger.info(f"Final episode rewards: {episode_rewards.flatten()}")
        logger.info(f"Average episode reward: {np.mean(episode_rewards):.4f}")
        logger.info(f"Total steps: {step_count}")
        
        logger.info("Allocation tracking completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Render interrupted by user")
    except Exception as e:
        logger.error(f"Error during render: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 