#!/usr/bin/env python3
"""
简化版本的1v1导弹战斗渲染脚本
使用训练好的模型进行控制，包括导弹发射决策
"""

import os
import sys
import logging
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from envs.JSBSim.envs import SingleCombatEnv
from envs.JSBSim.core.simulatior import MissileSimulator
from algorithms.ppo.ppo_actor import PPOActor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _t2n(x):
    """Tensor转numpy"""
    return x.detach().cpu().numpy()

class SimpleMissileRenderer:
    def __init__(self):
        # 基本配置
        self.num_agents = 2
        self.episode_length = 1000
        self.seed = 42
        
        # 模型路径
        self.model_path = "scripts/results/SingleCombat/1v1/ShootMissile/HierarchySelfplay/ppo/v1/wandb/latest-run/files/actor_480.pt"
        self.output_dir = "renders/missile_combat_simple"
        
        # 设备检测
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # 初始化环境和策略
        self._init_env()
        self._init_policies()
    
    def _init_env(self):
        """初始化环境"""
        self.env = SingleCombatEnv("1v1/ShootMissile/HierarchySelfplay")
        self.env.seed(self.seed)
        logging.info(f"环境初始化完成，设备: {self.device}")
    
    def _init_policies(self):
        """初始化策略网络"""
        # 策略参数
        class Args:
            def __init__(self, device):
                self.gain = 0.01
                self.hidden_size = '128 128'
                self.act_hidden_size = '128 128'
                self.activation_id = 1
                self.use_feature_normalization = False
                self.use_recurrent_policy = True
                self.recurrent_hidden_size = 128
                self.recurrent_hidden_layers = 1
                self.tpdv = dict(dtype=torch.float32, device=device)
                self.use_prior = True
        
        args = Args(self.device)
        
        # 创建策略网络
        self.ego_policy = PPOActor(args, self.env.observation_space, self.env.action_space, device=self.device)
        self.enm_policy = PPOActor(args, self.env.observation_space, self.env.action_space, device=self.device)
        
        # 加载模型
        if os.path.exists(self.model_path):
            self.ego_policy.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.enm_policy.load_state_dict(torch.load(self.model_path, map_location=self.device))
            logging.info(f"模型加载成功: {self.model_path}")
        else:
            logging.warning(f"模型文件不存在: {self.model_path}")
        
        self.ego_policy.eval()
        self.enm_policy.eval()
    
    def render_episode(self, episode_idx):
        """渲染单个episode"""
        # 创建输出目录
        episode_dir = Path(self.output_dir) / f"episode_{episode_idx + 1}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        
        # 重置环境
        obs = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        # 初始化RNN状态
        ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
        enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)
        masks = np.ones((self.num_agents // 2, 1))
        
        # 分离观察
        ego_obs = obs[:self.num_agents // 2, :]
        enm_obs = obs[self.num_agents // 2:, :]
        
        # 设置ACMI文件
        acmi_file = episode_dir / f"episode_{episode_idx + 1}.acmi"
        self.env.render(mode='txt', filepath=str(acmi_file))
        
        # 主循环
        for step in range(self.episode_length):
            # 获取动作（包含导弹发射决策）
            ego_actions, _, ego_rnn_states = self.ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
            enm_actions, _, enm_rnn_states = self.enm_policy(enm_obs, enm_rnn_states, masks, deterministic=True)
            # print(ego_actions, enm_actions)
            # 转换格式
            ego_actions = _t2n(ego_actions)
            enm_actions = _t2n(enm_actions)
            ego_rnn_states = _t2n(ego_rnn_states)
            enm_rnn_states = _t2n(enm_rnn_states)
            
            # 检查导弹发射条件并强制发射
            agent_ids = list(self.env.agents.keys())
            ego_id = agent_ids[0]
            enm_id = agent_ids[1]
            
            # 获取任务配置
            task = self.env.task
            max_attack_angle = getattr(task, 'max_attack_angle', 45)  # 默认45度
            max_attack_distance = getattr(task, 'max_attack_distance', 14000)  # 默认14000m
            min_attack_interval = getattr(task, 'min_attack_interval', 25)  # 默认25步
            
            # 检查每个智能体的发射条件
            for i, agent_id in enumerate([ego_id, enm_id]):
                agent = self.env.agents[agent_id]
                enemy = agent.enemies[0]  # 1v1只有一个敌人
                
                # 计算攻击角度和距离
                target = enemy.get_position() - agent.get_position()
                heading = agent.get_velocity()
                distance = np.linalg.norm(target)
                # 距离单位换算：观察空间中的距离单位是10km，需要转换为米
                # distance_meters = distance * 10000  # 转换为米
                attack_angle = np.rad2deg(np.arccos(np.clip(np.sum(target * heading) / (distance * np.linalg.norm(heading) + 1e-8), -1, 1)))
                
                # 检查发射间隔
                last_shoot_time = getattr(task, '_last_shoot_time', {}).get(agent_id, -min_attack_interval)
                shoot_interval = self.env.current_step - last_shoot_time
                
                # 每100步打印一次距离和角度信息
                if step % 100 == 0:
                    logging.info(f"步数 {step}: {agent_id} 距离:{distance:.0f}m, 角度:{attack_angle:.1f}°, 间隔:{shoot_interval}")
                
                # 检查是否满足发射条件
                can_shoot = (agent.is_alive and 
                           task.remaining_missiles[agent_id] > 0 and
                           attack_angle <= max_attack_angle and
                           distance <= max_attack_distance and
                           shoot_interval >= min_attack_interval)
                
                # 如果满足条件，强制设置发射动作为1
                if can_shoot:
                    if i == 0:  # ego
                        ego_actions[0][3] = 1.0
                        logging.info(f"步数 {step}: {agent_id} 满足发射条件，强制发射！角度:{attack_angle:.1f}°, 距离:{distance:.0f}m, 间隔:{shoot_interval}")
                    else:  # enm
                        enm_actions[0][3] = 1.0
                        logging.info(f"步数 {step}: {agent_id} 满足发射条件，强制发射！角度:{attack_angle:.1f}°, 距离:{distance:.0f}m, 间隔:{shoot_interval}")
            
            # 合并动作
            actions = np.concatenate((ego_actions, enm_actions), axis=0)
            
            # 打印动作信息（每10步打印一次）
            if step % 10 == 0:
                # 获取智能体ID
                agent_ids = list(self.env.agents.keys())
                ego_id = agent_ids[0]
                enm_id = agent_ids[1]
                
                # 解析动作 - 动作是4维向量：[flight_actions, shoot_action]
                ego_flight_actions = ego_actions[0][:3]  # 飞行控制动作 [altitude, heading, velocity]
                ego_shoot_action = ego_actions[0][3]     # 发射动作（第4维）
                enm_flight_actions = enm_actions[0][:3]  # 飞行控制动作
                enm_shoot_action = enm_actions[0][3]      # 发射动作（第4维）
                
                # 检查发射动作是否为布尔值
                ego_shoot_bool = bool(ego_shoot_action)
                enm_shoot_bool = bool(enm_shoot_action)
                
                # 检查任务配置
                task = self.env.task
                max_attack_angle = getattr(task, 'max_attack_angle', 'N/A')
                max_attack_distance = getattr(task, 'max_attack_distance', 'N/A')
                min_attack_interval = getattr(task, 'min_attack_interval', 'N/A')
                
                # 检查发射条件
                ego_agent = self.env.agents[ego_id]
                enm_agent = self.env.agents[enm_id]
                ego_is_alive = ego_agent.is_alive
                enm_is_alive = enm_agent.is_alive
                ego_missiles = getattr(task, 'remaining_missiles', {}).get(ego_id, 'N/A')
                enm_missiles = getattr(task, 'remaining_missiles', {}).get(enm_id, 'N/A')
                
                logging.info(f"步数: {step}")
                logging.info(f"  {ego_id} 动作: 飞行[{ego_flight_actions}], 发射[{ego_shoot_action}] (类型: {type(ego_shoot_action)}, 形状: {ego_shoot_action.shape if hasattr(ego_shoot_action, 'shape') else 'scalar'})")
                logging.info(f"  {enm_id} 动作: 飞行[{enm_flight_actions}], 发射[{enm_shoot_action}] (类型: {type(enm_shoot_action)}, 形状: {enm_shoot_action.shape if hasattr(enm_shoot_action, 'shape') else 'scalar'})")
                logging.info(f"  发射布尔值: {ego_id}[{ego_shoot_bool}], {enm_id}[{enm_shoot_bool}]")
                logging.info(f"  任务配置: max_attack_angle={max_attack_angle}, max_attack_distance={max_attack_distance}, min_attack_interval={min_attack_interval}")
                logging.info(f"  发射条件: {ego_id}[存活:{ego_is_alive}, 导弹:{ego_missiles}], {enm_id}[存活:{enm_is_alive}, 导弹:{enm_missiles}]")
            
            # 环境步进（导弹发射逻辑在task.step()中处理）
            obs, rewards, dones, infos = self.env.step(actions)
            episode_reward += sum(rewards[:self.num_agents // 2])
            episode_length += 1
            
            # 立即检查导弹数量变化
            if step % 10 == 0:
                bloods = [self.env.agents[agent_id].bloods for agent_id in self.env.agents.keys()]
                remaining_missiles = [self.env.task.remaining_missiles[agent_id] for agent_id in self.env.agents.keys()]
                logging.info(f"  血量: {bloods}, 剩余导弹: {remaining_missiles}")
                logging.info("-" * 50)
            
            # 渲染到ACMI文件
            self.env.render(mode='txt', filepath=str(acmi_file))
            
            # 检查是否结束
            if any(dones):
                break
            
            # 更新观察
            ego_obs = obs[:self.num_agents // 2, :]
            enm_obs = obs[self.num_agents // 2:, :]
        
        logging.info(f"Episode {episode_idx + 1} 完成:")
        logging.info(f"  长度: {episode_length}")
        logging.info(f"  总奖励: {float(episode_reward):.2f}")
        logging.info(f"  ACMI文件: {acmi_file}")
        
        return episode_reward, episode_length
    
    def render(self, num_episodes=1):
        """渲染多个episode"""
        logging.info(f"开始渲染 {num_episodes} 个episode")
        logging.info("使用训练好的模型进行控制，包括导弹发射决策")
        
        total_reward = 0
        total_length = 0
        
        for episode_idx in range(num_episodes):
            logging.info(f"渲染 Episode {episode_idx + 1}/{num_episodes}")
            episode_reward, episode_length = self.render_episode(episode_idx)
            total_reward += episode_reward
            total_length += episode_length
        
        logging.info(f"渲染完成！")
        logging.info(f"总episode数: {num_episodes}")
        logging.info(f"平均奖励: {float(total_reward) / num_episodes:.2f}")
        logging.info(f"平均长度: {total_length / num_episodes:.1f}")
        logging.info(f"输出目录: {self.output_dir}")
        
        # 提供TacView使用建议
        self._print_tacview_tips()
    
    def _print_tacview_tips(self):
        """打印TacView使用建议"""
        print("\n" + "=" * 60)
        print("TacView使用建议")
        print("=" * 60)
        print("1. 下载并安装 TacView: https://www.tacview.net/")
        print("2. 打开生成的 .acmi 文件")
        print("3. 在TacView中确保以下设置已启用:")
        print("   - View -> Objects -> Missiles")
        print("   - View -> Objects -> Explosions")
        print("   - View -> Objects -> Aircraft")
        print("4. 如果看不到导弹，尝试:")
        print("   - 重新加载文件")
        print("   - 检查对象过滤器设置")
        print("   - 使用不同的视图模式")
        print("5. 注意：导弹发射完全由训练好的模型决定")

def main():
    """主函数"""
    # 创建渲染器
    renderer = SimpleMissileRenderer()
    
    # 渲染1个episode
    renderer.render(num_episodes=1)

if __name__ == "__main__":
    main() 