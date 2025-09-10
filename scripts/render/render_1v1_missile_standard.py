#!/usr/bin/env python3
"""
修复版本的1v1导弹战斗渲染脚本
使用标准导弹ID格式，确保TacView能正确显示
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from envs.env_wrappers import ShareDummyVecEnv, DummyVecEnv
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.JSBSim.core.simulatior import MissileSimulator

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Render 1v1 missile combat with standard missile IDs")
    group = parser.add_argument_group("render")
    group.add_argument('--env-name', type=str, default="SingleCombat",
                       help="environment name")
    group.add_argument('--scenario-name', type=str, default="1v1_missile",
                       help="scenario name")
    group.add_argument('--render-episodes', type=int, default=1,
                       help="number of episodes to render")
    group.add_argument('--episode-length', type=int, default=100,
                       help="episode length")
    group.add_argument('--output-dir', type=str, default='renders/missile_combat_standard',
                       help="output directory for rendered files")
    group.add_argument('--force-shoot', action='store_true',
                       help="force missile launch at step 10")
    group.add_argument('--shoot-step', type=int, default=10,
                       help="step to force missile launch when force-shoot is enabled")
    group.add_argument('--seed', type=int, default=42,
                       help="random seed")
    all_args = parser.parse_known_args(args)[0]
    return all_args

def make_render_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SingleCombat":
                env = SingleCombatEnv(all_args.scenario_name)
                # 如果启用强制发射，包装任务类
                if all_args.force_shoot:
                    class ForcedShootTask:
                        def __init__(self, original_task, shoot_step):
                            self.original_task = original_task
                            self.shoot_step = shoot_step
                            self.missile_counter = 0
                            
                        def __getattr__(self, name):
                            return getattr(self.original_task, name)
                        
                        def step(self, env):
                            # 调用原始任务的step方法
                            self.original_task.step(env)
                            
                            # 在指定步数强制发射导弹
                            if env.current_step == self.shoot_step:
                                logging.info(f"强制发射导弹，当前步数: {env.current_step}")
                                for agent_id, agent in env.agents.items():
                                    if agent.is_alive and hasattr(self.original_task, 'remaining_missiles') and self.original_task.remaining_missiles[agent_id] > 0:
                                        target = agent.enemies[0]
                                        # 使用标准的导弹ID格式
                                        self.missile_counter += 1
                                        # 格式: M + 发射者ID + 导弹编号
                                        new_missile_uid = f"M{agent_id}{self.missile_counter:02d}"
                                        missile = MissileSimulator.create(
                                            parent=agent,
                                            target=target,
                                            uid=new_missile_uid,
                                            missile_model="AIM-9L"
                                        )
                                        env.add_temp_simulator(missile)
                                        self.original_task.remaining_missiles[agent_id] -= 1
                                        logging.info(f"智能体 {agent_id} 发射导弹 {new_missile_uid}")
                    
                    env.task = ForcedShootTask(env.task, all_args.shoot_step)
            elif all_args.env_name == "SingleControl":
                env = SingleControlEnv(all_args.scenario_name)
            elif all_args.env_name == "MultipleCombat":
                env = MultipleCombatEnv(all_args.scenario_name)
            else:
                logging.error("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.env_name == "MultipleCombat":
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return DummyVecEnv([get_env_fn(0)])

def render_episode(env, episode_idx, output_dir, all_args):
    """渲染单个episode"""
    obs = env.reset()
    episode_reward = 0
    episode_length = 0
    
    # 创建输出目录
    episode_dir = Path(output_dir) / f"episode_{episode_idx + 1}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置ACMI渲染文件
    render_file = episode_dir / "episode_1.acmi"
    
    # 记录初始状态
    env.render(mode='txt', filepath=str(render_file))
    
    for step in range(all_args.episode_length):
        # 随机动作
        actions = []
        for i in range(env.num_envs):
            action = np.random.uniform(-1, 1, env.action_space[i].shape)
            actions.append(action)
        
        obs, rewards, dones, infos = env.step(actions)
        episode_reward += sum(rewards)
        episode_length += 1
        
        # 记录到ACMI文件
        env.render(mode='txt', filepath=str(render_file))
        
        # 检查是否结束
        if any(dones):
            break
    
    logging.info(f"Episode {episode_idx + 1} completed:")
    logging.info(f"  Length: {episode_length}")
    logging.info(f"  Total Reward: {episode_reward:.2f}")
    logging.info(f"  ACMI file: {render_file}")
    
    return episode_reward, episode_length

def main():
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 解析参数
    all_args = parse_args()
    
    # 创建输出目录
    output_dir = Path(all_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建环境
    env = make_render_env(all_args)
    
    # 渲染多个episode
    total_reward = 0
    total_length = 0
    
    for episode_idx in range(all_args.render_episodes):
        logging.info(f"开始渲染 Episode {episode_idx + 1}/{all_args.render_episodes}")
        
        episode_reward, episode_length = render_episode(
            env, episode_idx, output_dir, all_args
        )
        
        total_reward += episode_reward
        total_length += episode_length
    
    env.close()
    
    logging.info(f"渲染完成！")
    logging.info(f"总episode数: {all_args.render_episodes}")
    logging.info(f"平均奖励: {total_reward / all_args.render_episodes:.2f}")
    logging.info(f"平均长度: {total_length / all_args.render_episodes:.1f}")
    logging.info(f"输出目录: {output_dir}")
    
    # 提供TacView使用建议
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

if __name__ == "__main__":
    main() 