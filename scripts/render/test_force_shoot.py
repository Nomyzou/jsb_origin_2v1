#!/usr/bin/env python3
"""
强制发射导弹的测试脚本
通过直接修改任务类来强制发射导弹
"""

import os
import sys
import torch
import random
import logging
import numpy as np
from pathlib import Path
import setproctitle
from datetime import datetime

# Deal with import error
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from config import get_config
from runner.share_jsbsim_runner import ShareJSBSimRunner
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import DummyVecEnv, ShareDummyVecEnv
from envs.JSBSim.core.simulatior import MissileSimulator


class ForcedShootTask:
    """强制发射导弹的任务包装器"""
    
    def __init__(self, original_task):
        self.original_task = original_task
        self.shoot_step = 10  # 在第10步发射导弹
        
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
                    new_missile_uid = f"M{agent_id[1:]}1"
                    missile = MissileSimulator.create(
                        parent=agent,
                        target=target,
                        uid=new_missile_uid,
                        missile_model="AIM-9L"
                    )
                    env.add_temp_simulator(missile)
                    self.original_task.remaining_missiles[agent_id] -= 1
                    logging.info(f"智能体 {agent_id} 发射导弹 {new_missile_uid}")


def make_render_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SingleCombat":
                env = SingleCombatEnv(all_args.scenario_name)
                # 强制发射导弹
                env.task = ForcedShootTask(env.task)
            elif all_args.env_name == "SingleControl":
                env = SingleControlEnv(all_args.scenario_name)
            elif all_args.env_name == "MultipleCombat":
                env = MultipleCombatEnv(all_args.scenario_name)
            else:
                logging.error(f"Unknown env_name: {all_args.env_name}")
                raise NotImplementedError
            return env
        return init_env
    return get_env_fn


def main():
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 解析参数
    parser = get_config()
    all_args = parser.parse_known_args()[0]
    
    # 设置环境名称
    all_args.env_name = "SingleCombat"
    all_args.scenario_name = "1v1/ShootMissile/HierarchySelfplay"
    
    # 创建环境
    env_fn = make_render_env(all_args)
    env = env_fn(0)()
    
    # 重置环境
    obs = env.reset()
    logging.info("环境初始化完成")
    
    # 运行几个步骤
    for step in range(20):
        # 正确的动作格式：numpy数组
        # action格式: [altitude, heading, velocity, shoot]
        # altitude: 0-2, heading: 0-4, velocity: 0-2, shoot: 0-1
        actions = np.array([
            [0, 2, 0, 1],  # A0100: 发射
            [0, 2, 0, 1]  # B0100: 发射
        ])
        obs, rewards, dones, infos = env.step(actions)
        
        # 检查导弹状态
        task = env.task
        remaining_missiles = getattr(task, 'remaining_missiles', {})
        logging.info(f"步数: {step}, 剩余导弹: {remaining_missiles}")
        
        if any(dones):
            break
    
    logging.info("测试完成")


if __name__ == "__main__":
    main() 