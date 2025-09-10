#!/usr/bin/env python3
"""
测试脚本：验证状态变量的获取情况
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
from envs.JSBSim.envs import MultipleCombatEnv
from envs.JSBSim.utils.situation_assessment import get_situation_adv

def test_state_variables():
    """测试状态变量的获取情况"""
    print("=== 状态变量测试 ===")
    
    # 创建环境
    env = MultipleCombatEnv('4v4/NoWeapon/Hierarchy1v1Style')
    
    # 获取状态变量定义
    state_var = env.task.state_var
    print(f"状态变量定义: {state_var}")
    print(f"状态变量数量: {len(state_var)}")
    
    # 获取所有飞机
    agents = env.agents
    print(f"飞机数量: {len(agents)}")
    
    # 测试每架飞机的状态获取
    for agent_id, agent in agents.items():
        print(f"\n--- {agent_id} ---")
        
        # 获取状态值
        state_values = agent.get_property_values(state_var)
        print(f"状态值数量: {len(state_values)}")
        
        # 打印关键状态变量
        print(f"经度 (索引0): {state_values[0]:.6f}")
        print(f"纬度 (索引1): {state_values[1]:.6f}")
        print(f"高度 (索引2): {state_values[2]:.2f}")
        print(f"横滚角 (索引3): {state_values[3]:.4f}")
        print(f"俯仰角 (索引4): {state_values[4]:.4f}")
        print(f"偏航角 (索引5): {state_values[5]:.4f}")
        print(f"北向速度 (索引6): {state_values[6]:.2f}")
        print(f"东向速度 (索引7): {state_values[7]:.2f}")
        print(f"下向速度 (索引8): {state_values[8]:.2f}")
        print(f"机体X速度 (索引9): {state_values[9]:.2f}")
        print(f"机体Y速度 (索引10): {state_values[10]:.2f}")
        print(f"机体Z速度 (索引11): {state_values[11]:.2f}")
        
        # 测试态势优势计算
        if agent_id.startswith('A'):  # 我方飞机
            print(f"\n计算 {agent_id} 的态势优势:")
            for enemy_id in agents.keys():
                if enemy_id.startswith('B'):  # 敌方飞机
                    try:
                        enemy_agent = agents[enemy_id]
                        enemy_state = enemy_agent.get_property_values(state_var)
                        
                        # 计算我方对敌方的优势
                        my_advantage = get_situation_adv(
                            state_values, enemy_state, 
                            center_lon=120.0, center_lat=60.0, center_alt=0.0
                        )
                        # 计算敌方对我方的优势
                        enemy_advantage = get_situation_adv(
                            enemy_state, state_values, 
                            center_lon=120.0, center_lat=60.0, center_alt=0.0
                        )
                        # 计算优势差值
                        advantage_diff = my_advantage - enemy_advantage
                        
                        print(f"  vs {enemy_id}: 我方优势={my_advantage:.4f}, 敌方优势={enemy_advantage:.4f}, 差值={advantage_diff:.4f}")
                    except Exception as e:
                        print(f"  vs {enemy_id}: 错误 - {e}")
    
    env.close()

if __name__ == "__main__":
    test_state_variables() 