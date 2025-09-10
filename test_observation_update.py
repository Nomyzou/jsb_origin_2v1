#!/usr/bin/env python3
"""
测试脚本：验证观察更新的正确性
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
from envs.JSBSim.envs import MultipleCombatEnv
from envs.JSBSim.utils.situation_assessment import get_situation_adv
from envs.JSBSim.utils.utils import LLA2NEU, get_AO_TA_R

def test_observation_update():
    """测试观察更新的正确性"""
    print("=== 观察更新测试 ===")
    
    # 创建环境
    env = MultipleCombatEnv('4v4/NoWeapon/Hierarchy1v1Style')
    
    # 重置环境
    obs, info = env.reset()
    
    # 获取状态变量
    state_var = env.task.state_var
    
    # 测试 A0100 的观察更新
    agent_id = 'A0100'
    target_id = 'B0100'  # 假设目标
    
    print(f"\n测试 {agent_id} 对 {target_id} 的观察更新:")
    
    # 获取原始观察
    original_obs = obs[list(env.agents.keys()).index(agent_id)]
    print(f"原始观察 (前9维): {original_obs[:9]}")
    print(f"原始观察 (后6维): {original_obs[9:]}")
    
    # 获取状态
    ego_agent = env.agents[agent_id]
    target_agent = env.agents[target_id]
    
    ego_state = np.array(ego_agent.get_property_values(state_var))
    target_state = np.array(target_agent.get_property_values(state_var))
    
    print(f"\n状态信息:")
    print(f"{agent_id} 高度: {ego_state[2]:.2f}, 速度: {ego_state[9]:.2f}")
    print(f"{target_id} 高度: {target_state[2]:.2f}, 速度: {target_state[9]:.2f}")
    
    # 计算相对信息
    center_lon, center_lat, center_alt = 120.0, 60.0, 0.0
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
    
    print(f"\n计算得到的相对信息:")
    print(f"速度差: {(target_state[9] - ego_state[9]):.2f} -> 归一化: {relative_info[0]:.4f}")
    print(f"高度差: {(target_state[2] - ego_state[2]):.2f} -> 归一化: {relative_info[1]:.4f}")
    print(f"AO: {AO:.4f}")
    print(f"TA: {TA:.4f}")
    print(f"距离: {R:.2f} -> 归一化: {relative_info[4]:.4f}")
    print(f"侧边标志: {side_flag}")
    
    # 构建更新后的观察
    updated_obs = np.concatenate([
        original_obs[:9],  # 保持我方信息不变
        relative_info      # 使用正确计算的相对信息
    ])
    
    print(f"\n更新后的观察 (后6维): {updated_obs[9:]}")
    
    # 验证观察维度
    print(f"\n观察维度验证:")
    print(f"原始观察维度: {len(original_obs)}")
    print(f"更新后观察维度: {len(updated_obs)}")
    print(f"前9维是否相同: {np.allclose(original_obs[:9], updated_obs[:9])}")
    
    env.close()

if __name__ == "__main__":
    test_observation_update() 