import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def wp(R):
    return sigmoid(1.5 * (R - 3))

def we(R):
    return 1 - wp(R)

def orientation_fn_v2(AO, TA):
    """PostureReward v2的orientation函数"""
    return 1 / (50 * AO / np.pi + 2) + 1 / 2 + min((np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5

def range_fn_v3(R):
    """PostureReward v3的range函数"""
    return 1 * (R < 5) + (R >= 5) * np.clip(-0.032 * R**2 + 0.284 * R + 0.38, 0, 1) + np.clip(np.exp(-0.16 * R), 0, 0.2)

def evasion_reward(AO, TA):
    """EvasionReward函数 - 鼓励规避"""
    # 鼓励AO接近180度（背向敌机）
    ao_evasion = np.exp(-((AO - np.pi) / (np.pi/3))**2)
    
    # 鼓励TA接近180度（敌机背向我方）
    ta_evasion = np.exp(-((TA - np.pi) / (np.pi/3))**2)
    
    return ao_evasion * ta_evasion

def dynamic_reward(AO, TA, R):
    """动态奖励函数"""
    wp_val = wp(R)
    we_val = we(R)
    
    posture_reward = orientation_fn_v2(AO, TA) * range_fn_v3(R)
    evasion_reward_val = evasion_reward(AO, TA)
    
    total_reward = wp_val * posture_reward + we_val * evasion_reward_val
    
    return total_reward, wp_val, we_val, posture_reward, evasion_reward_val

def analyze_ideal_attack():
    """分析理想攻击姿态的奖励"""
    
    print("=== 理想攻击姿态分析 ===")
    print("AO=0°, TA=180° (我机攻击，敌机背向)")
    
    # 测试不同距离
    distances = [1, 2, 3, 4, 5]
    
    for R in distances:
        AO = 0  # 我机机头正对敌机
        TA = np.pi  # 敌机机尾正对我机
        
        total_r, wp_val, we_val, posture_r, evasion_r = dynamic_reward(AO, TA, R)
        
        print(f"\nR={R}km:")
        print(f"  权重: wp={wp_val:.3f}, we={we_val:.3f}")
        print(f"  PostureReward: {posture_r:.3f}")
        print(f"  EvasionReward: {evasion_r:.3f}")
        print(f"  总奖励: {total_r:.3f}")
    
    print("\n=== 对比其他姿态 ===")
    R = 2  # 近距离
    print(f"\n在R={R}km距离下:")
    
    # 测试不同姿态
    poses = [
        (0, 0, "面对面"),
        (0, np.pi, "理想攻击"),
        (np.pi, 0, "背对攻击"),
        (np.pi, np.pi, "背对背"),
        (np.pi/2, np.pi/2, "侧向")
    ]
    
    for AO, TA, name in poses:
        total_r, wp_val, we_val, posture_r, evasion_r = dynamic_reward(AO, TA, R)
        print(f"  {name} (AO={AO*180/np.pi:.0f}°, TA={TA*180/np.pi:.0f}°): {total_r:.3f}")

if __name__ == "__main__":
    analyze_ideal_attack()
