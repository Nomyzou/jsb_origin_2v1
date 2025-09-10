import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def wp(R):
    return sigmoid(1.5 * (R - 3))

def we(R):
    return 1 - wp(R)

def fixed_posture_reward(AO, TA):
    """修正的PostureReward - 鼓励AO=0°, TA=180°"""
    # AO奖励：鼓励机头对准敌机 (AO=0°)
    ao_reward = np.exp(-(AO / (np.pi/6))**2)  # AO=0时最大
    
    # TA奖励：鼓励敌机背向我方 (TA=180°)
    ta_reward = np.exp(-((TA - np.pi) / (np.pi/6))**2)  # TA=180°时最大
    
    return ao_reward * ta_reward

def fixed_evasion_reward(AO, TA):
    """修正的EvasionReward - 鼓励规避"""
    # AO奖励：鼓励背向敌机 (AO=180°)
    ao_evasion = np.exp(-((AO - np.pi) / (np.pi/6))**2)  # AO=180°时最大
    
    # TA奖励：鼓励敌机背向我方 (TA=180°)
    ta_evasion = np.exp(-((TA - np.pi) / (np.pi/6))**2)  # TA=180°时最大
    
    return ao_evasion * ta_evasion

def range_reward(R):
    """距离奖励"""
    if R < 2:
        return 0.5  # 过近
    elif 2 <= R <= 5:
        return 1.0  # 理想距离
    else:
        return 0.8  # 过远

def fixed_dynamic_reward(AO, TA, R):
    """修正的动态奖励函数"""
    wp_val = wp(R)
    we_val = we(R)
    
    posture_reward = fixed_posture_reward(AO, TA) * range_reward(R)
    evasion_reward_val = fixed_evasion_reward(AO, TA) * range_reward(R)
    
    total_reward = wp_val * posture_reward + we_val * evasion_reward_val
    
    return total_reward, wp_val, we_val, posture_reward, evasion_reward_val

def analyze_fixed_reward():
    """分析修正后的奖励函数"""
    
    print("=== 修正后的奖励函数分析 ===")
    print("理想攻击姿态: AO=0°, TA=180°")
    
    # 测试不同距离
    distances = [1, 2, 3, 4, 5]
    
    for R in distances:
        AO = 0  # 我机机头正对敌机
        TA = np.pi  # 敌机机尾正对我机
        
        total_r, wp_val, we_val, posture_r, evasion_r = fixed_dynamic_reward(AO, TA, R)
        
        print(f"\nR={R}km:")
        print(f"  权重: wp={wp_val:.3f}, we={we_val:.3f}")
        print(f"  PostureReward: {posture_r:.3f}")
        print(f"  EvasionReward: {evasion_r:.3f}")
        print(f"  总奖励: {total_r:.3f}")
    
    print("\n=== 对比不同姿态 ===")
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
        total_r, wp_val, we_val, posture_r, evasion_r = fixed_dynamic_reward(AO, TA, R)
        print(f"  {name} (AO={AO*180/np.pi:.0f}°, TA={TA*180/np.pi:.0f}°): {total_r:.3f}")

if __name__ == "__main__":
    analyze_fixed_reward()
