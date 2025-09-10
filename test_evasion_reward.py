import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def test_evasion_reward():
    """测试修改后的EvasionReward"""
    
    # 模拟EvasionReward的参数
    evasion_distance = 6.0
    face_to_face_angle_threshold = 30.0
    penalty_scale = 10.0
    attack_ao_threshold = 20.0
    attack_ta_threshold = 120.0
    attack_reward_scale = 5.0
    
    def calculate_evasion_reward(AO_deg, TA_deg, distance_km):
        """计算EvasionReward"""
        reward = 0.0
        
        # === 惩罚机制：面对面飞行 ===
        is_face_to_face = (AO_deg < face_to_face_angle_threshold and 
                          TA_deg < face_to_face_angle_threshold)
        
        if is_face_to_face and distance_km <= evasion_distance:
            ao_factor = max(0, 1.0 - AO_deg / face_to_face_angle_threshold)
            ta_factor = max(0, 1.0 - TA_deg / face_to_face_angle_threshold)
            angle_factor = ao_factor * ta_factor
            distance_factor = max(0, 1.0 - distance_km / evasion_distance)
            penalty = penalty_scale * angle_factor * distance_factor
            reward -= penalty
        
        # === 奖励机制：AO≤20°且TA≥120° ===
        is_good_attack_angle = (AO_deg <= attack_ao_threshold and 
                               TA_deg >= attack_ta_threshold)
        
        if is_good_attack_angle:
            attack_ao_factor = max(0, 1.0 - AO_deg / attack_ao_threshold)
            attack_ta_factor = max(0, (TA_deg - attack_ta_threshold) / (180.0 - attack_ta_threshold))
            
            if 2.0 <= distance_km <= 8.0:
                attack_distance_factor = 1.0
            elif distance_km < 2.0:
                attack_distance_factor = 0.5 + 0.5 * (distance_km / 2.0)
            else:
                attack_distance_factor = max(0.3, 1.0 - 0.1 * (distance_km - 8.0))
            
            attack_reward = attack_reward_scale * attack_ao_factor * attack_ta_factor * attack_distance_factor
            reward += attack_reward
        
        return reward
    
    # 测试不同姿态
    print("=== EvasionReward 测试结果 ===")
    test_cases = [
        (0, 0, 3, "面对面接近"),
        (0, 180, 3, "理想攻击"),
        (180, 0, 3, "背对攻击"),
        (180, 180, 3, "背对背"),
        (10, 130, 3, "AO=10°, TA=130°"),
        (20, 120, 3, "AO=20°, TA=120°"),
        (25, 150, 3, "AO=25°, TA=150°"),
        (5, 90, 3, "AO=5°, TA=90°"),
        (15, 140, 2, "近距离攻击"),
        (15, 140, 10, "远距离攻击"),
    ]
    
    for AO_deg, TA_deg, distance_km, description in test_cases:
        reward = calculate_evasion_reward(AO_deg, TA_deg, distance_km)
        print(f"{description:12} (AO={AO_deg:3d}°, TA={TA_deg:3d}°, R={distance_km:2d}km): {reward:6.2f}")
    
    # 可视化奖励函数
    visualize_evasion_reward(calculate_evasion_reward)

def visualize_evasion_reward(calc_func):
    """可视化EvasionReward"""
    
    # 创建AO和TA的角度范围
    AO_range = np.linspace(0, 180, 50)
    TA_range = np.linspace(0, 180, 50)
    AO_grid, TA_grid = np.meshgrid(AO_range, TA_range)
    
    # 测试不同距离
    distances = [2, 4, 6]  # km
    
    fig = plt.figure(figsize=(18, 12))
    
    for i, distance_km in enumerate(distances):
        # 计算奖励
        reward_grid = np.zeros_like(AO_grid)
        
        for row in range(AO_grid.shape[0]):
            for col in range(AO_grid.shape[1]):
                reward_grid[row, col] = calc_func(AO_grid[row, col], TA_grid[row, col], distance_km)
        
        # 3D表面图
        ax1 = fig.add_subplot(3, 4, i*4 + 1, projection='3d')
        surf = ax1.plot_surface(AO_grid, TA_grid, reward_grid, cmap='RdYlBu_r', alpha=0.8)
        ax1.set_xlabel('AO (度)')
        ax1.set_ylabel('TA (度)')
        ax1.set_zlabel('EvasionReward')
        ax1.set_title(f'EvasionReward at R={distance_km}km')
        
        # 等高线图
        ax2 = fig.add_subplot(3, 4, i*4 + 2)
        contour = ax2.contour(AO_grid, TA_grid, reward_grid, levels=20)
        ax2.clabel(contour, inline=True, fontsize=8)
        ax2.set_xlabel('AO (度)')
        ax2.set_ylabel('TA (度)')
        ax2.set_title(f'EvasionReward Contours at R={distance_km}km')
        ax2.grid(True)
        
        # 标记关键区域
        ax2.axhline(y=120, color='red', linestyle='--', alpha=0.7, label='TA=120°')
        ax2.axvline(x=20, color='blue', linestyle='--', alpha=0.7, label='AO=20°')
        ax2.legend()
        
        # 关键点分析
        ax3 = fig.add_subplot(3, 4, i*4 + 3)
        key_points = [
            (0, 0, 'Face-to-face'),
            (0, 180, 'Ideal Attack'),
            (180, 0, 'Back Attack'),
            (180, 180, 'Back-to-back'),
            (10, 130, 'AO=10°, TA=130°'),
            (20, 120, 'AO=20°, TA=120°'),
            (25, 150, 'AO=25°, TA=150°'),
            (5, 90, 'AO=5°, TA=90°')
        ]
        
        rewards = []
        labels = []
        for ao_deg, ta_deg, label in key_points:
            reward = calc_func(ao_deg, ta_deg, distance_km)
            rewards.append(reward)
            labels.append(f'{label}\n({ao_deg}°,{ta_deg}°)')
        
        colors = ['red' if r < 0 else 'green' for r in rewards]
        bars = ax3.bar(range(len(rewards)), rewards, color=colors, alpha=0.7)
        ax3.set_xticks(range(len(rewards)))
        ax3.set_xticklabels(labels, rotation=45, ha='right')
        ax3.set_ylabel('Reward')
        ax3.set_title(f'Key Points at R={distance_km}km')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 添加数值标签
        for bar, reward in zip(bars, rewards):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{reward:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 奖励分布统计
        ax4 = fig.add_subplot(3, 4, i*4 + 4)
        flat_rewards = reward_grid.flatten()
        ax4.hist(flat_rewards, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.set_xlabel('Reward Value')
        ax4.set_ylabel('Frequency')
        ax4.set_title(f'Reward Distribution at R={distance_km}km')
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero Line')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig('evasion_reward_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    test_evasion_reward()
