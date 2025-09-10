import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def sigmoid(x):
    """Sigmoid函数"""
    return 1 / (1 + np.exp(-x))

def wp(R):
    """PostureReward权重函数"""
    return sigmoid(1.5 * (R - 3))

def we(R):
    """EvasionReward权重函数"""
    return 1 - wp(R)

def orientation_fn_v2(AO, TA):
    """PostureReward v2的orientation函数"""
    return 1 / (50 * AO / np.pi + 2) + 1 / 2 + min((np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5

def range_fn_v3(R):
    """PostureReward v3的range函数"""
    return 1 * (R < 5) + (R >= 5) * np.clip(-0.032 * R**2 + 0.284 * R + 0.38, 0, 1) + np.clip(np.exp(-0.16 * R), 0, 0.2)

def current_evasion_reward(AO, TA, R):
    """当前的EvasionReward函数 - 基于修改后的代码"""
    AO_deg = AO * 180.0 / np.pi
    TA_deg = TA * 180.0 / np.pi
    distance_km = R
    
    reward = 0.0
    
    # === 惩罚机制：面对面飞行 ===
    evasion_distance = 6.0
    face_to_face_angle_threshold = 30.0
    penalty_scale = 10.0
    
    is_face_to_face = (AO_deg < face_to_face_angle_threshold and 
                      TA_deg < face_to_face_angle_threshold)
    
    if is_face_to_face and distance_km <= evasion_distance:
        ao_factor = max(0, 1.0 - AO_deg / face_to_face_angle_threshold)
        ta_factor = max(0, 1.0 - TA_deg / face_to_face_angle_threshold)
        angle_factor = ao_factor * ta_factor
        distance_factor = max(0, 1.0 - distance_km / evasion_distance)
        penalty = penalty_scale * angle_factor * distance_factor
        reward -= penalty
    
    # === 奖励机制：AO≤20°且TA≥45° ===
    attack_ao_threshold = 20.0
    attack_ta_threshold = 45.0  # 注意：这里应该是45度，不是120度
    attack_reward_scale = 5.0
    
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

def dynamic_reward(AO, TA, R):
    """动态奖励函数"""
    # 计算权重
    wp_val = wp(R)
    we_val = we(R)
    
    # 计算各项奖励
    posture_reward = orientation_fn_v2(AO, TA) * range_fn_v3(R)
    evasion_reward_val = current_evasion_reward(AO, TA, R)
    
    # 组合奖励
    total_reward = wp_val * posture_reward + we_val * evasion_reward_val
    
    return total_reward, wp_val, we_val, posture_reward, evasion_reward_val

def visualize_current_reward():
    """可视化当前的奖励函数"""
    
    # 创建AO和TA的角度范围
    AO_range = np.linspace(0, np.pi, 50)
    TA_range = np.linspace(0, np.pi, 50)
    AO_grid, TA_grid = np.meshgrid(AO_range, TA_range)
    
    # 测试不同距离
    distances = [2, 4, 6]  # km
    
    fig = plt.figure(figsize=(20, 15))
    
    for i, R in enumerate(distances):
        # 计算各项奖励
        dynamic_reward_grid = np.zeros_like(AO_grid)
        posture_reward_grid = np.zeros_like(AO_grid)
        evasion_reward_grid = np.zeros_like(AO_grid)
        wp_grid = np.zeros_like(AO_grid)
        we_grid = np.zeros_like(AO_grid)
        
        for row in range(AO_grid.shape[0]):
            for col in range(AO_grid.shape[1]):
                total_r, wp_val, we_val, posture_r, evasion_r = dynamic_reward(
                    AO_grid[row, col], TA_grid[row, col], R
                )
                dynamic_reward_grid[row, col] = total_r
                posture_reward_grid[row, col] = posture_r
                evasion_reward_grid[row, col] = evasion_r
                wp_grid[row, col] = wp_val
                we_grid[row, col] = we_val
        
        # 1. 动态总奖励 - 3D表面图
        ax1 = fig.add_subplot(4, 5, i*5 + 1, projection='3d')
        surf = ax1.plot_surface(AO_grid * 180/np.pi, TA_grid * 180/np.pi, dynamic_reward_grid, 
                               cmap='viridis', alpha=0.8)
        ax1.set_xlabel('AO (degrees)')
        ax1.set_ylabel('TA (degrees)')
        ax1.set_zlabel('Dynamic Reward')
        ax1.set_title(f'Dynamic Total Reward at R={R}km')
        
        # 2. 动态总奖励 - 等高线图
        ax2 = fig.add_subplot(4, 5, i*5 + 2)
        contour = ax2.contour(AO_grid * 180/np.pi, TA_grid * 180/np.pi, dynamic_reward_grid, levels=15)
        ax2.clabel(contour, inline=True, fontsize=6)
        ax2.set_xlabel('AO (degrees)')
        ax2.set_ylabel('TA (degrees)')
        ax2.set_title(f'Dynamic Reward Contours at R={R}km')
        ax2.grid(True)
        
        # 标记关键区域
        ax2.axhline(y=45, color='red', linestyle='--', alpha=0.7, label='TA=45°')
        ax2.axvline(x=20, color='blue', linestyle='--', alpha=0.7, label='AO=20°')
        ax2.legend(fontsize=8)
        
        # 3. PostureReward
        ax3 = fig.add_subplot(4, 5, i*5 + 3)
        contour3 = ax3.contour(AO_grid * 180/np.pi, TA_grid * 180/np.pi, posture_reward_grid, levels=10)
        ax3.clabel(contour3, inline=True, fontsize=6)
        ax3.set_xlabel('AO (degrees)')
        ax3.set_ylabel('TA (degrees)')
        ax3.set_title(f'PostureReward at R={R}km')
        ax3.grid(True)
        
        # 4. EvasionReward
        ax4 = fig.add_subplot(4, 5, i*5 + 4)
        contour4 = ax4.contour(AO_grid * 180/np.pi, TA_grid * 180/np.pi, evasion_reward_grid, levels=10)
        ax4.clabel(contour4, inline=True, fontsize=6)
        ax4.set_xlabel('AO (degrees)')
        ax4.set_ylabel('TA (degrees)')
        ax4.set_title(f'EvasionReward at R={R}km')
        ax4.grid(True)
        
        # 标记关键区域
        ax4.axhline(y=45, color='red', linestyle='--', alpha=0.7, label='TA=45°')
        ax4.axvline(x=20, color='blue', linestyle='--', alpha=0.7, label='AO=20°')
        ax4.legend(fontsize=8)
        
        # 5. 权重分析
        ax5 = fig.add_subplot(4, 5, i*5 + 5)
        ax5.bar(['Posture', 'Evasion'], [wp(R), we(R)], color=['blue', 'red'], alpha=0.7)
        ax5.set_ylabel('Weight')
        ax5.set_title(f'Weights at R={R}km')
        ax5.set_ylim(0, 1)
        
        # 添加数值标签
        ax5.text(0, wp(R) + 0.05, f'{wp(R):.3f}', ha='center', va='bottom')
        ax5.text(1, we(R) + 0.05, f'{we(R):.3f}', ha='center', va='bottom')
    
    # 添加距离权重变化图
    ax_weight = fig.add_subplot(4, 1, 4)
    R_range = np.linspace(1, 10, 100)
    wp_values = [wp(r) for r in R_range]
    we_values = [we(r) for r in R_range]
    
    ax_weight.plot(R_range, wp_values, 'b-', linewidth=2, label='Posture Weight')
    ax_weight.plot(R_range, we_values, 'r-', linewidth=2, label='Evasion Weight')
    ax_weight.set_xlabel('Distance (km)')
    ax_weight.set_ylabel('Weight')
    ax_weight.set_title('Dynamic Weight Functions')
    ax_weight.grid(True)
    ax_weight.legend()
    ax_weight.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('current_reward_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_key_scenarios():
    """测试关键场景"""
    print("=== 当前奖励函数测试结果 ===")
    
    test_cases = [
        (0, 0, 3, "面对面接近"),
        (0, 180, 3, "理想攻击"),
        (180, 0, 3, "背对攻击"),
        (180, 180, 3, "背对背"),
        (10, 50, 3, "AO=10°, TA=50°"),
        (20, 45, 3, "AO=20°, TA=45°"),
        (25, 60, 3, "AO=25°, TA=60°"),
        (5, 30, 3, "AO=5°, TA=30°"),
        (15, 50, 2, "近距离攻击"),
        (15, 50, 10, "远距离攻击"),
    ]
    
    for AO_deg, TA_deg, distance_km, description in test_cases:
        AO_rad = AO_deg * np.pi / 180
        TA_rad = TA_deg * np.pi / 180
        
        total_r, wp_val, we_val, posture_r, evasion_r = dynamic_reward(AO_rad, TA_rad, distance_km)
        
        print(f"{description:12} (AO={AO_deg:3d}°, TA={TA_deg:3d}°, R={distance_km:2d}km):")
        print(f"  Total: {total_r:6.2f}, Posture: {posture_r:6.2f}, Evasion: {evasion_r:6.2f}")
        print(f"  Weights: wp={wp_val:.3f}, we={we_val:.3f}")
        print()

if __name__ == "__main__":
    test_key_scenarios()
    visualize_current_reward()
