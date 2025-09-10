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
    """PostureReward v2的orientation函数 - 修复数值问题"""
    # 避免除零错误
    ao_term = 1 / (50 * AO / np.pi + 2) + 1 / 2
    
    # 修复TA项的计算
    ta_normalized = max(2 * TA / np.pi, 1e-6)  # 避免除零
    ta_term = min((np.arctanh(1. - ta_normalized)) / (2 * np.pi), 0.) + 0.5
    
    return ao_term + ta_term

def range_fn_v3(R):
    """PostureReward v3的range函数"""
    return 1 * (R < 5) + (R >= 5) * np.clip(-0.032 * R**2 + 0.284 * R + 0.38, 0, 1) + np.clip(np.exp(-0.16 * R), 0, 0.2)

def smooth_penalty_factor(angle_deg, threshold_deg):
    """平滑的惩罚因子计算"""
    if angle_deg >= threshold_deg:
        return 0.0
    # 使用平滑的衰减函数
    return np.exp(-2 * angle_deg / threshold_deg)

def smooth_reward_factor(angle_deg, threshold_deg, is_upper_bound=True):
    """平滑的奖励因子计算"""
    if is_upper_bound:
        # 对于上界（如TA），角度越大奖励越高
        if angle_deg < threshold_deg:
            return 0.0
        # 使用平滑的增长函数
        normalized = (angle_deg - threshold_deg) / (180.0 - threshold_deg)
        return np.tanh(2 * normalized)  # 使用tanh确保平滑
    else:
        # 对于下界（如AO），角度越小奖励越高
        if angle_deg > threshold_deg:
            return 0.0
        # 使用平滑的衰减函数
        normalized = angle_deg / threshold_deg
        return np.exp(-3 * normalized)

def smooth_evasion_reward(AO, TA, R):
    """平滑的EvasionReward函数"""
    AO_deg = AO * 180.0 / np.pi
    TA_deg = TA * 180.0 / np.pi
    distance_km = R
    
    reward = 0.0
    
    # === 平滑的惩罚机制：面对面飞行 ===
    evasion_distance = 8.0
    face_to_face_angle_threshold = 45.0
    penalty_scale = 5.0
    
    # 使用平滑函数计算面对面程度
    ao_penalty_factor = smooth_penalty_factor(AO_deg, face_to_face_angle_threshold)
    ta_penalty_factor = smooth_penalty_factor(TA_deg, face_to_face_angle_threshold)
    angle_penalty_factor = ao_penalty_factor * ta_penalty_factor
    
    # 平滑的距离因子
    if distance_km <= evasion_distance:
        distance_penalty_factor = np.exp(-distance_km / evasion_distance)
    else:
        distance_penalty_factor = 0.0
    
    # 计算惩罚值
    penalty = penalty_scale * angle_penalty_factor * distance_penalty_factor
    reward -= penalty
    
    # === 平滑的奖励机制：攻击姿态 ===
    attack_ao_threshold = 30.0
    attack_ta_threshold = 60.0
    attack_reward_scale = 3.0
    
    # 使用平滑函数计算攻击姿态的奖励
    ao_reward_factor = smooth_reward_factor(AO_deg, attack_ao_threshold, is_upper_bound=False)
    ta_reward_factor = smooth_reward_factor(TA_deg, attack_ta_threshold, is_upper_bound=True)
    angle_reward_factor = ao_reward_factor * ta_reward_factor
    
    # 平滑的距离因子
    if distance_km <= 2.0:
        distance_reward_factor = 0.5 + 0.5 * np.tanh(distance_km - 1.0)  # 过近时奖励递减
    elif distance_km <= 8.0:
        distance_reward_factor = 1.0  # 理想距离
    else:
        distance_reward_factor = np.exp(-0.2 * (distance_km - 8.0))  # 过远时奖励递减
    
    # 计算攻击奖励
    attack_reward = attack_reward_scale * angle_reward_factor * distance_reward_factor
    reward += attack_reward
    
    return reward, penalty, attack_reward

def dynamic_reward(AO, TA, R):
    """动态奖励函数"""
    # 计算权重
    wp_val = wp(R)
    we_val = we(R)
    
    # 计算各项奖励
    posture_reward = orientation_fn_v2(AO, TA) * range_fn_v3(R)
    evasion_reward_val, penalty, attack_reward = smooth_evasion_reward(AO, TA, R)
    
    # 组合奖励
    total_reward = wp_val * posture_reward + we_val * evasion_reward_val
    
    return total_reward, wp_val, we_val, posture_reward, evasion_reward_val

def test_smooth_reward():
    """测试平滑奖励函数"""
    print("=== 平滑奖励函数测试结果 ===")
    
    test_cases = [
        (0, 0, 3, "面对面接近"),
        (0, 180, 3, "理想攻击"),
        (180, 0, 3, "背对攻击"),
        (180, 180, 3, "背对背"),
        (10, 70, 3, "AO=10°, TA=70°"),
        (30, 60, 3, "AO=30°, TA=60°"),
        (45, 90, 3, "AO=45°, TA=90°"),
        (15, 45, 3, "AO=15°, TA=45°"),
        (20, 80, 2, "近距离攻击"),
        (20, 80, 10, "远距离攻击"),
    ]
    
    for AO_deg, TA_deg, distance_km, description in test_cases:
        AO_rad = AO_deg * np.pi / 180
        TA_rad = TA_deg * np.pi / 180
        
        total_r, wp_val, we_val, posture_r, evasion_r = dynamic_reward(AO_rad, TA_rad, distance_km)
        
        print(f"{description:12} (AO={AO_deg:3d}°, TA={TA_deg:3d}°, R={distance_km:2d}km):")
        print(f"  Total: {total_r:6.2f}, Posture: {posture_r:6.2f}, Evasion: {evasion_r:6.2f}")
        print(f"  Weights: wp={wp_val:.3f}, we={we_val:.3f}")
        print()

def visualize_smooth_reward():
    """可视化平滑奖励函数"""
    
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
        
        for row in range(AO_grid.shape[0]):
            for col in range(AO_grid.shape[1]):
                total_r, wp_val, we_val, posture_r, evasion_r = dynamic_reward(
                    AO_grid[row, col], TA_grid[row, col], R
                )
                dynamic_reward_grid[row, col] = total_r
                posture_reward_grid[row, col] = posture_r
                evasion_reward_grid[row, col] = evasion_r
        
        # 1. 动态总奖励 - 3D表面图
        ax1 = fig.add_subplot(4, 5, i*5 + 1, projection='3d')
        surf = ax1.plot_surface(AO_grid * 180/np.pi, TA_grid * 180/np.pi, dynamic_reward_grid, 
                               cmap='viridis', alpha=0.8)
        ax1.set_xlabel('AO (degrees)')
        ax1.set_ylabel('TA (degrees)')
        ax1.set_zlabel('Dynamic Reward')
        ax1.set_title(f'Smooth Dynamic Reward at R={R}km')
        
        # 2. 动态总奖励 - 等高线图
        ax2 = fig.add_subplot(4, 5, i*5 + 2)
        contour = ax2.contour(AO_grid * 180/np.pi, TA_grid * 180/np.pi, dynamic_reward_grid, levels=15)
        ax2.clabel(contour, inline=True, fontsize=6)
        ax2.set_xlabel('AO (degrees)')
        ax2.set_ylabel('TA (degrees)')
        ax2.set_title(f'Smooth Reward Contours at R={R}km')
        ax2.grid(True)
        
        # 标记关键区域
        ax2.axhline(y=60, color='red', linestyle='--', alpha=0.7, label='TA=60°')
        ax2.axvline(x=30, color='blue', linestyle='--', alpha=0.7, label='AO=30°')
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
        ax4.set_title(f'Smooth EvasionReward at R={R}km')
        ax4.grid(True)
        
        # 标记关键区域
        ax4.axhline(y=60, color='red', linestyle='--', alpha=0.7, label='TA=60°')
        ax4.axvline(x=30, color='blue', linestyle='--', alpha=0.7, label='AO=30°')
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
    plt.savefig('smooth_reward_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    test_smooth_reward()
    visualize_smooth_reward()
