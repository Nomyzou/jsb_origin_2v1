import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def wp(R):
    return sigmoid(1.5 * (R - 3))

def we(R):
    return 1 - wp(R)

def fixed_posture_reward(AO, TA):
    """修正的PostureReward - 鼓励AO=0°, TA=180°"""
    ao_reward = np.exp(-(AO / (np.pi/6))**2)
    ta_reward = np.exp(-((TA - np.pi) / (np.pi/6))**2)
    return ao_reward * ta_reward

def fixed_evasion_reward(AO, TA):
    """修正的EvasionReward - 鼓励规避"""
    ao_evasion = np.exp(-((AO - np.pi) / (np.pi/6))**2)
    ta_evasion = np.exp(-((TA - np.pi) / (np.pi/6))**2)
    return ao_evasion * ta_evasion

def range_reward(R):
    """距离奖励"""
    if R < 2:
        return 0.5
    elif 2 <= R <= 5:
        return 1.0
    else:
        return 0.8

def fixed_dynamic_reward(AO, TA, R):
    """修正的动态奖励函数"""
    wp_val = wp(R)
    we_val = we(R)
    
    posture_reward = fixed_posture_reward(AO, TA) * range_reward(R)
    evasion_reward_val = fixed_evasion_reward(AO, TA) * range_reward(R)
    
    total_reward = wp_val * posture_reward + we_val * evasion_reward_val
    
    return total_reward, wp_val, we_val, posture_reward, evasion_reward_val

def visualize_fixed_reward():
    """可视化修正后的奖励函数"""
    
    # 创建AO和TA的角度范围
    AO_range = np.linspace(0, np.pi, 50)
    TA_range = np.linspace(0, np.pi, 50)
    AO_grid, TA_grid = np.meshgrid(AO_range, TA_range)
    
    # 测试不同距离
    distances = [2, 3, 4]  # km
    
    fig = plt.figure(figsize=(18, 12))
    
    for i, R in enumerate(distances):
        # 计算修正后的动态奖励
        dynamic_reward_grid = np.zeros_like(AO_grid)
        wp_grid = np.zeros_like(AO_grid)
        we_grid = np.zeros_like(AO_grid)
        
        for row in range(AO_grid.shape[0]):
            for col in range(AO_grid.shape[1]):
                total_r, wp_val, we_val, posture_r, evasion_r = fixed_dynamic_reward(
                    AO_grid[row, col], TA_grid[row, col], R
                )
                dynamic_reward_grid[row, col] = total_r
                wp_grid[row, col] = wp_val
                we_grid[row, col] = we_val
        
        # 3D表面图
        ax1 = fig.add_subplot(3, 4, i*4 + 1, projection='3d')
        surf = ax1.plot_surface(AO_grid * 180/np.pi, TA_grid * 180/np.pi, dynamic_reward_grid, 
                               cmap='viridis', alpha=0.8)
        ax1.set_xlabel('AO (度)')
        ax1.set_ylabel('TA (度)')
        ax1.set_zlabel('Dynamic Reward')
        ax1.set_title(f'Fixed Dynamic Reward at R={R}km')
        
        # 等高线图
        ax2 = fig.add_subplot(3, 4, i*4 + 2)
        contour = ax2.contour(AO_grid * 180/np.pi, TA_grid * 180/np.pi, dynamic_reward_grid, levels=15)
        ax2.clabel(contour, inline=True, fontsize=8)
        ax2.set_xlabel('AO (度)')
        ax2.set_ylabel('TA (度)')
        ax2.set_title(f'Fixed Dynamic Reward Contours at R={R}km')
        ax2.grid(True)
        
        # 权重变化
        ax3 = fig.add_subplot(3, 4, i*4 + 3)
        ax3.bar(['Posture', 'Evasion'], [wp(R), we(R)], color=['blue', 'red'], alpha=0.7)
        ax3.set_ylabel('Weight')
        ax3.set_title(f'Weights at R={R}km')
        ax3.set_ylim(0, 1)
        
        # 关键点分析
        ax4 = fig.add_subplot(3, 4, i*4 + 4)
        key_points = [
            (0, 0, 'Face-to-face'),
            (0, 180, 'Ideal Attack'),
            (180, 0, 'Back Attack'),
            (180, 180, 'Back-to-back'),
            (90, 90, 'Side')
        ]
        
        rewards = []
        labels = []
        for ao_deg, ta_deg, label in key_points:
            ao_rad = ao_deg * np.pi / 180
            ta_rad = ta_deg * np.pi / 180
            total_r, wp_val, we_val, posture_r, evasion_r = fixed_dynamic_reward(ao_rad, ta_rad, R)
            rewards.append(total_r)
            labels.append(f'{label}\n({ao_deg}°,{ta_deg}°)')
        
        bars = ax4.bar(range(len(rewards)), rewards, color='green', alpha=0.7)
        ax4.set_xticks(range(len(rewards)))
        ax4.set_xticklabels(labels, rotation=45, ha='right')
        ax4.set_ylabel('Reward')
        ax4.set_title(f'Key Points at R={R}km')
        
        # 添加数值标签
        for bar, reward in zip(bars, rewards):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{reward:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('fixed_dynamic_reward_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印策略分析
    print("\n=== 修正后的策略分析 ===")
    for r in [2, 3, 4]:
        wp_val = wp(r)
        we_val = we(r)
        
        # 分析理想攻击姿态的奖励
        ideal_attack_reward, _, _, _, _ = fixed_dynamic_reward(0, np.pi, r)
        
        if wp_val > 0.7:
            strategy = "攻击优先"
        elif we_val > 0.7:
            strategy = "规避优先"
        else:
            strategy = "平衡策略"
        
        print(f"R={r}km: {strategy} (wp={wp_val:.3f}, we={we_val:.3f})")
        print(f"  理想攻击姿态(AO=0°, TA=180°)奖励: {ideal_attack_reward:.3f}")

if __name__ == "__main__":
    visualize_fixed_reward()
