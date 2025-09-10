import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def orientation_fn_v2(AO, TA):
    """PostureReward v2的orientation函数"""
    return 1 / (50 * AO / np.pi + 2) + 1 / 2 + min((np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5

def range_fn_v3(R):
    """PostureReward v3的range函数"""
    return 1 * (R < 5) + (R >= 5) * np.clip(-0.032 * R**2 + 0.284 * R + 0.38, 0, 1) + np.clip(np.exp(-0.16 * R), 0, 0.2)

def analyze_posture_reward():
    """分析PostureReward的行为"""
    
    # 创建AO和TA的角度范围
    AO_range = np.linspace(0, np.pi, 100)  # 0到180度
    TA_range = np.linspace(0, np.pi, 100)  # 0到180度
    
    # 创建网格
    AO_grid, TA_grid = np.meshgrid(AO_range, TA_range)
    
    # 计算orientation奖励
    orientation_reward = np.zeros_like(AO_grid)
    for i in range(AO_grid.shape[0]):
        for j in range(AO_grid.shape[1]):
            orientation_reward[i, j] = orientation_fn_v2(AO_grid[i, j], TA_grid[i, j])
    
    # 创建图形
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 3D表面图
    ax1 = fig.add_subplot(221, projection='3d')
    surf = ax1.plot_surface(AO_grid * 180/np.pi, TA_grid * 180/np.pi, orientation_reward, 
                           cmap='viridis', alpha=0.8)
    ax1.set_xlabel('AO (度)')
    ax1.set_ylabel('TA (度)')
    ax1.set_zlabel('Orientation Reward')
    ax1.set_title('PostureReward v2: Orientation Function')
    
    # 2. 等高线图
    ax2 = fig.add_subplot(222)
    contour = ax2.contour(AO_grid * 180/np.pi, TA_grid * 180/np.pi, orientation_reward, levels=20)
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.set_xlabel('AO (度)')
    ax2.set_ylabel('TA (度)')
    ax2.set_title('Orientation Reward Contours')
    ax2.grid(True)
    
    # 3. AO=0时的TA奖励曲线
    ax3 = fig.add_subplot(223)
    TA_line = np.linspace(0, np.pi, 200)
    AO_zero_reward = [orientation_fn_v2(0, ta) for ta in TA_line]
    ax3.plot(TA_line * 180/np.pi, AO_zero_reward, 'b-', linewidth=2)
    ax3.set_xlabel('TA (度)')
    ax3.set_ylabel('Orientation Reward')
    ax3.set_title('AO=0°时的TA奖励曲线')
    ax3.grid(True)
    
    # 4. TA=0时的AO奖励曲线
    ax4 = fig.add_subplot(224)
    AO_line = np.linspace(0, np.pi, 200)
    TA_zero_reward = [orientation_fn_v2(ao, 0) for ao in AO_line]
    ax4.plot(AO_line * 180/np.pi, TA_zero_reward, 'r-', linewidth=2)
    ax4.set_xlabel('AO (度)')
    ax4.set_ylabel('Orientation Reward')
    ax4.set_title('TA=0°时的AO奖励曲线')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('posture_reward_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 分析关键点
    print("=== PostureReward v2 关键点分析 ===")
    print(f"AO=0°, TA=0° (面对面): {orientation_fn_v2(0, 0):.4f}")
    print(f"AO=0°, TA=90° (垂直): {orientation_fn_v2(0, np.pi/2):.4f}")
    print(f"AO=90°, TA=0° (侧向): {orientation_fn_v2(np.pi/2, 0):.4f}")
    print(f"AO=90°, TA=90° (背向): {orientation_fn_v2(np.pi/2, np.pi/2):.4f}")
    
    # 分析range函数
    print("\n=== Range Function v3 分析 ===")
    R_test = [1, 3, 5, 7, 10, 15, 20]
    for r in R_test:
        reward = range_fn_v3(r)
        print(f"R={r}km: {reward:.4f}")
    
    # 分析总奖励
    print("\n=== 总奖励分析 (Orientation × Range) ===")
    for ao_deg in [0, 30, 60, 90]:
        for ta_deg in [0, 30, 60, 90]:
            ao_rad = ao_deg * np.pi / 180
            ta_rad = ta_deg * np.pi / 180
            for r in [3, 5, 10]:
                orientation_r = orientation_fn_v2(ao_rad, ta_rad)
                range_r = range_fn_v3(r)
                total_r = orientation_r * range_r
                print(f"AO={ao_deg}°, TA={ta_deg}°, R={r}km: {total_r:.4f}")

if __name__ == "__main__":
    analyze_posture_reward()
