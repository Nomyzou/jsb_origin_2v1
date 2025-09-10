import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
import matplotlib.colors as mcolors

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

class PostureReward:
    def __init__(self):
        self.target_dist = 3.0
        self.orientation_fn = self.get_orientation_function('v2')
        self.range_fn = self.get_range_function('v3')
    
    def get_orientation_function(self, version):
        if version == 'v2':
            def orientation_func(AO, TA):
                # 修复TA=180°时的无穷大问题
                ta_term = 2 * TA / np.pi
                if ta_term >= 1.0:
                    ta_term = 0.999  # 避免arctanh(0)的无穷大
                return 1 / (50 * AO / np.pi + 2) + 1 / 2 \
                    + min((np.arctanh(1. - max(ta_term, 1e-4))) / (2 * np.pi), 0.) + 0.5
            return orientation_func
        else:
            raise NotImplementedError(f"Unknown orientation function version: {version}")
    
    def get_range_function(self, version):
        if version == 'v3':
            return lambda R: 1 * (R < 5) + (R >= 5) * np.clip(-0.032 * R**2 + 0.284 * R + 0.38, 0, 1) + np.clip(np.exp(-0.16 * R), 0, 0.2)
        else:
            raise NotImplementedError(f"Unknown range function version: {version}")
    
    def get_reward(self, AO, TA, R):
        """计算PostureReward"""
        distance_km = R / 1000
        orientation_reward = self.orientation_fn(AO, TA)
        range_reward = self.range_fn(distance_km)
        new_reward = orientation_reward * range_reward
        
        # 应用距离缩放
        if distance_km <= 6.0:
            scale_factor = 0.4 + 0.6 * (distance_km / 6.0)
            new_reward *= scale_factor
        
        return new_reward

class EvasionReward:
    def __init__(self):
        self.evasion_distance = 8.0
        self.face_to_face_angle_threshold = 45.0
        self.penalty_scale = 5.0
        self.attack_ao_threshold = 30.0
        self.attack_ta_threshold = 60.0
        self.attack_reward_scale = 3.0
        self.back_turn_ta_threshold = 20.0
        self.back_turn_ao_threshold = 90.0
        self.back_turn_penalty_scale = 2.0
        self.reward_scale = 1.0/3.0  # 缩放到1/3
    
    def smooth_penalty_factor(self, angle_deg, threshold_deg):
        if angle_deg >= threshold_deg:
            return 0.0
        return np.exp(-2 * angle_deg / threshold_deg)
    
    def smooth_reward_factor(self, angle_deg, threshold_deg, is_upper_bound=True):
        if is_upper_bound:
            if angle_deg < threshold_deg:
                return 0.0
            normalized = (angle_deg - threshold_deg) / (180.0 - threshold_deg)
            return np.tanh(2 * normalized)
        else:
            if angle_deg > threshold_deg:
                return 0.0
            normalized = angle_deg / threshold_deg
            return np.exp(-3 * normalized)
    
    def smooth_back_turn_penalty_factor(self, ta_deg, ao_deg):
        # TA因子：TA越小惩罚越重
        if ta_deg >= self.back_turn_ta_threshold:
            ta_factor = 0.0
        else:
            ta_factor = np.exp(-ta_deg / self.back_turn_ta_threshold)
        
        # AO因子：AO越大惩罚越重
        if ao_deg <= self.back_turn_ao_threshold:
            ao_factor = 0.0
        else:
            normalized = (ao_deg - self.back_turn_ao_threshold) / (180.0 - self.back_turn_ao_threshold)
            ao_factor = np.tanh(2 * normalized)
        
        return ta_factor * ao_factor
    
    def get_reward(self, AO, TA, R):
        """计算EvasionReward"""
        AO_deg = AO * 180.0 / np.pi
        TA_deg = TA * 180.0 / np.pi
        distance_km = R / 1000.0
        
        reward = 0.0
        
        # 惩罚机制：面对面飞行
        ao_penalty_factor = self.smooth_penalty_factor(AO_deg, self.face_to_face_angle_threshold)
        ta_penalty_factor = self.smooth_penalty_factor(TA_deg, self.face_to_face_angle_threshold)
        angle_penalty_factor = ao_penalty_factor * ta_penalty_factor
        
        if distance_km <= self.evasion_distance:
            distance_penalty_factor = np.exp(-distance_km / self.evasion_distance)
        else:
            distance_penalty_factor = 0.0
        
        penalty = self.penalty_scale * angle_penalty_factor * distance_penalty_factor
        reward -= penalty
        
        # 奖励机制：攻击姿态
        ao_reward_factor = self.smooth_reward_factor(AO_deg, self.attack_ao_threshold, is_upper_bound=False)
        ta_reward_factor = self.smooth_reward_factor(TA_deg, self.attack_ta_threshold, is_upper_bound=True)
        angle_reward_factor = ao_reward_factor * ta_reward_factor
        
        if distance_km <= 2.0:
            distance_reward_factor = 0.5 + 0.5 * np.tanh(distance_km - 1.0)
        elif distance_km <= 8.0:
            distance_reward_factor = 1.0
        else:
            distance_reward_factor = np.exp(-0.2 * (distance_km - 8.0))
        
        attack_reward = self.attack_reward_scale * angle_reward_factor * distance_reward_factor
        reward += attack_reward
        
        # 背对惩罚机制
        back_turn_factor = self.smooth_back_turn_penalty_factor(TA_deg, AO_deg)
        
        if distance_km <= 5.0:
            back_turn_distance_factor = np.exp(-distance_km / 5.0)
        else:
            back_turn_distance_factor = 0.0
        
        back_turn_penalty = self.back_turn_penalty_scale * back_turn_factor * back_turn_distance_factor
        reward -= back_turn_penalty
        
        # 应用整体奖励缩放
        reward *= self.reward_scale
        
        return reward

def plot_3d_reward_surface_for_distance(distance_km):
    """为指定距离绘制3D奖励函数曲面"""
    posture_reward = PostureReward()
    evasion_reward = EvasionReward()
    
    # 创建图形
    fig = plt.figure(figsize=(20, 15))
    
    distance_m = distance_km * 1000
    
    # 创建角度网格
    ao_range = np.linspace(0, 180, 50)
    ta_range = np.linspace(0, 180, 50)
    AO_grid, TA_grid = np.meshgrid(ao_range, ta_range)
    
    # 计算奖励值
    posture_reward_grid = np.zeros_like(AO_grid)
    evasion_reward_grid = np.zeros_like(AO_grid)
    total_reward_grid = np.zeros_like(AO_grid)
    
    for i, ao in enumerate(ao_range):
        for j, ta in enumerate(ta_range):
            ao_rad = ao * np.pi / 180
            ta_rad = ta * np.pi / 180
            
            posture_r = posture_reward.get_reward(ao_rad, ta_rad, distance_m)
            evasion_r = evasion_reward.get_reward(ao_rad, ta_rad, distance_m)
            
            posture_reward_grid[j, i] = posture_r
            evasion_reward_grid[j, i] = evasion_r
            total_reward_grid[j, i] = posture_r + evasion_r
    
    # 1. PostureReward 3D曲面
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(AO_grid, TA_grid, posture_reward_grid, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('AO角度 (度)')
    ax1.set_ylabel('TA角度 (度)')
    ax1.set_zlabel('PostureReward')
    ax1.set_title(f'PostureReward 3D曲面 (距离={distance_km}km)')
    plt.colorbar(surf1, ax=ax1, shrink=0.5)
    
    # 2. EvasionReward 3D曲面
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(AO_grid, TA_grid, evasion_reward_grid, cmap='RdYlBu', alpha=0.8)
    ax2.set_xlabel('AO角度 (度)')
    ax2.set_ylabel('TA角度 (度)')
    ax2.set_zlabel('EvasionReward')
    ax2.set_title(f'EvasionReward 3D曲面 (距离={distance_km}km)')
    plt.colorbar(surf2, ax=ax2, shrink=0.5)
    
    # 3. 总奖励 3D曲面
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    surf3 = ax3.plot_surface(AO_grid, TA_grid, total_reward_grid, cmap='RdYlGn', alpha=0.8)
    ax3.set_xlabel('AO角度 (度)')
    ax3.set_ylabel('TA角度 (度)')
    ax3.set_zlabel('总奖励')
    ax3.set_title(f'总奖励 3D曲面 (距离={distance_km}km)')
    plt.colorbar(surf3, ax=ax3, shrink=0.5)
    
    # 4. PostureReward 热力图
    ax4 = fig.add_subplot(2, 3, 4)
    im4 = ax4.imshow(posture_reward_grid, extent=[0, 180, 0, 180], aspect='auto', cmap='viridis')
    ax4.set_xlabel('AO角度 (度)')
    ax4.set_ylabel('TA角度 (度)')
    ax4.set_title(f'PostureReward 热力图 (距离={distance_km}km)')
    plt.colorbar(im4, ax=ax4)
    
    # 添加等高线
    contour4 = ax4.contour(AO_grid, TA_grid, posture_reward_grid, levels=10, colors='white', alpha=0.7, linewidths=0.5)
    ax4.clabel(contour4, inline=True, fontsize=8)
    
    # 5. EvasionReward 热力图
    ax5 = fig.add_subplot(2, 3, 5)
    im5 = ax5.imshow(evasion_reward_grid, extent=[0, 180, 0, 180], aspect='auto', cmap='RdYlBu')
    ax5.set_xlabel('AO角度 (度)')
    ax5.set_ylabel('TA角度 (度)')
    ax5.set_title(f'EvasionReward 热力图 (距离={distance_km}km)')
    plt.colorbar(im5, ax=ax5)
    
    # 添加等高线
    contour5 = ax5.contour(AO_grid, TA_grid, evasion_reward_grid, levels=10, colors='black', alpha=0.7, linewidths=0.5)
    ax5.clabel(contour5, inline=True, fontsize=8)
    
    # 6. 总奖励 热力图
    ax6 = fig.add_subplot(2, 3, 6)
    im6 = ax6.imshow(total_reward_grid, extent=[0, 180, 0, 180], aspect='auto', cmap='RdYlGn')
    ax6.set_xlabel('AO角度 (度)')
    ax6.set_ylabel('TA角度 (度)')
    ax6.set_title(f'总奖励 热力图 (距离={distance_km}km)')
    plt.colorbar(im6, ax=ax6)
    
    # 添加等高线
    contour6 = ax6.contour(AO_grid, TA_grid, total_reward_grid, levels=10, colors='black', alpha=0.7, linewidths=0.5)
    ax6.clabel(contour6, inline=True, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'3d_reward_surface_{distance_km}km.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印关键点分析
    print(f"\n=== 关键点奖励值分析 (距离={distance_km}km) ===")
    key_points = [
        (0, 0, "面对面"),
        (0, 90, "侧向"),
        (0, 120, "追击"),
        (0, 180, "追击"),
        (30, 0, "小角度AO"),
        (90, 0, "垂直AO"),
        (120, 0, "大角度AO"),
        (180, 0, "背对"),
        (30, 30, "小角度组合"),
        (90, 90, "垂直组合"),
        (120, 120, "大角度组合")
    ]
    
    for ao_deg, ta_deg, desc in key_points:
        ao_rad = ao_deg * np.pi / 180
        ta_rad = ta_deg * np.pi / 180
        
        posture_r = posture_reward.get_reward(ao_rad, ta_rad, distance_m)
        evasion_r = evasion_reward.get_reward(ao_rad, ta_rad, distance_m)
        total_r = posture_r + evasion_r
        
        print(f"AO={ao_deg:3d}°, TA={ta_deg:3d}° ({desc:8s}): "
              f"Posture={posture_r:6.3f}, Evasion={evasion_r:6.3f}, Total={total_r:6.3f}")

def plot_all_distances():
    """绘制所有距离的3D可视化"""
    distances = [1.0, 3.0, 6.0]
    
    for distance_km in distances:
        print(f"\n正在生成距离={distance_km}km的3D可视化...")
        plot_3d_reward_surface_for_distance(distance_km)
    
    print("\n所有距离的3D可视化已完成！")

if __name__ == "__main__":
    plot_all_distances()
