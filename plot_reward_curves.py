import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

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
            return lambda AO, TA: 1 / (50 * AO / np.pi + 2) + 1 / 2 \
                + min((np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5
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

        new_reward = np.clip(new_reward, -2.0, 2.0)
        return new_reward

# class EvasionReward:
#     def __init__(self):
#         self.evasion_distance = 8.0
#         self.face_to_face_angle_threshold = 45.0
#         self.penalty_scale = 5.0
#         self.attack_ao_threshold = 30.0
#         self.attack_ta_threshold = 60.0
#         self.attack_reward_scale = 3.0
    
#     def smooth_penalty_factor(self, angle_deg, threshold_deg):
#         if angle_deg >= threshold_deg:
#             return 0.0
#         return np.exp(-2 * angle_deg / threshold_deg)
    
#     def smooth_reward_factor(self, angle_deg, threshold_deg, is_upper_bound=True):
#         if is_upper_bound:
#             if angle_deg < threshold_deg:
#                 return 0.0
#             normalized = (angle_deg - threshold_deg) / (180.0 - threshold_deg)
#             return np.tanh(2 * normalized)
#         else:
#             if angle_deg > threshold_deg:
#                 return 0.0
#             normalized = angle_deg / threshold_deg
#             return np.exp(-3 * normalized)
    
#     def get_reward(self, AO, TA, R):
#         """计算EvasionReward"""
#         AO_deg = AO * 180.0 / np.pi
#         TA_deg = TA * 180.0 / np.pi
#         distance_km = R / 1000.0
        
#         reward = 0.0
        
#         # 惩罚机制：面对面飞行
#         ao_penalty_factor = self.smooth_penalty_factor(AO_deg, self.face_to_face_angle_threshold)
#         ta_penalty_factor = self.smooth_penalty_factor(TA_deg, self.face_to_face_angle_threshold)
#         angle_penalty_factor = ao_penalty_factor * ta_penalty_factor
        
#         if distance_km <= self.evasion_distance:
#             distance_penalty_factor = np.exp(-distance_km / self.evasion_distance)
#         else:
#             distance_penalty_factor = 0.0
        
#         penalty = self.penalty_scale * angle_penalty_factor * distance_penalty_factor
#         reward -= penalty
        
#         # 奖励机制：攻击姿态
#         ao_reward_factor = self.smooth_reward_factor(AO_deg, self.attack_ao_threshold, is_upper_bound=False)
#         ta_reward_factor = self.smooth_reward_factor(TA_deg, self.attack_ta_threshold, is_upper_bound=True)
#         angle_reward_factor = ao_reward_factor * ta_reward_factor
        
#         if distance_km <= 2.0:
#             distance_reward_factor = 0.5 + 0.5 * np.tanh(distance_km - 1.0)
#         elif distance_km <= 8.0:
#             distance_reward_factor = 1.0
#         else:
#             distance_reward_factor = np.exp(-0.2 * (distance_km - 8.0))
        
#         attack_reward = self.attack_reward_scale * angle_reward_factor * distance_reward_factor
#         reward += attack_reward
        
#         return reward
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
        reward = np.clip(reward, -2.0, 2.0)
        
        return reward
def plot_reward_curves():
    """绘制奖励函数曲线"""
    posture_reward = PostureReward()
    evasion_reward = EvasionReward()
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('PostureReward + EvasionReward 奖励函数曲线', fontsize=16, fontweight='bold')
    
    # 1. 距离变化曲线 (AO=0, TA=0)
    distances = np.linspace(0.5, 15, 100)
    ao_zero, ta_zero = 0, 0
    
    posture_rewards_dist = [posture_reward.get_reward(ao_zero, ta_zero, d*1000) for d in distances]
    evasion_rewards_dist = [evasion_reward.get_reward(ao_zero, ta_zero, d*1000) for d in distances]
    total_rewards_dist = [p + e for p, e in zip(posture_rewards_dist, evasion_rewards_dist)]
    
    axes[0,0].plot(distances, posture_rewards_dist, 'b-', label='PostureReward', linewidth=2)
    axes[0,0].plot(distances, evasion_rewards_dist, 'r-', label='EvasionReward', linewidth=2)
    axes[0,0].plot(distances, total_rewards_dist, 'g-', label='Total Reward', linewidth=3)
    axes[0,0].set_xlabel('距离 (km)')
    axes[0,0].set_ylabel('奖励值')
    axes[0,0].set_title('距离变化 (AO=0°, TA=0°)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axvline(x=6, color='orange', linestyle='--', alpha=0.7, label='6km缩放阈值')
    axes[0,0].legend()
    
    # 2. AO角度变化曲线 (距离=3km, TA=0)
    ao_angles = np.linspace(0, 180, 100)
    distance_3km, ta_zero = 3000, 0
    
    posture_rewards_ao = [posture_reward.get_reward(ao*np.pi/180, ta_zero, distance_3km) for ao in ao_angles]
    evasion_rewards_ao = [evasion_reward.get_reward(ao*np.pi/180, ta_zero, distance_3km) for ao in ao_angles]
    total_rewards_ao = [p + e for p, e in zip(posture_rewards_ao, evasion_rewards_ao)]
    
    axes[0,1].plot(ao_angles, posture_rewards_ao, 'b-', label='PostureReward', linewidth=2)
    axes[0,1].plot(ao_angles, evasion_rewards_ao, 'r-', label='EvasionReward', linewidth=2)
    axes[0,1].plot(ao_angles, total_rewards_ao, 'g-', label='Total Reward', linewidth=3)
    axes[0,1].set_xlabel('AO角度 (度)')
    axes[0,1].set_ylabel('奖励值')
    axes[0,1].set_title('AO角度变化 (距离=3km, TA=0°)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. TA角度变化曲线 (距离=3km, AO=0)
    ta_angles = np.linspace(0, 180, 100)
    distance_3km, ao_zero = 3000, 0
    
    posture_rewards_ta = [posture_reward.get_reward(ao_zero, ta*np.pi/180, distance_3km) for ta in ta_angles]
    evasion_rewards_ta = [evasion_reward.get_reward(ao_zero, ta*np.pi/180, distance_3km) for ta in ta_angles]
    total_rewards_ta = [p + e for p, e in zip(posture_rewards_ta, evasion_rewards_ta)]
    
    axes[1,0].plot(ta_angles, posture_rewards_ta, 'b-', label='PostureReward', linewidth=2)
    axes[1,0].plot(ta_angles, evasion_rewards_ta, 'r-', label='EvasionReward', linewidth=2)
    axes[1,0].plot(ta_angles, total_rewards_ta, 'g-', label='Total Reward', linewidth=3)
    axes[1,0].set_xlabel('TA角度 (度)')
    axes[1,0].set_ylabel('奖励值')
    axes[1,0].set_title('TA角度变化 (距离=3km, AO=0°)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. 热力图：AO-TA组合 (距离=3km)
    ao_range = np.linspace(0, 180, 50)
    ta_range = np.linspace(0, 180, 50)
    AO_grid, TA_grid = np.meshgrid(ao_range, ta_range)
    
    total_reward_grid = np.zeros_like(AO_grid)
    for i, ao in enumerate(ao_range):
        for j, ta in enumerate(ta_range):
            posture_r = posture_reward.get_reward(ao*np.pi/180, ta*np.pi/180, distance_3km)
            evasion_r = evasion_reward.get_reward(ao*np.pi/180, ta*np.pi/180, distance_3km)
            total_reward_grid[j, i] = posture_r + evasion_r
    
    im = axes[1,1].imshow(total_reward_grid, extent=[0, 180, 0, 180], aspect='auto', cmap='RdYlGn')
    axes[1,1].set_xlabel('AO角度 (度)')
    axes[1,1].set_ylabel('TA角度 (度)')
    axes[1,1].set_title('总奖励热力图 (距离=3km)')
    plt.colorbar(im, ax=axes[1,1], label='总奖励值')
    
    # 添加等高线
    contour = axes[1,1].contour(AO_grid, TA_grid, total_reward_grid, levels=10, colors='black', alpha=0.5, linewidths=0.5)
    axes[1,1].clabel(contour, inline=True, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('reward_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印一些关键点的奖励值
    print("\n=== 关键点奖励值分析 ===")
    print("距离=3km, AO=0°, TA=0° (面对面):")
    p1 = posture_reward.get_reward(0, 0, 3000)
    e1 = evasion_reward.get_reward(0, 0, 3000)
    print(f"  PostureReward: {p1:.3f}")
    print(f"  EvasionReward: {e1:.3f}")
    print(f"  总奖励: {p1+e1:.3f}")
    
    print("\n距离=3km, AO=0°, TA=90° (侧向):")
    p2 = posture_reward.get_reward(0, np.pi/2, 3000)
    e2 = evasion_reward.get_reward(0, np.pi/2, 3000)
    print(f"  PostureReward: {p2:.3f}")
    print(f"  EvasionReward: {e2:.3f}")
    print(f"  总奖励: {p2+e2:.3f}")
    
    print("\n距离=3km, AO=0°, TA=180° (追击):")
    p3 = posture_reward.get_reward(0, np.pi, 3000)
    e3 = evasion_reward.get_reward(0, np.pi, 3000)
    print(f"  PostureReward: {p3:.3f}")
    print(f"  EvasionReward: {e3:.3f}")
    print(f"  总奖励: {p3+e3:.3f}")

if __name__ == "__main__":
    plot_reward_curves()
