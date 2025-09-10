import numpy as np
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R


class EvasionReward(BaseRewardFunction):
    """
    EvasionReward
    惩罚面对面接近的情况，鼓励提前规避
    - 当敌我双方都朝向对方（AO和TA都接近0）且距离在6km内时给予惩罚
    - 鼓励智能体提前规避而不是正面迎敌
    - 新增：AO在20度以内且TA在120度以上时给予正奖励
    - 新增：当TA在20度以内且我方AO大于90度时给予惩罚（避免背对敌机）
    - 整体奖励缩放到1/3
    """
    def __init__(self, config):
        super().__init__(config)
        # 配置参数
        self.evasion_distance = getattr(self.config, f'{self.__class__.__name__}_distance', 8.0)  # 放宽到8km
        self.face_to_face_angle_threshold = getattr(self.config, f'{self.__class__.__name__}_angle_threshold', 45.0)  # 放宽到45度
        self.penalty_scale = getattr(self.config, f'{self.__class__.__name__}_penalty_scale', 5.0)  # 降低惩罚强度
        
        # 新增奖励参数
        self.attack_ao_threshold = getattr(self.config, f'{self.__class__.__name__}_attack_ao_threshold', 30.0)  # 放宽到30度
        self.attack_ta_threshold = getattr(self.config, f'{self.__class__.__name__}_attack_ta_threshold', 60.0)  # 放宽到60度
        self.attack_reward_scale = getattr(self.config, f'{self.__class__.__name__}_attack_reward_scale', 3.0)  # 降低奖励强度
        
        # 新增背对惩罚参数
        self.back_turn_ta_threshold = getattr(self.config, f'{self.__class__.__name__}_back_turn_ta_threshold', 20.0)  # TA阈值20度
        self.back_turn_ao_threshold = getattr(self.config, f'{self.__class__.__name__}_back_turn_ao_threshold', 90.0)  # AO阈值90度
        self.back_turn_penalty_scale = getattr(self.config, f'{self.__class__.__name__}_back_turn_penalty_scale', 2.0)  # 背对惩罚强度
        
        # 整体奖励缩放因子
        self.reward_scale = getattr(self.config, f'{self.__class__.__name__}_reward_scale', 1.0/3.0)  # 缩放到1/3
        
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_penalty', '_distance_factor', '_angle_factor', '_attack_reward', '_attack_ao_factor', '_attack_ta_factor', '_back_turn_penalty']]

    def smooth_penalty_factor(self, angle_deg, threshold_deg):
        """平滑的惩罚因子计算"""
        if angle_deg >= threshold_deg:
            return 0.0
        # 使用平滑的衰减函数
        return np.exp(-2 * angle_deg / threshold_deg)

    def smooth_reward_factor(self, angle_deg, threshold_deg, is_upper_bound=True):
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

    def smooth_back_turn_penalty_factor(self, ta_deg, ao_deg):
        """计算背对惩罚因子"""
        # TA因子：TA越小惩罚越重
        if ta_deg >= self.back_turn_ta_threshold:
            ta_factor = 0.0
        else:
            ta_factor = np.exp(-ta_deg / self.back_turn_ta_threshold)
        
        # AO因子：AO越大惩罚越重
        if ao_deg <= self.back_turn_ao_threshold:
            ao_factor = 0.0
        else:
            # 使用平滑的增长函数，AO从90度到180度
            normalized = (ao_deg - self.back_turn_ao_threshold) / (180.0 - self.back_turn_ao_threshold)
            ao_factor = np.tanh(2 * normalized)
        
        return ta_factor * ao_factor

    def get_reward(self, task, env, agent_id):
        """
        计算规避奖励
        
        Args:
            task: task instance
            env: environment instance
            agent_id: current agent id
            
        Returns:
            (float): reward (负值表示惩罚，正值表示奖励)
        """
        reward = 0.0
        penalty = 0.0
        attack_reward = 0.0
        back_turn_penalty = 0.0
        
        # 获取我方飞机状态
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])
        
        for enemy in env.agents[agent_id].enemies:
            if not enemy.is_alive:
                continue
                
            # 获取敌机状态
            enemy_feature = np.hstack([enemy.get_position(),
                                       enemy.get_velocity()])
            
            # 计算相对角度和距离
            AO, TA, R = get_AO_TA_R(ego_feature, enemy_feature)
            
            # 转换为度数
            AO_deg = AO * 180.0 / np.pi
            TA_deg = TA * 180.0 / np.pi
            distance_km = R / 1000.0
            
            # === 平滑的惩罚机制：面对面飞行 ===
            # 使用平滑函数计算面对面程度
            ao_penalty_factor = self.smooth_penalty_factor(AO_deg, self.face_to_face_angle_threshold)
            ta_penalty_factor = self.smooth_penalty_factor(TA_deg, self.face_to_face_angle_threshold)
            angle_penalty_factor = ao_penalty_factor * ta_penalty_factor
            
            # 平滑的距离因子
            if distance_km <= self.evasion_distance:
                distance_penalty_factor = np.exp(-distance_km / self.evasion_distance)
            else:
                distance_penalty_factor = 0.0
            
            # 计算惩罚值
            penalty = self.penalty_scale * angle_penalty_factor * distance_penalty_factor
            reward -= penalty
            
            # === 平滑的奖励机制：攻击姿态 ===
            # 使用平滑函数计算攻击姿态的奖励
            ao_reward_factor = self.smooth_reward_factor(AO_deg, self.attack_ao_threshold, is_upper_bound=False)
            ta_reward_factor = self.smooth_reward_factor(TA_deg, self.attack_ta_threshold, is_upper_bound=True)
            angle_reward_factor = ao_reward_factor * ta_reward_factor
            
            # 平滑的距离因子
            if distance_km <= 2.0:
                distance_reward_factor = 0.5 + 0.5 * np.tanh(distance_km - 1.0)  # 过近时奖励递减
            elif distance_km <= 8.0:
                distance_reward_factor = 1.0  # 理想距离
            else:
                distance_reward_factor = np.exp(-0.2 * (distance_km - 8.0))  # 过远时奖励递减
            
            # 计算攻击奖励
            attack_reward = self.attack_reward_scale * angle_reward_factor * distance_reward_factor
            reward += attack_reward
            
            # === 新增：背对惩罚机制 ===
            # 当TA在20度以内且AO大于90度时给予惩罚
            back_turn_factor = self.smooth_back_turn_penalty_factor(TA_deg, AO_deg)
            
            # 距离因子：距离越近惩罚越重
            if distance_km <= 5.0:
                back_turn_distance_factor = np.exp(-distance_km / 5.0)
            else:
                back_turn_distance_factor = 0.0
            
            # 计算背对惩罚
            back_turn_penalty = self.back_turn_penalty_scale * back_turn_factor * back_turn_distance_factor
            reward -= back_turn_penalty
            
            # 记录详细信息用于调试
            if hasattr(self, '_debug') and self._debug:
                print(f"Agent {agent_id}: AO={AO_deg:.1f}°, TA={TA_deg:.1f}°, Distance={distance_km:.1f}km")
                print(f"  Penalty: {penalty:.2f} (AO_factor={ao_penalty_factor:.2f}, TA_factor={ta_penalty_factor:.2f}, Dist_factor={distance_penalty_factor:.2f})")
                print(f"  Attack Reward: {attack_reward:.2f} (AO_factor={ao_reward_factor:.2f}, TA_factor={ta_reward_factor:.2f}, Dist_factor={distance_reward_factor:.2f})")
                print(f"  Back Turn Penalty: {back_turn_penalty:.2f} (factor={back_turn_factor:.2f}, dist_factor={back_turn_distance_factor:.2f})")
        
        # 应用整体奖励缩放
        reward *= self.reward_scale
        reward = np.clip(reward, -2.0, 2.0)
        
        return self._process(reward, agent_id, (reward, penalty, attack_reward, ao_reward_factor, ta_reward_factor, back_turn_penalty))

    def set_debug(self, debug=True):
        """设置调试模式"""
        self._debug = debug
