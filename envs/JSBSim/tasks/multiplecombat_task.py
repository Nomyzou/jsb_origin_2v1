import numpy as np
from gymnasium import spaces
from typing import Tuple
import torch
import logging

from ..tasks import SingleCombatTask
from ..core.catalog import Catalog as c
from ..core.simulatior import MissileSimulator
from ..reward_functions import AltitudeReward, PostureReward, EventDrivenReward, MissilePostureReward,EvasionReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, SafeReturn
from ..utils.utils import get_AO_TA_R, LLA2NEU, get_root_dir
from ..model.baseline_actor import BaselineActor


class MultipleCombatTask(SingleCombatTask):
    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = [
            AltitudeReward(self.config),
            PostureReward(self.config),
            EventDrivenReward(self.config),
            EvasionReward(self.config)
        ]

        self.termination_conditions = [
            SafeReturn(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]

    @property
    def num_agents(self) -> int:
        cfg_aircraft = getattr(self.config, 'aircraft_configs', None)
        if isinstance(cfg_aircraft, dict) and len(cfg_aircraft) > 0:
            return len(cfg_aircraft)
        # fallback for legacy configs
        return 4

    def load_variables(self):
        self.state_var = [
            c.position_long_gc_deg,             # 0. lontitude  (unit: °)
            c.position_lat_geod_deg,            # 1. latitude   (unit: °)
            c.position_h_sl_m,                  # 2. altitude   (unit: m)
            c.attitude_roll_rad,                # 3. roll       (unit: rad)
            c.attitude_pitch_rad,               # 4. pitch      (unit: rad)
            c.attitude_heading_true_rad,        # 5. yaw        (unit: rad)
            c.velocities_v_north_mps,           # 6. v_north    (unit: m/s)
            c.velocities_v_east_mps,            # 7. v_east     (unit: m/s)
            c.velocities_v_down_mps,            # 8. v_down     (unit: m/s)
            c.velocities_u_mps,                 # 9. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 # 10. v_body_y  (unit: m/s)
            c.velocities_w_mps,                 # 11. v_body_z  (unit: m/s)
            c.velocities_vc_mps,                # 12. vc        (unit: m/s)
            c.accelerations_n_pilot_x_norm,     # 13. a_north   (unit: G)
            c.accelerations_n_pilot_y_norm,     # 14. a_east    (unit: G)
            c.accelerations_n_pilot_z_norm,     # 15. a_down    (unit: G)
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,             # [-1., 1.]
            c.fcs_elevator_cmd_norm,            # [-1., 1.]
            c.fcs_rudder_cmd_norm,              # [-1., 1.]
            c.fcs_throttle_cmd_norm,            # [0.4, 0.9]
        ]
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    def load_observation_space(self):
        self.obs_length = 9 + (self.num_agents - 1) * 6
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(self.num_agents * self.obs_length,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])

    def get_obs(self, env, agent_id):
        norm_obs = np.zeros(self.obs_length)
        # (1) ego info normalization
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])
        norm_obs[0] = ego_state[2] / 5000            # 0. ego altitude   (unit: 5km)
        norm_obs[1] = np.sin(ego_state[3])           # 1. ego_roll_sin
        norm_obs[2] = np.cos(ego_state[3])           # 2. ego_roll_cos
        norm_obs[3] = np.sin(ego_state[4])           # 3. ego_pitch_sin
        norm_obs[4] = np.cos(ego_state[4])           # 4. ego_pitch_cos
        norm_obs[5] = ego_state[9] / 340             # 5. ego v_body_x   (unit: mh)
        norm_obs[6] = ego_state[10] / 340            # 6. ego v_body_y   (unit: mh)
        norm_obs[7] = ego_state[11] / 340            # 7. ego v_body_z   (unit: mh)
        norm_obs[8] = ego_state[12] / 340            # 8. ego vc   (unit: mh)(unit: 5G)
        # (2) relative inof w.r.t partner+enemies state
        offset = 8
        for sim in env.agents[agent_id].partners + env.agents[agent_id].enemies:
            state = np.array(sim.get_property_values(self.state_var))
            cur_ned = LLA2NEU(*state[:3], env.center_lon, env.center_lat, env.center_alt)
            feature = np.array([*cur_ned, *(state[6:9])])
            AO, TA, R, side_flag = get_AO_TA_R(ego_feature, feature, return_side=True)
            norm_obs[offset+1] = (state[9] - ego_state[9]) / 340
            norm_obs[offset+2] = (state[2] - ego_state[2]) / 1000
            norm_obs[offset+3] = AO
            norm_obs[offset+4] = TA
            norm_obs[offset+5] = R / 10000
            norm_obs[offset+6] = side_flag
            offset += 6
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        """Convert discrete action index into continuous value.
        """
        norm_act = np.zeros(4)
        norm_act[0] = action[0] * 2. / (self.action_space.nvec[0] - 1.) - 1.
        norm_act[1] = action[1] * 2. / (self.action_space.nvec[1] - 1.) - 1.
        norm_act[2] = action[2] * 2. / (self.action_space.nvec[2] - 1.) - 1.
        norm_act[3] = action[3] * 0.5 / (self.action_space.nvec[3] - 1.) + 0.4
        return norm_act

    def get_reward(self, env, agent_id, info: dict = ...) -> Tuple[float, dict]:
        if env.agents[agent_id].is_alive:
            return super().get_reward(env, agent_id, info=info)
        else:
            return 0.0, info


class HierarchicalMultipleCombatTask(MultipleCombatTask):
    
    def __init__(self, config: str):
        super().__init__(config)
        self.lowlevel_policy = BaselineActor()
        try:
            try:
                sd = torch.load(get_root_dir() + '/model/baseline_model.pt', map_location=torch.device('cpu'), weights_only=True)
            except TypeError:
                sd = torch.load(get_root_dir() + '/model/baseline_model.pt', map_location=torch.device('cpu'))
            self.lowlevel_policy.load_state_dict(sd)
        except Exception as e:
            logging.warning(f"Failed to load baseline_model.pt strictly: {e}")
            self.lowlevel_policy.load_state_dict(sd, strict=False)
        self.lowlevel_policy.eval()
        self.norm_delta_altitude = np.array([0.1, 0, -0.1])
        self.norm_delta_heading = np.array([-np.pi / 6, -np.pi / 12, 0, np.pi / 12, np.pi / 6])
        self.norm_delta_velocity = np.array([0.05, 0, -0.05])

    def load_action_space(self):
        self.action_space = spaces.MultiDiscrete([3, 5, 3])

    def normalize_action(self, env, agent_id, action):
        """Convert high-level action into low-level action.
        """
        # generate low-level input_obs
        raw_obs = self.get_obs(env, agent_id)
        input_obs = np.zeros(12)
        # (1) delta altitude/heading/velocity
        input_obs[0] = self.norm_delta_altitude[action[0]]
        input_obs[1] = self.norm_delta_heading[action[1]]
        input_obs[2] = self.norm_delta_velocity[action[2]]
        # (2) ego info
        input_obs[3:12] = raw_obs[:9]
        input_obs = np.expand_dims(input_obs, axis=0)
        # output low-level action
        _action, _rnn_states = self.lowlevel_policy(input_obs, self._inner_rnn_states[agent_id])
        action = _action.detach().cpu().numpy().squeeze(0)
        self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()
        # normalize low-level action
        norm_act = np.zeros(4)
        norm_act[0] = action[0] / 20 - 1.
        norm_act[1] = action[1] / 20 - 1.
        norm_act[2] = action[2] / 20 - 1.
        norm_act[3] = action[3] / 58 + 0.4
        return norm_act

    def reset(self, env):
        """Task-specific reset, include reward function reset.
        """
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        return super().reset(env)



class HierarchicalMultipleCombatShootTask(HierarchicalMultipleCombatTask):
    def __init__(self, config: str):
        super().__init__(config)
        self.max_attack_angle = getattr(self.config, 'max_attack_angle', 180)
        self.max_attack_distance = getattr(self.config, 'max_attack_distance', np.inf)
        self.min_attack_interval = getattr(self.config, 'min_attack_interval', 125)
        self.reward_functions = [
            PostureReward(self.config),
            MissilePostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config)
        ]
    
    def load_observation_space(self):
        self.obs_length = 9 + self.num_agents  * 6
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(self.num_agents * self.obs_length,))
    
    def load_action_space(self):
        self.action_space = spaces.MultiDiscrete([3, 5, 3, 2])


    def get_obs(self, env, agent_id):
        norm_obs = np.zeros(self.obs_length)
        # (1) ego info normalization
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])
        norm_obs[0] = ego_state[2] / 5000            # 0. ego altitude   (unit: 5km)
        norm_obs[1] = np.sin(ego_state[3])           # 1. ego_roll_sin
        norm_obs[2] = np.cos(ego_state[3])           # 2. ego_roll_cos
        norm_obs[3] = np.sin(ego_state[4])           # 3. ego_pitch_sin
        norm_obs[4] = np.cos(ego_state[4])           # 4. ego_pitch_cos
        norm_obs[5] = ego_state[9] / 340             # 5. ego v_body_x   (unit: mh)
        norm_obs[6] = ego_state[10] / 340            # 6. ego v_body_y   (unit: mh)
        norm_obs[7] = ego_state[11] / 340            # 7. ego v_body_z   (unit: mh)
        norm_obs[8] = ego_state[12] / 340            # 8. ego vc   (unit: mh)(unit: 5G)
        # (2) relative inof w.r.t partner+enemies state
        offset = 8
        for sim in env.agents[agent_id].partners + env.agents[agent_id].enemies:
            state = np.array(sim.get_property_values(self.state_var))
            cur_ned = LLA2NEU(*state[:3], env.center_lon, env.center_lat, env.center_alt)
            feature = np.array([*cur_ned, *(state[6:9])])
            AO, TA, R, side_flag = get_AO_TA_R(ego_feature, feature, return_side=True)
            norm_obs[offset+1] = (state[9] - ego_state[9]) / 340
            norm_obs[offset+2] = (state[2] - ego_state[2]) / 1000
            norm_obs[offset+3] = AO
            norm_obs[offset+4] = TA
            norm_obs[offset+5] = R / 10000
            norm_obs[offset+6] = side_flag
            offset += 6
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        # (3) missile info TODO: multiple missile and parnter's missile?
        missile_sim = env.agents[agent_id].check_missile_warning() #
        if missile_sim is not None:
            missile_feature = np.concatenate((missile_sim.get_position(), missile_sim.get_velocity()))
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, missile_feature, return_side=True)
            norm_obs[offset + 1] = (np.linalg.norm(missile_sim.get_velocity()) - ego_state[9]) / 340
            norm_obs[offset + 2] = (missile_feature[2] - ego_state[2]) / 1000
            norm_obs[offset + 3] = ego_AO
            norm_obs[offset + 4] = ego_TA
            norm_obs[offset + 5] = R / 10000
            norm_obs[offset + 6] = side_flag
        return norm_obs

    def reset(self, env):
        """Reset fighter blood & missile status
        """
        self._last_shoot_time = {agent_id: -self.min_attack_interval for agent_id in env.agents.keys()}
        self._remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self._shoot_action = {agent_id: False for agent_id in env.agents.keys()}
        return super().reset(env)

    def normalize_action(self, env, agent_id, action):
        self._shoot_action[agent_id] = action[3] > 0
        return super().normalize_action(env, agent_id, action[:3])

    def step(self, env):
        SingleCombatTask.step(self, env)
        for agent_id, agent in env.agents.items():
            # [RL-based missile launch with limited condition]
            # Determine whether can launch missile at the nearest enemy aircraft
            target_list = list(map(lambda x: x.get_position() - agent.get_position(), agent.enemies))
            target_distance = list(map(np.linalg.norm, target_list))
            target_index = np.argmin(target_distance)
            target = target_list[target_index]
            heading = agent.get_velocity()
            distance = target_distance[target_index]
            attack_angle = np.rad2deg(np.arccos(np.clip(np.sum(target * heading) / (distance * np.linalg.norm(heading) + 1e-8), -1, 1)))
            shoot_interval = env.current_step - self._last_shoot_time[agent_id]

            shoot_flag = agent.is_alive and self._shoot_action[agent_id] and self._remaining_missiles[agent_id] > 0 \
                and attack_angle <= self.max_attack_angle and distance <= self.max_attack_distance and shoot_interval >= self.min_attack_interval
            if shoot_flag:
                new_missile_uid = agent_id + str(self._remaining_missiles[agent_id])
                env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=agent.enemies[target_index], uid=new_missile_uid))
                self._remaining_missiles[agent_id] -= 1
                self._last_shoot_time[agent_id] = env.current_step


class MultipleCombat1v1StyleTask(MultipleCombatTask):
    """
    4v4场景的1v1风格任务类
    每架飞机只观察对应编号的敌机，观察空间为15维（与1v1相同）
    """
    
    def __init__(self, config):
        super().__init__(config)
        
    def load_observation_space(self):
        # 15维观察空间：9维自身信息 + 6维对应敌机信息
        self.obs_length = 15
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(self.num_agents * self.obs_length,))
    
    def get_obs(self, env, agent_id):
        """
        获取1v1风格的观察
        每架飞机只观察对应编号的敌机
        A0100 -> B0100, A0200 -> B0200, A0300 -> B0300, A0400 -> B0400
        """
        norm_obs = np.zeros(self.obs_length)
        
        # (1) ego info normalization (9维)
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])
        
        norm_obs[0] = ego_state[2] / 5000            # 0. ego altitude   (unit: 5km)
        norm_obs[1] = np.sin(ego_state[3])           # 1. ego_roll_sin
        norm_obs[2] = np.cos(ego_state[3])          # 2. ego_roll_cos
        norm_obs[3] = np.sin(ego_state[4])           # 3. ego_pitch_sin
        norm_obs[4] = np.cos(ego_state[4])           # 4. ego_pitch_cos
        norm_obs[5] = ego_state[9] / 340             # 5. ego v_body_x   (unit: mh)
        norm_obs[6] = ego_state[10] / 340            # 6. ego v_body_y   (unit: mh)
        norm_obs[7] = ego_state[11] / 340            # 7. ego v_body_z   (unit: mh)
        norm_obs[8] = ego_state[12] / 340            # 8. ego vc   (unit: mh)
        
        # (2) 找到对应编号的敌机
        target_enemy_id = self._get_corresponding_enemy(agent_id)
        if target_enemy_id in env.agents and env.agents[target_enemy_id].is_alive:
            enemy_state = np.array(env.agents[target_enemy_id].get_property_values(self.state_var))
            enemy_cur_ned = LLA2NEU(*enemy_state[:3], env.center_lon, env.center_lat, env.center_alt)
            enemy_feature = np.array([*enemy_cur_ned, *(enemy_state[6:9])])
            
            # 计算相对信息 (6维)
            AO, TA, R, side_flag = get_AO_TA_R(ego_feature, enemy_feature, return_side=True)
            norm_obs[9] = (enemy_state[9] - ego_state[9]) / 340      # 9. delta_v_body_x (unit: mh)
            norm_obs[10] = (enemy_state[2] - ego_state[2]) / 1000    # 10. delta_altitude (unit: km)
            norm_obs[11] = AO                                        # 11. ego_AO (unit: rad)
            norm_obs[12] = TA                                        # 12. ego_TA (unit: rad)
            norm_obs[13] = R / 10000                                # 13. relative_distance (unit: 10km)
            norm_obs[14] = side_flag                                 # 14. side_flag
        else:
            # 如果对应敌机不存在或已死亡，填充零值
            norm_obs[9:15] = 0.0
        
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs
    
    def _get_corresponding_enemy(self, agent_id):
        """
        根据飞机编号找到对应的敌机
        A0100 -> B0100, A0200 -> B0200, A0300 -> B0300, A0400 -> B0400
        """
        if agent_id.startswith('A'):
            # 红方飞机，对应蓝方敌机
            enemy_id = 'B' + agent_id[1:]
        elif agent_id.startswith('B'):
            # 蓝方飞机，对应红方敌机
            enemy_id = 'A' + agent_id[1:]
        else:
            # 未知编号，返回第一个敌机
            return None
        
        return enemy_id


class HierarchicalMultipleCombat1v1StyleTask(HierarchicalMultipleCombatTask):
    """
    4v4场景的1v1风格分层任务类
    继承分层结构，但使用1v1风格的观察空间
    """
    
    def __init__(self, config):
        super().__init__(config)
        
    def load_observation_space(self):
        # 15维观察空间：9维自身信息 + 6维对应敌机信息
        self.obs_length = 15
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(self.num_agents * self.obs_length,))
    
    def get_obs(self, env, agent_id):
        """
        获取1v1风格的观察
        每架飞机只观察对应编号的敌机
        A0100 -> B0100, A0200 -> B0200, A0300 -> B0300, A0400 -> B0400
        """
        norm_obs = np.zeros(self.obs_length)
        
        # (1) ego info normalization (9维)
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])
        
        norm_obs[0] = ego_state[2] / 5000            # 0. ego altitude   (unit: 5km)
        norm_obs[1] = np.sin(ego_state[3])           # 1. ego_roll_sin
        norm_obs[2] = np.cos(ego_state[3])          # 2. ego_roll_cos
        norm_obs[3] = np.sin(ego_state[4])           # 3. ego_pitch_sin
        norm_obs[4] = np.cos(ego_state[4])           # 4. ego_pitch_cos
        norm_obs[5] = ego_state[9] / 340             # 5. ego v_body_x   (unit: mh)
        norm_obs[6] = ego_state[10] / 340            # 6. ego v_body_y   (unit: mh)
        norm_obs[7] = ego_state[11] / 340            # 7. ego v_body_z   (unit: mh)
        norm_obs[8] = ego_state[12] / 340            # 8. ego vc   (unit: mh)
        
        # (2) 找到对应编号的敌机
        target_enemy_id = self._get_corresponding_enemy(agent_id)
        if target_enemy_id in env.agents and env.agents[target_enemy_id].is_alive:
            enemy_state = np.array(env.agents[target_enemy_id].get_property_values(self.state_var))
            enemy_cur_ned = LLA2NEU(*enemy_state[:3], env.center_lon, env.center_lat, env.center_alt)
            enemy_feature = np.array([*enemy_cur_ned, *(enemy_state[6:9])])
            
            # 计算相对信息 (6维)
            AO, TA, R, side_flag = get_AO_TA_R(ego_feature, enemy_feature, return_side=True)
            norm_obs[9] = (enemy_state[9] - ego_state[9]) / 340      # 9. delta_v_body_x (unit: mh)
            norm_obs[10] = (enemy_state[2] - ego_state[2]) / 1000    # 10. delta_altitude (unit: km)
            norm_obs[11] = AO                                        # 11. ego_AO (unit: rad)
            norm_obs[12] = TA                                        # 12. ego_TA (unit: rad)
            norm_obs[13] = R / 10000                                # 13. relative_distance (unit: 10km)
            norm_obs[14] = side_flag                                 # 14. side_flag
        else:
            # 如果对应敌机不存在或已死亡，填充零值
            norm_obs[9:15] = 0.0
        
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs
    
    def _get_corresponding_enemy(self, agent_id):
        """
        根据飞机编号找到对应的敌机
        A0100 -> B0100, A0200 -> B0200, A0300 -> B0300, A0400 -> B0400
        """
        if agent_id.startswith('A'):
            # 红方飞机，对应蓝方敌机
            enemy_id = 'B' + agent_id[1:]
        elif agent_id.startswith('B'):
            # 蓝方飞机，对应红方敌机
            enemy_id = 'A' + agent_id[1:]
        else:
            # 未知编号，返回第一个敌机
            return None
        
        return enemy_id


# ... existing code ...

class HierarchicalMultipleCombat2v1StyleTask(HierarchicalMultipleCombatTask):
    """
    5v4场景的2v1风格分层任务类
    支持两种观察空间：
    1. 15维：9维自身信息 + 6维对应敌机信息
    2. 21维：9维自身信息 + 6维友机信息 + 6维对应敌机信息
    
    蓝方飞机：固定使用15维观察空间
    红方飞机：根据运行情况动态决定使用15维还是21维观察空间
    """
    
    def __init__(self, config):
        super().__init__(config)
        # 红方飞机观察空间类型映射（运行时动态设置）
        self.red_obs_type_mapping = {}
        # 蓝方飞机固定使用15维观察空间
        self.blue_obs_type_mapping = {}
        
    def set_red_obs_type(self, agent_id, obs_type):
        """
        设置红方飞机的观察空间类型
        Args:
            agent_id: 飞机ID
            obs_type: 0表示15维，1表示21维
        """
        if agent_id.startswith('A'):
            self.red_obs_type_mapping[agent_id] = obs_type
    
    def set_red_obs_types(self, obs_type_dict):
        """
        批量设置红方飞机的观察空间类型
        Args:
            obs_type_dict: {agent_id: obs_type} 字典
        """
        for agent_id, obs_type in obs_type_dict.items():
            if agent_id.startswith('A'):
                self.red_obs_type_mapping[agent_id] = obs_type
    
    def load_observation_space(self):
        # 支持两种观察空间：15维和21维
        self.obs_length_15 = 15  # 9 + 6
        self.obs_length_21 = 21  # 9 + 6 + 6
        
        # 使用最大观察空间作为统一空间
        self.obs_length = self.obs_length_21
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(self.num_agents * self.obs_length,))
    
    def get_obs(self, env, agent_id):
        """
        获取2v1风格的观察
        蓝方飞机固定使用15维观察空间
        红方飞机根据设置使用15维或21维观察空间
        """
        if agent_id.startswith('B'):
            # 蓝方飞机固定使用15维观察空间
            return self._get_obs_15d(env, agent_id)
        elif agent_id.startswith('A'):
            # 红方飞机根据设置使用观察空间
            obs_type = self.red_obs_type_mapping.get(agent_id, 0)  # 默认15维
            if obs_type == 0:
                return self._get_obs_15d(env, agent_id)
            else:
                return self._get_obs_21d(env, agent_id)
        else:
            # 未知飞机类型，使用15维观察空间
            return self._get_obs_15d(env, agent_id)
    
    def _get_obs_15d(self, env, agent_id):
        """
        获取15维观察空间
        9维自身信息 + 6维对应敌机信息
        """
        norm_obs = np.zeros(self.obs_length)
        
        # (1) ego info normalization (9维)
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])
        
        norm_obs[0] = ego_state[2] / 5000            # 0. ego altitude   (unit: 5km)
        norm_obs[1] = np.sin(ego_state[3])           # 1. ego_roll_sin
        norm_obs[2] = np.cos(ego_state[3])          # 2. ego_roll_cos
        norm_obs[3] = np.sin(ego_state[4])           # 3. ego_pitch_sin
        norm_obs[4] = np.cos(ego_state[4])           # 4. ego_pitch_cos
        norm_obs[5] = ego_state[9] / 340             # 5. ego v_body_x   (unit: mh)
        norm_obs[6] = ego_state[10] / 340            # 6. ego v_body_y   (unit: mh)
        norm_obs[7] = ego_state[11] / 340            # 7. ego v_body_z   (unit: mh)
        norm_obs[8] = ego_state[12] / 340            # 8. ego vc   (unit: mh)
        
        # (2) 找到对应编号的敌机
        target_enemy_id = self._get_corresponding_enemy(agent_id)
        if target_enemy_id in env.agents and env.agents[target_enemy_id].is_alive:
            enemy_state = np.array(env.agents[target_enemy_id].get_property_values(self.state_var))
            enemy_cur_ned = LLA2NEU(*enemy_state[:3], env.center_lon, env.center_lat, env.center_alt)
            enemy_feature = np.array([*enemy_cur_ned, *(enemy_state[6:9])])
            
            # 计算相对信息 (6维)
            AO, TA, R, side_flag = get_AO_TA_R(ego_feature, enemy_feature, return_side=True)
            norm_obs[9] = (enemy_state[9] - ego_state[9]) / 340      # 9. delta_v_body_x (unit: mh)
            norm_obs[10] = (enemy_state[2] - ego_state[2]) / 1000    # 10. delta_altitude (unit: km)
            norm_obs[11] = AO                                        # 11. ego_AO (unit: rad)
            norm_obs[12] = TA                                        # 12. ego_TA (unit: rad)
            norm_obs[13] = R / 10000                                # 13. relative_distance (unit: 10km)
            norm_obs[14] = side_flag                                 # 14. side_flag
        else:
            # 如果对应敌机不存在或已死亡，填充零值
            norm_obs[9:15] = 0.0
        
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs
    
    def _get_obs_21d(self, env, agent_id):
        """
        获取21维观察空间
        9维自身信息 + 6维友机信息 + 6维对应敌机信息
        """
        norm_obs = np.zeros(self.obs_length)
        
        # (1) ego info normalization (9维)
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])
        
        norm_obs[0] = ego_state[2] / 5000            # 0. ego altitude   (unit: 5km)
        norm_obs[1] = np.sin(ego_state[3])           # 1. ego_roll_sin
        norm_obs[2] = np.cos(ego_state[3])          # 2. ego_roll_cos
        norm_obs[3] = np.sin(ego_state[4])           # 3. ego_pitch_sin
        norm_obs[4] = np.cos(ego_state[4])           # 4. ego_pitch_cos
        norm_obs[5] = ego_state[9] / 340             # 5. ego v_body_x   (unit: mh)
        norm_obs[6] = ego_state[10] / 340            # 6. ego v_body_y   (unit: mh)
        norm_obs[7] = ego_state[11] / 340            # 7. ego v_body_z   (unit: mh)
        norm_obs[8] = ego_state[12] / 340            # 8. ego vc   (unit: mh)
        
        # (2) 找到友机信息 (6维)
        partner_id = self._get_partner_aircraft(agent_id)
        if partner_id in env.agents and env.agents[partner_id].is_alive:
            partner_state = np.array(env.agents[partner_id].get_property_values(self.state_var))
            partner_cur_ned = LLA2NEU(*partner_state[:3], env.center_lon, env.center_lat, env.center_alt)
            partner_feature = np.array([*partner_cur_ned, *(partner_state[6:9])])
            
            # 计算与友机的相对信息 (6维)
            AO, TA, R, side_flag = get_AO_TA_R(ego_feature, partner_feature, return_side=True)
            norm_obs[9] = (partner_state[9] - ego_state[9]) / 340      # 9. delta_v_body_x (unit: mh)
            norm_obs[10] = (partner_state[2] - ego_state[2]) / 1000    # 10. delta_altitude (unit: km)
            norm_obs[11] = AO                                        # 11. ego_AO (unit: rad)
            norm_obs[12] = TA                                        # 12. ego_TA (unit: rad)
            norm_obs[13] = R / 10000                                # 13. relative_distance (unit: 10km)
            norm_obs[14] = side_flag                                 # 14. side_flag
        else:
            # 如果友机不存在或已死亡，填充零值
            norm_obs[9:15] = 0.0
        
        # (3) 找到对应编号的敌机 (6维)
        target_enemy_id = self._get_corresponding_enemy(agent_id)
        if target_enemy_id in env.agents and env.agents[target_enemy_id].is_alive:
            enemy_state = np.array(env.agents[target_enemy_id].get_property_values(self.state_var))
            enemy_cur_ned = LLA2NEU(*enemy_state[:3], env.center_lon, env.center_lat, env.center_alt)
            enemy_feature = np.array([*enemy_cur_ned, *(enemy_state[6:9])])
            
            # 计算与敌机的相对信息 (6维)
            AO, TA, R, side_flag = get_AO_TA_R(ego_feature, enemy_feature, return_side=True)
            norm_obs[15] = (enemy_state[9] - ego_state[9]) / 340      # 15. delta_v_body_x (unit: mh)
            norm_obs[16] = (enemy_state[2] - ego_state[2]) / 1000    # 16. delta_altitude (unit: km)
            norm_obs[17] = AO                                        # 17. ego_AO (unit: rad)
            norm_obs[18] = TA                                        # 18. ego_TA (unit: rad)
            norm_obs[19] = R / 10000                                # 19. relative_distance (unit: 10km)
            norm_obs[20] = side_flag                                 # 20. side_flag
        else:
            # 如果对应敌机不存在或已死亡，填充零值
            norm_obs[15:21] = 0.0
        
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs
    
    def _get_corresponding_enemy(self, agent_id):
        """
        根据飞机编号找到对应的敌机
        A0100 -> B0100, A0200 -> B0200, A0300 -> B0300, A0400 -> B0400, A0500 -> B0400
        """
        if agent_id.startswith('A'):
            # 红方飞机，对应蓝方敌机
            if agent_id == 'A0500':
                # A0500对应B0400（因为蓝方只有4架飞机）
                enemy_id = 'B0400'
            else:
                enemy_id = 'B' + agent_id[1:]
        elif agent_id.startswith('B'):
            # 蓝方飞机，对应红方敌机
            if agent_id == 'B0400':
                # B0400对应A0400和A0500（因为红方有5架飞机）
                # 这里选择A0400作为主要对应
                enemy_id = 'A0400'
            else:
                enemy_id = 'A' + agent_id[1:]
        else:
            # 未知编号，返回第一个敌机
            return None
        
        return enemy_id
    
    def _get_partner_aircraft(self, agent_id):
        """
        获取友机编号
        为21维观察空间的飞机分配友机
        """
        if agent_id.startswith('A'):
            # 红方飞机友机配对
            if agent_id == 'A0100':
                return 'A0200'  # A0100的友机是A0200
            elif agent_id == 'A0200':
                return 'A0100'  # A0200的友机是A0100
            elif agent_id == 'A0300':
                return 'A0400'  # A0300的友机是A0400
            elif agent_id == 'A0400':
                return 'A0300'  # A0400的友机是A0300
            elif agent_id == 'A0500':
                return 'A0400'  # A0500的友机是A0400
            else:
                return None
        else:
            # 蓝方飞机不使用21维观察空间，不需要友机
            return None
    
    def get_obs_type(self, agent_id):
        """
        获取指定飞机的观察空间类型
        返回: 0表示15维，1表示21维
        """
        if agent_id.startswith('B'):
            return 0  # 蓝方飞机固定使用15维
        elif agent_id.startswith('A'):
            return self.red_obs_type_mapping.get(agent_id, 0)  # 红方飞机根据设置
        else:
            return 0  # 默认15维
    
    def get_obs_length(self, agent_id):
        """
        获取指定飞机的观察空间长度
        """
        obs_type = self.get_obs_type(agent_id)
        if obs_type == 0:
            return self.obs_length_15
        else:
            return self.obs_length_21
    
    def get_red_obs_types(self):
        """
        获取所有红方飞机的观察空间类型
        返回: {agent_id: obs_type} 字典
        """
        return self.red_obs_type_mapping.copy()
    
    def reset(self, env):
        """
        重置任务，包括红方飞机观察空间类型映射
        """
        # 重置红方飞机观察空间类型映射
        self.red_obs_type_mapping = {}
        return super().reset(env)