import logging
import time
from typing import List
import random

import numpy as np
import torch

from algorithms.utils.buffer import SharedReplayBuffer
from .base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class Share2v1JSBSimRunner(Runner):

    def load(self):
        self.obs_space = self.envs.observation_space
        self.share_obs_space = self.envs.share_observation_space
        self.act_space = self.envs.action_space
        self.num_agents = self.envs.num_agents  # 3个智能体 (2红方 + 1蓝方)
        self.use_selfplay = self.all_args.use_selfplay  # type: bool

        # 添加环境配置检查
        logging.info("=== 环境配置检查 ===")
        logging.info(f"envs.num_agents: {self.envs.num_agents}")
        logging.info(f"self.num_agents: {self.num_agents}")
        logging.info(f"self.n_rollout_threads: {self.n_rollout_threads}")
        logging.info(f"self.obs_space: {self.obs_space}")
        logging.info(f"self.share_obs_space: {self.share_obs_space}")
        logging.info(f"self.act_space: {self.act_space}")
        logging.info("=== 环境配置检查结束 ===")

        # # policy & algorithm
        # if self.algorithm_name == "mappo":
        #     from algorithms.mappo.ppo_trainer import PPOTrainer as Trainer
        #     from algorithms.mappo.ppo_policy import PPOPolicy as Policy
        # else:
        #     raise NotImplementedError
        # self.policy = Policy(self.all_args, self.obs_space, self.share_obs_space, self.act_space, device=self.device)
        # self.trainer = Trainer(self.all_args, device=self.device)

        # # buffer - 只存储红方智能体的经验 (2个)
        # if self.use_selfplay:
        #     self.buffer = SharedReplayBuffer(self.all_args, 2, self.obs_space, self.share_obs_space, self.act_space)
        # else:
        #     self.buffer = SharedReplayBuffer(self.all_args, self.num_agents, self.obs_space, self.share_obs_space, self.act_space)

        # # 在创建buffer后立即检查
        # logging.info("=== Buffer创建后检查 ===")
        # logging.info(f"buffer.obs shape: {self.buffer.obs.shape}")
        # logging.info(f"buffer.share_obs shape: {self.buffer.share_obs.shape}")
        # logging.info(f"buffer.rnn_states_actor shape: {self.buffer.rnn_states_actor.shape}")
        # logging.info(f"buffer.rnn_states_critic shape: {self.buffer.rnn_states_critic.shape}")
        # logging.info(f"buffer.masks shape: {self.buffer.masks.shape}")
        # logging.info("=== Buffer创建后检查结束 ===")

        # # [Selfplay] allocate memory for opponent policy/data in training
        # if self.use_selfplay:
        #     # 自博弈算法选择器
        #     from algorithms.utils.selfplay import get_algorithm
        #     self.selfplay_algo = get_algorithm(self.all_args.selfplay_algorithm)

        #     # 断言检查：确保对手数量不超过训练线程数
        #     assert self.all_args.n_choose_opponents <= self.n_rollout_threads, \
        #         "Number of different opponents({}) must less than or equal to number of training threads({})!" \
        #         .format(self.all_args.n_choose_opponents, self.n_rollout_threads)
            
        #     # 策略池管理 - ELO评分系统
        #     self.policy_pool = {'latest': self.all_args.init_elo}  # type: dict[str, float]
            
        #     # 创建敌机专用的观测空间：15维（自身9维 + 我方第一架飞机6维）
        #     from gymnasium import spaces
        #     opponent_obs_space = spaces.Box(
        #         low=-10, high=10., 
        #         shape=(15,),  # 敌机自身9维 + 我方第一架飞机6维
        #         dtype=np.float32
        #     )
            
        #     self.opponent_policy = [
        #         Policy(self.all_args, opponent_obs_space, self.share_obs_space, self.act_space, device=self.device)
        #         for _ in range(self.all_args.n_choose_opponents)]
            
        #     # 环境分配策略
        #     self.opponent_env_split = np.array_split(np.arange(self.n_rollout_threads), len(self.opponent_policy))
            
        #     # 对手数据存储 (蓝方飞机) - 使用15维观察空间
        #     # 创建正确维度的对手观察数据存储
        #     # 对手数据维度应该与蓝方飞机数量匹配，只有1个蓝方飞机
        #     num_opponents = 1  # 蓝方飞机数量
        #     logging.info(f"对手数据维度: 蓝方飞机数量 = {num_opponents}")
            
        #     # 创建正确维度的对手数据
        #     opponent_obs_shape = [self.n_rollout_threads, num_opponents, 15]  # [32, 1, 15]
        #     self.opponent_obs = np.zeros(opponent_obs_shape, dtype=self.buffer.obs[0].dtype)

        #     # 在对手数据初始化后检查
        #     logging.info("=== 对手数据初始化后检查 ===")
        #     logging.info(f"opponent_obs shape: {self.opponent_obs.shape}")
            
        #     # 检查对手数据是否有异常
        #     if 0 in self.opponent_obs.shape:
        #         logging.error(f"错误: opponent_obs包含0维度: {self.opponent_obs.shape}")
            
        #     logging.info("=== 对手数据初始化后检查结束 ===")

        #     # 评估用对手策略
        #     if self.use_eval:
        #         self.eval_opponent_policy = Policy(self.all_args, opponent_obs_space, self.share_obs_space, self.act_space, device=self.device)

        # policy & algorithm
        if self.algorithm_name == "mappo":
            from algorithms.mappo.ppo_trainer import PPOTrainer as Trainer
            from algorithms.mappo.ppo_policy import PPOPolicy as Policy
        else:
            raise NotImplementedError
        self.policy = Policy(self.all_args, self.obs_space, self.share_obs_space, self.act_space, device=self.device)
        self.trainer = Trainer(self.all_args, device=self.device)

        # buffer - 只存储红方智能体的经验 (2个)
        if self.use_selfplay:
            self.buffer = SharedReplayBuffer(self.all_args, 2, self.obs_space, self.share_obs_space, self.act_space)
        else:
            self.buffer = SharedReplayBuffer(self.all_args, self.num_agents, self.obs_space, self.share_obs_space, self.act_space)

        # 在创建buffer后立即检查
        logging.info("=== Buffer创建后检查 ===")
        logging.info(f"buffer.obs shape: {self.buffer.obs.shape}")
        logging.info(f"buffer.share_obs shape: {self.buffer.share_obs.shape}")
        logging.info(f"buffer.rnn_states_actor shape: {self.buffer.rnn_states_actor.shape}")
        logging.info(f"buffer.rnn_states_critic shape: {self.buffer.rnn_states_critic.shape}")
        logging.info(f"buffer.masks shape: {self.buffer.masks.shape}")
        logging.info("=== Buffer创建后检查结束 ===")

        # [Selfplay] allocate memory for opponent policy/data in training
        if self.use_selfplay:
            # 自博弈算法选择器
            from algorithms.utils.selfplay import get_algorithm
            self.selfplay_algo = get_algorithm(self.all_args.selfplay_algorithm)

            # 断言检查：确保对手数量不超过训练线程数
            assert self.all_args.n_choose_opponents <= self.n_rollout_threads, \
                "Number of different opponents({}) must less than or equal to number of training threads({})!" \
                .format(self.all_args.n_choose_opponents, self.n_rollout_threads)
            
            # 策略池管理 - ELO评分系统
            self.policy_pool = {'latest': self.all_args.init_elo}  # type: dict[str, float]
            
            # 创建敌机专用的观测空间：15维（自身9维 + 我方第一架飞机6维）
            from gymnasium import spaces
            opponent_obs_space = spaces.Box(
                low=-10, high=10., 
                shape=(15,),  # 敌机自身9维 + 我方第一架飞机6维
                dtype=np.float32
            )
            
            # 蓝方飞机使用PPO算法，不是MAPPO
            from algorithms.ppo.ppo_policy import PPOPolicy as OpponentPolicy
            self.opponent_policy = [
                OpponentPolicy(self.all_args, opponent_obs_space, self.act_space, device=self.device)
                for _ in range(self.all_args.n_choose_opponents)]
            
            # 环境分配策略
            self.opponent_env_split = np.array_split(np.arange(self.n_rollout_threads), len(self.opponent_policy))
            
            # 对手数据存储 (蓝方飞机) - 使用15维观察空间
            # 创建正确维度的对手观察数据存储
            # 对手数据维度应该与蓝方飞机数量匹配，只有1个蓝方飞机
            num_opponents = 1  # 蓝方飞机数量
            logging.info(f"对手数据维度: 蓝方飞机数量 = {num_opponents}")
            
            # 创建正确维度的对手数据
            opponent_obs_shape = [self.n_rollout_threads, num_opponents, 15]  # [32, 1, 15]
            self.opponent_obs = np.zeros(opponent_obs_shape, dtype=self.buffer.obs[0].dtype)

            # 在对手数据初始化后检查
            logging.info("=== 对手数据初始化后检查 ===")
            logging.info(f"opponent_obs shape: {self.opponent_obs.shape}")
            
            # 检查对手数据是否有异常
            if 0 in self.opponent_obs.shape:
                logging.error(f"错误: opponent_obs包含0维度: {self.opponent_obs.shape}")
            
            logging.info("=== 对手数据初始化后检查结束 ===")

            # 评估用对手策略 - 蓝方也使用PPO
            if self.use_eval:
                self.eval_opponent_policy = OpponentPolicy(self.all_args, opponent_obs_space, self.act_space, device=self.device)
            # 定义可选的敌方策略文件路径
            # self.opponent_model_paths = {
            #     250: "results/SingleCombat/1v1/NoWeapon/HierarchySelfplay/ppo/v1/wandb/latest-run/files/actor_250.pt",
            #     440: "results/SingleCombat/1v1/NoWeapon/HierarchySelfplay/ppo/v1/wandb/latest-run/files/actor_440.pt",
            #     1040: "results/SingleCombat/1v1/NoWeapon/HierarchySelfplay/ppo/v1/wandb/latest-run/files/actor_1040.pt"
            # }
            
            self.base_model_path = "results/SingleCombat/1v1/NoWeapon/HierarchySelfplay/ppo/v1/wandb/latest-run/files/actor_{}.pt"
            self.model_checkpoints = [250, 440, 1040]  # 只需要修改这个列表
            
            # 自动生成opponent_model_paths
            self.opponent_model_paths = {
                checkpoint: self.base_model_path.format(checkpoint) 
                for checkpoint in self.model_checkpoints
            }
            # 随机选择初始策略
            initial_strategy = random.choice([250, 440, 1040])
            initial_model_path = self.opponent_model_paths[initial_strategy]

            
            logging.info(f"Loading initial opponent model from: {initial_model_path} (strategy: {initial_strategy})")
            
            for policy in self.opponent_policy:
                try:
                    policy.actor.load_state_dict(torch.load(initial_model_path, map_location=self.device))
                    policy.prep_rollout()
                    logging.info(f"Successfully loaded initial opponent model (strategy: {initial_strategy})")
                except Exception as e:
                    logging.error(f"Failed to load initial opponent model: {e}")
                    logging.info("Using randomly initialized opponent model instead")

            logging.info(f"\n Load initial opponent: Using actor_{initial_strategy}.pt for all opponents.\n")

        if self.model_dir is not None:
            self.restore()

    def build_opponent_obs(self, obs):
        """
        为敌机构建独立的观测空间：15维
        - 敌机自身9维（0-8）
        - 我方第一架飞机6维（9-14）
        
        Args:
            obs: 原始观测数据，shape为 [n_envs, n_agents, obs_dim]
        
        Returns:
            opponent_obs: 敌机观测数据，shape为 [n_envs, 1, 15]
        """
        n_envs = obs.shape[0]
        opponent_obs = np.zeros((n_envs, 1, 15), dtype=obs.dtype)
        
        # 敌机自身信息：前9维（0-8）
        opponent_obs[:, 0, :9] = obs[:, 2, :9]  # 蓝方飞机（索引2）的前9维
        
        # 我方第一架飞机信息：6维（9-14）
        # 从原始观测中提取我方第一架飞机相对于敌机的信息
        # 原始观测中，我方第一架飞机的相对信息在索引9-14
        opponent_obs[:, 0, 9:15] = obs[:, 2, 9:15]  # 蓝方飞机观测中关于我方第一架飞机的6维信息
        
        return opponent_obs

    def run(self):
        self.warmup()

        start = time.time()
        self.total_num_steps = 0
        episodes = self.num_env_steps // self.buffer_size // self.n_rollout_threads

        for episode in range(episodes):

            for step in range(self.buffer_size):
                # Sample actions
                values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.collect(step)

                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos = self.envs.step(actions)

                data = obs, share_obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            self.total_num_steps = (episode + 1) * self.buffer_size * self.n_rollout_threads

            # save model
            if (episode % self.save_interval == 0) or (episode == episodes - 1):
                self.save(episode)

            # log information
            # if episode % self.log_interval == 0:
            if episode % 2 == 0:
                end = time.time()
                logging.info("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                             .format(self.all_args.scenario_name,
                                     self.algorithm_name,
                                     self.experiment_name,
                                     episode,
                                     episodes,
                                     self.total_num_steps,
                                     self.num_env_steps,
                                     int(self.total_num_steps / (end - start))))

                train_infos["average_episode_rewards"] = self.buffer.rewards.sum() / (self.buffer.masks == False).sum()
                logging.info("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_info(train_infos, self.total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(self.total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs = self.envs.reset()
        # [Selfplay] divide ego/opponent of initial obs
        if self.use_selfplay:
            # 在数据分割前添加检查
            # logging.info("=== Warmup中数据分割前检查 ===")
            # logging.info(f"原始obs shape: {obs.shape}")
            # logging.info(f"原始share_obs shape: {share_obs.shape}")
            
            # # 检查数据分割的逻辑
            # logging.info(f"红方飞机数量: 2 (A0100, A0200)")
            # logging.info(f"蓝方飞机数量: {obs.shape[1] - 2} (B0100, B0200...)")
            # logging.info(f"总智能体数量: {obs.shape[1]}")
            
            # 为敌机构建独立的观测空间
            self.opponent_obs = self.build_opponent_obs(obs)
            
            # 确保敌方观察数据不为空
            if self.opponent_obs.size == 0:
                logging.warning("Warning: opponent_obs is empty in warmup!")
            
            obs = obs[:, :2, ...]                # 红方飞机 (A0100, A0200)
            share_obs = share_obs[:, :2, ...]
            
            logging.info(f"分割后红方obs shape: {obs.shape}")
            logging.info(f"分割后蓝方opponent_obs shape: {self.opponent_obs.shape}")
            logging.info("=== Warmup中数据分割前检查结束 ===")
            
        self.buffer.step = 0
        self.buffer.obs[0] = obs.copy()
        self.buffer.share_obs[0] = share_obs.copy()


          
    @torch.no_grad()
    def collect(self, step):
        self.policy.prep_rollout()
        
        # 在策略调用前添加检查
        # if step == 0:
        #     logging.info("=== Collect中策略调用前检查 ===")
        #     logging.info(f"step: {step}")
        #     logging.info(f"buffer.share_obs[{step}] shape: {self.buffer.share_obs[step].shape}")
        #     logging.info(f"buffer.obs[{step}] shape: {self.buffer.obs[step].shape}")
        #     logging.info(f"buffer.rnn_states_actor[{step}] shape: {self.buffer.rnn_states_actor[step].shape}")
        #     logging.info(f"buffer.rnn_states_critic[{step}] shape: {self.buffer.rnn_states_critic[step].shape}")
        #     logging.info(f"buffer.masks[{step}] shape: {self.buffer.masks[step].shape}")
            
        #     if self.use_selfplay:
        #         logging.info(f"opponent_obs shape: {self.opponent_obs.shape}")
            
        #     logging.info("=== Collect中策略调用前检查结束 ===")
        
        values, actions, action_log_probs, rnn_states_actor, rnn_states_critic \
            = self.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                      np.concatenate(self.buffer.obs[step]),
                                      np.concatenate(self.buffer.rnn_states_actor[step]),
                                      np.concatenate(self.buffer.rnn_states_critic[step]),
                                      np.concatenate(self.buffer.masks[step]))
        # split parallel data [N*M, shape] => [N, M, shape]
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states_actor = np.array(np.split(_t2n(rnn_states_actor), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        # [Selfplay] get actions of opponent policy (蓝方飞机)
        if self.use_selfplay:
            # 修复：创建正确形状的opponent_actions数组
            opponent_actions = np.zeros((self.n_rollout_threads, 1, actions.shape[2]), dtype=actions.dtype)
            for policy_idx, policy in enumerate(self.opponent_policy):
                env_idx = self.opponent_env_split[policy_idx]
                # 修复索引问题：确保env_idx不为空且正确索引
                if len(env_idx) > 0:
                    opponent_obs_batch = self.opponent_obs[env_idx]
                    
                    # 在对手策略调用前添加调试
                    # if step == 0:
                    #     logging.info(f"=== 对手策略调用调试 ===")
                    #     logging.info(f"policy_idx: {policy_idx}, env_idx: {env_idx}")
                    #     logging.info(f"opponent_obs_batch shape: {opponent_obs_batch.shape}")
                    #     logging.info(f"=== 对手策略调用调试结束 ===")
                    
                    # 确保数据维度正确
                    if opponent_obs_batch.size > 0:
                        # 使用敌机专用的15维观测数据
                        opponent_obs_15d = np.concatenate(opponent_obs_batch)  # shape: (n_envs, 15)
                        
                        # 为蓝方创建虚拟的RNN状态和掩码（因为使用固定策略）
                        n_envs = len(env_idx)
                        dummy_rnn_states = np.zeros((n_envs, 1, 128), dtype=np.float32)  # 假设RNN隐藏维度为128
                        dummy_masks = np.ones((n_envs, 1), dtype=np.float32)
                        
                        try:
                            # 蓝方使用固定策略，但需要提供RNN状态和掩码参数
                            opponent_action, _ = policy.act(opponent_obs_15d, 
                                                          dummy_rnn_states, 
                                                          dummy_masks, 
                                                          deterministic=True)
                            
                            # 重塑回原始维度
                            opponent_action_reshaped = np.array(np.split(_t2n(opponent_action), len(env_idx)))
                            
                            # 修复动作赋值逻辑
                            opponent_actions[env_idx] = opponent_action_reshaped
                            
                            # 添加动作验证
                            # if step == 0:
                            #     logging.info(f"=== 对手动作生成成功 ===")
                            #     logging.info(f"opponent_action shape: {opponent_action.shape}")
                            #     logging.info(f"opponent_action_reshaped shape: {opponent_action_reshaped.shape}")
                            #     logging.info(f"opponent_actions shape: {opponent_actions.shape}")
                            #     logging.info(f"opponent_actions[env_idx] shape: {opponent_actions[env_idx].shape}")
                            #     logging.info(f"=== 对手动作生成成功结束 ===")
                                
                        except Exception as e:
                            logging.error(f"对手策略调用失败: {e}")
                            # 使用随机动作作为备选
                            random_action = np.random.randint(0, 3, size=(len(env_idx), 1, 3))  # 假设动作空间为3维
                            opponent_actions[env_idx] = random_action
                            logging.warning("使用随机动作作为备选")
            
            # 验证动作拼接前的维度
            # if step == 0:
            #     logging.info(f"=== 动作拼接前检查 ===")
            #     logging.info(f"actions shape: {actions.shape}")
            #     logging.info(f"opponent_actions shape: {opponent_actions.shape}")
            #     logging.info(f"=== 动作拼接前检查结束 ===")
            
            # 确保动作维度正确
            if actions.shape[1] == 2 and opponent_actions.shape[1] == 1:
                actions = np.concatenate((actions, opponent_actions), axis=1)
                # if step == 0:
                #     logging.info(f"=== 动作拼接后检查 ===")
                #     logging.info(f"拼接后actions shape: {actions.shape}")
                #     logging.info(f"=== 动作拼接后检查结束 ===")
            else:
                logging.error(f"动作维度不匹配: actions.shape={actions.shape}, opponent_actions.shape={opponent_actions.shape}")
                # 创建正确维度的动作
                if actions.shape[1] == 2:
                    # 如果只有红方动作，添加蓝方动作
                    if opponent_actions.shape[1] == 0:
                        # 创建随机蓝方动作
                        random_opponent_actions = np.random.randint(0, 3, size=(actions.shape[0], 1, actions.shape[2]))
                        actions = np.concatenate((actions, random_opponent_actions), axis=1)
                    else:
                        actions = np.concatenate((actions, opponent_actions), axis=1)
                else:
                    logging.error(f"无法修复动作维度: actions.shape={actions.shape}")

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
    @torch.no_grad()
    def compute(self):
        self.policy.prep_rollout()
        next_values = self.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                             np.concatenate(self.buffer.rnn_states_critic[-1]),
                                             np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.buffer.n_rollout_threads))
        self.buffer.compute_returns(next_values)

    def insert(self, data: List[np.ndarray]):
        obs, share_obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic = data
        dones = dones.squeeze(axis=-1)
        dones_env = np.all(dones, axis=-1)

        rnn_states_actor[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_actor.shape[1:]), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_critic.shape[1:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
        
        # [Selfplay] divide ego/opponent of collecting data
        if self.use_selfplay:
            # 确保敌方数据不为空
            if obs.shape[1] >= 3:  # 确保有足够的智能体
                # 为敌机构建独立的观测空间
                self.opponent_obs = self.build_opponent_obs(obs)
            else:
                logging.warning(f"Warning: Not enough agents for opponent data. obs shape: {obs.shape}")
                # 创建空的敌方数据
                self.opponent_obs = np.zeros((self.n_rollout_threads, 0, 15, *obs.shape[3:]), dtype=obs.dtype)

            obs = obs[:, :2, ...]                # 红方飞机 (A0100, A0200)
            share_obs = share_obs[:, :2, ...]
            actions = actions[:, :2, ...]
            rewards = rewards[:, :2, ...]
            masks = masks[:, :2, ...]
            active_masks = active_masks[:, :2, ...]

        self.buffer.insert(obs, share_obs, actions, rewards, masks, action_log_probs, values, \
            rnn_states_actor, rnn_states_critic, active_masks = active_masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        logging.info("\nStart evaluation...")
        total_episodes, eval_episode_rewards = 0, []
        eval_cumulative_rewards = np.zeros((self.n_eval_rollout_threads, *self.buffer.rewards.shape[2:]), dtype=np.float32)

        eval_obs, eval_share_obs = self.eval_envs.reset()
        eval_masks = np.ones((self.n_eval_rollout_threads, *self.buffer.masks.shape[2:]), dtype=np.float32)
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)


        # [Selfplay] Choose opponent policy for evaluation
        if self.use_selfplay:
            eval_choose_opponents = [self.selfplay_algo.choose(self.policy_pool) for _ in range(self.all_args.n_choose_opponents)]
            assert self.eval_episodes >= self.all_args.n_choose_opponents, \
            f"Number of evaluation episodes:{self.eval_episodes} should be greater than number of opponents:{self.all_args.n_choose_opponents}"
            eval_each_episodes = self.eval_episodes // self.all_args.n_choose_opponents
            eval_cur_opponent_idx = 0
            logging.info(f" Choose opponents {eval_choose_opponents} for evaluation")
            # TODO: use eval results to update elo

        while total_episodes < self.eval_episodes:

            # [Selfplay] Load opponent policy - 随机选择策略进行评估
            if self.use_selfplay and total_episodes >= eval_cur_opponent_idx * eval_each_episodes:
                # 随机选择策略进行评估
                chosen_strategy = random.choice([250, 440, 1040])
                eval_model_path = self.opponent_model_paths[chosen_strategy]
                self.eval_opponent_policy.actor.load_state_dict(torch.load(eval_model_path, weights_only=True))
                self.eval_opponent_policy.prep_rollout()
                eval_cur_opponent_idx += 1
                logging.info(f" Load random opponent (actor_{chosen_strategy}.pt) for evaluation ({total_episodes+1}/{self.eval_episodes})")

                # reset obs/rnn/mask
                eval_obs, eval_share_obs = self.eval_envs.reset()
                eval_masks = np.ones_like(eval_masks, dtype=np.float32)
                eval_rnn_states = np.zeros_like(eval_rnn_states, dtype=np.float32)
                
                # 为敌机构建独立的观测空间
                eval_opponent_obs = self.build_opponent_obs(eval_obs)
                eval_obs = eval_obs[:, :2, ...]           # 红方飞机
                
            self.policy.prep_rollout()
            eval_actions, eval_rnn_states = self.policy.act(np.concatenate(eval_obs),
                                                            np.concatenate(eval_rnn_states),
                                                            np.concatenate(eval_masks), deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # [Selfplay] get actions of opponent policy
            # if self.use_selfplay:
            #     logging.info(f"=== 评估阶段对手策略调用前检查 ===")
            #     logging.info(f"eval_opponent_obs shape: {eval_opponent_obs.shape}")
            #     logging.info(f"=== 评估阶段对手策略调用前检查结束 ===")
                
            #     # 蓝方使用固定策略，不需要RNN状态和掩码
            #     eval_opponent_actions = self.eval_opponent_policy.act(np.concatenate(eval_opponent_obs), deterministic=True)
            #     eval_opponent_actions = np.array(np.split(_t2n(eval_opponent_actions), self.n_eval_rollout_threads))
            #     eval_actions = np.concatenate((eval_actions, eval_opponent_actions), axis=1)
            if self.use_selfplay:
                # logging.info(f"=== 评估阶段对手策略调用前检查 ===")
                # logging.info(f"eval_opponent_obs shape: {eval_opponent_obs.shape}")
                # logging.info(f"=== 评估阶段对手策略调用前检查结束 ===")
                
                # 为蓝方创建虚拟的RNN状态和掩码
                n_eval_envs = self.n_eval_rollout_threads
                dummy_rnn_states = np.zeros((n_eval_envs, 1, 128), dtype=np.float32)
                dummy_masks = np.ones((n_eval_envs, 1), dtype=np.float32)
                
                # 蓝方使用固定策略，但需要提供RNN状态和掩码参数
                eval_opponent_actions, _ = self.eval_opponent_policy.act(np.concatenate(eval_opponent_obs), 
                                                                        dummy_rnn_states, 
                                                                        dummy_masks, 
                                                                        deterministic=True)
                eval_opponent_actions = np.array(np.split(_t2n(eval_opponent_actions), self.n_eval_rollout_threads))
                eval_actions = np.concatenate((eval_actions, eval_opponent_actions), axis=1)                
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)

            # [Selfplay] get ego reward
            if self.use_selfplay:
                eval_rewards = eval_rewards[:, :2, ...]  # 只保留红方奖励

            eval_cumulative_rewards += eval_rewards
            eval_dones_env = np.all(eval_dones.squeeze(axis=-1), axis=-1)
            total_episodes += np.sum(eval_dones_env)
            eval_episode_rewards.append(eval_cumulative_rewards[eval_dones_env == True])
            eval_cumulative_rewards[eval_dones_env == True] = 0

            eval_masks = np.ones_like(eval_masks, dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), *eval_masks.shape[1:]), dtype=np.float32)
            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), *eval_rnn_states.shape[1:]), dtype=np.float32)
            # [Selfplay] reset opponent mask/rnn_states
            if self.use_selfplay:
                # 为敌机构建独立的观测空间
                eval_opponent_obs = self.build_opponent_obs(eval_obs)
                eval_obs = eval_obs[:, :2, ...]           # 红方飞机

        eval_infos = {}
        eval_infos['eval_average_episode_rewards'] = np.concatenate(eval_episode_rewards).mean() 
        logging.info(" eval average episode rewards: " + str(eval_infos['eval_average_episode_rewards']))
        self.log_info(eval_infos, total_num_steps)

        # [Selfplay] Reset opponent
        if self.use_selfplay:
            self.reset_opponent()
        logging.info("...End evaluation")

    @torch.no_grad()
    def render(self):
        logging.info("\nStart render ...")
        self.render_opponent_index = self.all_args.render_opponent_index
        render_episode_rewards = 0
        render_obs, render_share_obs = self.envs.reset()
        render_masks = np.ones((1, *self.buffer.masks.shape[2:]), dtype=np.float32)
        render_rnn_states = np.zeros((1, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
        self.envs.render(mode='txt', filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi')
        if self.use_selfplay:
            # 随机选择策略进行渲染
            chosen_strategy = random.choice([250, 440, 1040])
            render_model_path = self.opponent_model_paths[chosen_strategy]
            self.eval_opponent_policy.actor.load_state_dict(torch.load(render_model_path, weights_only=True))
            self.eval_opponent_policy.prep_rollout()
            logging.info(f"Using random opponent strategy (actor_{chosen_strategy}.pt) for rendering")
            # reset obs/rnn/mask
            render_obs, render_share_obs = self.envs.reset()
            render_masks = np.ones_like(render_masks, dtype=np.float32)
            render_rnn_states = np.zeros_like(render_rnn_states, dtype=np.float32)
            
            # 为敌机构建独立的观测空间
            render_opponent_obs = self.build_opponent_obs(render_obs)
            render_obs = render_obs[:, :2, ...]           # 红方飞机
            
        while True:
            self.policy.prep_rollout()
            render_actions, render_rnn_states = self.policy.act(np.concatenate(render_obs),
                                                                np.concatenate(render_rnn_states),
                                                                np.concatenate(render_masks),
                                                                deterministic=True)
            render_actions = np.expand_dims(_t2n(render_actions), axis=0)
            render_rnn_states = np.expand_dims(_t2n(render_rnn_states), axis=0)
            
            # [Selfplay] get actions of opponent policy
            # if self.use_selfplay:
            #     # 蓝方使用固定策略，不需要RNN状态和掩码
            #     render_opponent_actions = self.eval_opponent_policy.act(np.concatenate(render_opponent_obs), deterministic=True)
            #     render_opponent_actions = np.expand_dims(_t2n(render_opponent_actions), axis=0)
            #     render_actions = np.concatenate((render_actions, render_opponent_actions), axis=1)
                
            # [Selfplay] get actions of opponent policy
            if self.use_selfplay:
                # 为蓝方创建虚拟的RNN状态和掩码
                n_render_envs = 1
                dummy_rnn_states = np.zeros((n_render_envs, 1, 128), dtype=np.float32)
                dummy_masks = np.ones((n_render_envs, 1), dtype=np.float32)
                
                # 蓝方使用固定策略，但需要提供RNN状态和掩码参数
                render_opponent_actions, _ = self.eval_opponent_policy.act(np.concatenate(render_opponent_obs), 
                                                                         dummy_rnn_states, 
                                                                         dummy_masks, 
                                                                         deterministic=True)
                render_opponent_actions = np.expand_dims(_t2n(render_opponent_actions), axis=0)
                render_actions = np.concatenate((render_actions, render_opponent_actions), axis=1)
            # Obser reward and next obs
            render_obs, render_share_obs, render_rewards, render_dones, render_infos = self.envs.step(render_actions)





            if self.use_selfplay:
                render_rewards = render_rewards[:, :2, ...]  # 只保留红方奖励
            render_episode_rewards += render_rewards
            self.envs.render(mode='txt', filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi')
            if render_dones.all():
                break
            if self.use_selfplay:
                # 为敌机构建独立的观测空间
                render_opponent_obs = self.build_opponent_obs(render_obs)
                render_obs = render_obs[:, :2, ...]           # 红方飞机

        render_infos = {}
        render_infos['render_episode_reward'] = render_episode_rewards
        logging.info("render episode reward of agent: " + str(render_infos['render_episode_reward']))

    def save(self, episode):
        policy_actor_state_dict = self.policy.actor.state_dict()
        torch.save(policy_actor_state_dict, str(self.save_dir) + '/actor_latest.pt')
        policy_critic_state_dict = self.policy.critic.state_dict()
        torch.save(policy_critic_state_dict, str(self.save_dir) + '/critic_latest.pt')
        # [Selfplay] save policy & performance
        if self.use_selfplay:
            torch.save(policy_actor_state_dict, str(self.save_dir) + f'/actor_{episode}.pt')
            self.policy_pool[str(episode)] = self.all_args.init_elo

    def reset_opponent(self):
        # 随机选择策略进行训练
        chosen_strategies = []
        for policy in self.opponent_policy:
            chosen_strategy = random.choice([250, 440, 1040])
            chosen_strategies.append(chosen_strategy)
            model_path = self.opponent_model_paths[chosen_strategy]
            try:
                policy.actor.load_state_dict(torch.load(model_path, map_location=self.device))
                policy.prep_rollout()
            except Exception as e:
                logging.error(f"Failed to load opponent model {chosen_strategy} in reset: {e}")
        logging.info(f"Using random opponent strategies {chosen_strategies} for training")

        # clear buffer
        self.buffer.clear()
        self.opponent_obs = np.zeros_like(self.opponent_obs)

        # reset env
        obs, share_obs = self.envs.reset()
        if self.all_args.n_choose_opponents > 0:
            # 为敌机构建独立的观测空间
            self.opponent_obs = self.build_opponent_obs(obs)
            obs = obs[:, :2, ...]                # 红方飞机
            share_obs = share_obs[:, :2, ...]
        self.buffer.obs[0] = obs.copy()
        self.buffer.share_obs[0] = share_obs.copy()