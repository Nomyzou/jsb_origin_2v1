import logging
import time
from typing import List

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

        # [Fixed Opponent] allocate memory for fixed opponent policy/data in training
        # if self.use_selfplay:
        #     # 创建固定对手策略 (蓝方飞机)
        #     from gymnasium import spaces
        #     opponent_obs_space = spaces.Box(
        #         low=-10, high=10., 
        #         shape=(15,),  # Only use first 15 dimensions
        #         dtype=np.float32
        #     )
            
        #     self.opponent_policy = Policy(self.all_args, opponent_obs_space, self.share_obs_space, self.act_space, device=self.device)
            
        #     # 对手数据存储 (蓝方飞机) - 使用15维观察空间
        #     num_opponents = 1  # 蓝方飞机数量
        #     logging.info(f"对手数据维度: 蓝方飞机数量 = {num_opponents}")
            
        #     # 创建正确维度的对手数据
        #     opponent_obs_shape = [self.n_rollout_threads, num_opponents, 15]  # [32, 1, 15]
        #     self.opponent_obs = np.zeros(opponent_obs_shape, dtype=self.buffer.obs[0].dtype)
            
        #     # 创建正确维度的RNN状态
        #     opponent_rnn_shape = [self.n_rollout_threads, num_opponents, 1, 128]  # [32, 1, 1, 128]
        #     self.opponent_rnn_states = np.zeros(opponent_rnn_shape, dtype=self.buffer.rnn_states_actor[0].dtype)
            
        #     # 创建正确维度的掩码
        #     opponent_mask_shape = [self.n_rollout_threads, num_opponents, 1]  # [32, 1, 1]
        #     self.opponent_masks = np.ones(opponent_mask_shape, dtype=self.buffer.masks[0].dtype)

        #     # 评估用对手策略
        #     if self.use_eval:
        #         self.eval_opponent_policy = Policy(self.all_args, opponent_obs_space, self.share_obs_space, self.act_space, device=self.device)
        if self.use_selfplay:
            # 创建固定对手策略 (蓝方飞机)
            from gymnasium import spaces
            opponent_obs_space = spaces.Box(
                low=-10, high=10., 
                shape=(21,),  # 前9维 + 6个0 + 后6维 = 21维
                dtype=np.float32
            )
            
            self.opponent_policy = Policy(self.all_args, opponent_obs_space, self.share_obs_space, self.act_space, device=self.device)
            
            # 对手数据存储 (蓝方飞机) - 使用21维观察空间
            num_opponents = 1  # 蓝方飞机数量
            logging.info(f"对手数据维度: 蓝方飞机数量 = {num_opponents}")
            
            # 创建正确维度的对手数据
            opponent_obs_shape = [self.n_rollout_threads, num_opponents, 21]  # [32, 1, 21]
            self.opponent_obs = np.zeros(opponent_obs_shape, dtype=self.buffer.obs[0].dtype)
            
            # 创建正确维度的RNN状态
            opponent_rnn_shape = [self.n_rollout_threads, num_opponents, 1, 128]  # [32, 1, 1, 128]
            self.opponent_rnn_states = np.zeros(opponent_rnn_shape, dtype=self.buffer.rnn_states_actor[0].dtype)
            
            # 创建正确维度的掩码
            opponent_mask_shape = [self.n_rollout_threads, num_opponents, 1]  # [32, 1, 1]
            self.opponent_masks = np.ones(opponent_mask_shape, dtype=self.buffer.masks[0].dtype)

            # 评估用对手策略
            if self.use_eval:
                self.eval_opponent_policy = Policy(self.all_args, opponent_obs_space, self.share_obs_space, self.act_space, device=self.device)

            # 加载预训练的敌方模型
            if hasattr(self.all_args, 'opponent_model_path') and self.all_args.opponent_model_path is not None:
                logging.info(f"Loading pre-trained opponent model from: {self.all_args.opponent_model_path}")
                try:
                    self.opponent_policy.actor.load_state_dict(torch.load(self.all_args.opponent_model_path, map_location=self.device))
                    self.opponent_policy.prep_rollout()
                    logging.info("Successfully loaded pre-trained opponent model")
                    
                    # 同时加载到评估策略
                    if self.use_eval:
                        self.eval_opponent_policy.actor.load_state_dict(torch.load(self.all_args.opponent_model_path, map_location=self.device))
                        self.eval_opponent_policy.prep_rollout()
                        logging.info("Successfully loaded pre-trained opponent model for evaluation")
                        
                except Exception as e:
                    logging.error(f"Failed to load opponent model: {e}")
                    logging.info("Using randomly initialized opponent model instead")
            else:
                logging.warning("No opponent model path provided, using randomly initialized opponent model")

            logging.info("\n Load fixed opponent model for blue aircraft.\n")

        if self.model_dir is not None:
            self.restore()

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
            if episode % self.log_interval == 0:
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
        # [Fixed Opponent] divide ego/opponent of initial obs
        if self.use_selfplay:
            # 在数据分割前添加检查
            logging.info("=== Warmup中数据分割前检查 ===")
            logging.info(f"原始obs shape: {obs.shape}")
            logging.info(f"原始share_obs shape: {share_obs.shape}")
            
            # 检查数据分割的逻辑
            logging.info(f"红方飞机数量: 2 (A0100, A0200)")
            logging.info(f"蓝方飞机数量: {obs.shape[1] - 2} (B0100, B0200...)")
            logging.info(f"总智能体数量: {obs.shape[1]}")
            
        #     # 只存储前15维观察数据，匹配预训练模型的输入维度
        #     self.opponent_obs = obs[:, 2:, :15, ...]  # 蓝方飞机 (B0100)，只取前15维
        #     # 确保敌方观察数据不为空
        #     if self.opponent_obs.size == 0:
        #         logging.warning("Warning: opponent_obs is empty in warmup!")
        #     obs = obs[:, :2, ...]                # 红方飞机 (A0100, A0200)
        #     share_obs = share_obs[:, :2, ...]
            
        #     logging.info(f"分割后红方obs shape: {obs.shape}")
        #     logging.info(f"分割后蓝方opponent_obs shape: {self.opponent_obs.shape}")
        #     logging.info("=== Warmup中数据分割前检查结束 ===")
            
        # self.buffer.step = 0
        # self.buffer.obs[0] = obs.copy()
        # self.buffer.share_obs[0] = share_obs.copy()
            # 构建21维观察数据：前9维 + 6个0 + 后6维
            if obs.shape[1] >= 3:  # 确保有足够的智能体
                # 获取原始15维数据
                original_obs = obs[:, 2:, :15, ...]  # 蓝方飞机 (B0100)，取前15维
                
                # 构建21维数据：前9维 + 6个0 + 后6维
                self.opponent_obs = np.zeros((original_obs.shape[0], original_obs.shape[1], 21, *original_obs.shape[3:]), dtype=original_obs.dtype)
                self.opponent_obs[:, :, :9, ...] = original_obs[:, :, :9, ...]  # 前9维
                self.opponent_obs[:, :, 9:15, ...] = 0  # 中间6个0
                self.opponent_obs[:, :, 15:21, ...] = original_obs[:, :, 9:15, ...]  # 后6维
            else:
                logging.warning("Warning: Not enough agents for opponent data in warmup!")
                self.opponent_obs = np.zeros((obs.shape[0], 0, 21, *obs.shape[3:]), dtype=obs.dtype)
            
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
        if step == 0:
            logging.info("=== Collect中策略调用前检查 ===")
            logging.info(f"step: {step}")
            logging.info(f"buffer.share_obs[{step}] shape: {self.buffer.share_obs[step].shape}")
            logging.info(f"buffer.obs[{step}] shape: {self.buffer.obs[step].shape}")
            logging.info(f"buffer.rnn_states_actor[{step}] shape: {self.buffer.rnn_states_actor[step].shape}")
            logging.info(f"buffer.rnn_states_critic[{step}] shape: {self.buffer.rnn_states_critic[step].shape}")
            logging.info(f"buffer.masks[{step}] shape: {self.buffer.masks[step].shape}")
            
            if self.use_selfplay:
                logging.info(f"opponent_obs shape: {self.opponent_obs.shape}")
                logging.info(f"opponent_rnn_states shape: {self.opponent_rnn_states.shape}")
                logging.info(f"opponent_masks shape: {self.opponent_masks.shape}")
            
            logging.info("=== Collect中策略调用前检查结束 ===")
        
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

        # [Fixed Opponent] get actions of fixed opponent policy (蓝方飞机)
        # if self.use_selfplay:
        #     # 修复：创建正确形状的opponent_actions数组
        #     opponent_actions = np.zeros((self.n_rollout_threads, 1, actions.shape[2]), dtype=actions.dtype)
            
        #     # 确保数据维度正确
        #     if self.opponent_obs.size > 0:
        #         # 只使用前15维观察数据，匹配预训练模型的输入维度
        #         opponent_obs_15d = np.concatenate(self.opponent_obs)[:, :15]
                
        #         # 修复RNN状态和掩码的维度问题
        #         opponent_rnn_concat = np.concatenate(self.opponent_rnn_states)  # (32, 1, 128)
        #         opponent_mask_concat = np.concatenate(self.opponent_masks)  # (32, 1)
                
        #         # 重塑为正确的维度
        #         opponent_rnn_reshaped = opponent_rnn_concat.reshape(-1, 1, opponent_rnn_concat.shape[-1])  # (32, 1, 128)
        #         opponent_mask_reshaped = opponent_mask_concat.reshape(-1, 1)  # (32, 1)
                
        #         try:
        #             opponent_action, opponent_rnn_states \
        #                 = self.opponent_policy.act(opponent_obs_15d,
        #                                            opponent_rnn_reshaped,
        #                                            opponent_mask_reshaped)
                    
        #             # 重塑回原始维度
        #             opponent_action_reshaped = np.array(np.split(_t2n(opponent_action), self.n_rollout_threads))
        #             opponent_rnn_reshaped = np.array(np.split(_t2n(opponent_rnn_states), self.n_rollout_threads))
                    
        #             opponent_actions = opponent_action_reshaped
        #             self.opponent_rnn_states = opponent_rnn_reshaped
                    
        #             # 添加动作验证
        #             if step == 0:
        #                 logging.info(f"=== 固定对手动作生成成功 ===")
        #                 logging.info(f"opponent_action shape: {opponent_action.shape}")
        #                 logging.info(f"opponent_action_reshaped shape: {opponent_action_reshaped.shape}")
        #                 logging.info(f"opponent_actions shape: {opponent_actions.shape}")
        #                 logging.info(f"=== 固定对手动作生成成功结束 ===")
                        
        #         except Exception as e:
        #             logging.error(f"固定对手策略调用失败: {e}")
        #             # 使用随机动作作为备选
        #             random_action = np.random.randint(0, 3, size=(self.n_rollout_threads, 1, 3))  # 假设动作空间为3维
        #             opponent_actions = random_action
        #             logging.warning("使用随机动作作为备选")
        if self.use_selfplay:
            # 修复：创建正确形状的opponent_actions数组
            opponent_actions = np.zeros((self.n_rollout_threads, 1, actions.shape[2]), dtype=actions.dtype)
            
            # 确保数据维度正确
            if self.opponent_obs.size > 0:
                # 使用21维观察数据
                opponent_obs_21d = np.concatenate(self.opponent_obs)[:, :21]
                
                # 修复RNN状态和掩码的维度问题
                opponent_rnn_concat = np.concatenate(self.opponent_rnn_states)  # (32, 1, 128)
                opponent_mask_concat = np.concatenate(self.opponent_masks)  # (32, 1)
                
                # 重塑为正确的维度
                opponent_rnn_reshaped = opponent_rnn_concat.reshape(-1, 1, opponent_rnn_concat.shape[-1])  # (32, 1, 128)
                opponent_mask_reshaped = opponent_mask_concat.reshape(-1, 1)  # (32, 1)
                
                try:
                    opponent_action, opponent_rnn_states \
                        = self.opponent_policy.act(opponent_obs_21d,
                                                   opponent_rnn_reshaped,
                                                   opponent_mask_reshaped)
                    
                    # 重塑回原始维度
                    opponent_action_reshaped = np.array(np.split(_t2n(opponent_action), self.n_rollout_threads))
                    opponent_rnn_reshaped = np.array(np.split(_t2n(opponent_rnn_states), self.n_rollout_threads))
                    
                    opponent_actions = opponent_action_reshaped
                    self.opponent_rnn_states = opponent_rnn_reshaped
                    
                    # 添加动作验证
                    if step == 0:
                        logging.info(f"=== 固定对手动作生成成功 ===")
                        logging.info(f"opponent_action shape: {opponent_action.shape}")
                        logging.info(f"opponent_action_reshaped shape: {opponent_action_reshaped.shape}")
                        logging.info(f"opponent_actions shape: {opponent_actions.shape}")
                        logging.info(f"=== 固定对手动作生成成功结束 ===")
                        
                except Exception as e:
                    logging.error(f"固定对手策略调用失败: {e}")
                    # 使用随机动作作为备选
                    random_action = np.random.randint(0, 3, size=(self.n_rollout_threads, 1, 3))  # 假设动作空间为3维
                    opponent_actions = random_action
                    logging.warning("使用随机动作作为备选")            
            # 验证动作拼接前的维度
            if step == 0:
                logging.info(f"=== 动作拼接前检查 ===")
                logging.info(f"actions shape: {actions.shape}")
                logging.info(f"opponent_actions shape: {opponent_actions.shape}")
                logging.info(f"=== 动作拼接前检查结束 ===")
            
            # 确保动作维度正确
            if actions.shape[1] == 2 and opponent_actions.shape[1] == 1:
                actions = np.concatenate((actions, opponent_actions), axis=1)
                if step == 0:
                    logging.info(f"=== 动作拼接后检查 ===")
                    logging.info(f"拼接后actions shape: {actions.shape}")
                    logging.info(f"=== 动作拼接后检查结束 ===")
            else:
                logging.error(f"动作维度不匹配: actions.shape={actions.shape}, opponent_actions.shape={opponent_actions.shape}")

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
        
        # [Fixed Opponent] divide ego/opponent of collecting data
        # if self.use_selfplay:
        #     # 确保敌方数据不为空
        #     if obs.shape[1] >= 3:  # 确保有足够的智能体
        #         # 只存储前15维观察数据，匹配预训练模型的输入维度
        #         self.opponent_obs = obs[:, 2:, :15, ...]  # 蓝方飞机 (B0100)，只取前15维
        #         self.opponent_masks = masks[:, 2:, ...]
        #     else:
        #         logging.warning(f"Warning: Not enough agents for opponent data. obs shape: {obs.shape}")
        #         # 创建空的敌方数据
        #         self.opponent_obs = np.zeros((self.n_rollout_threads, 0, 15, *obs.shape[3:]), dtype=obs.dtype)
        #         self.opponent_masks = np.zeros((self.n_rollout_threads, 0, *masks.shape[2:]), dtype=masks.dtype)
        if self.use_selfplay:
            # 确保敌方数据不为空
            if obs.shape[1] >= 3:  # 确保有足够的智能体
                # 获取原始15维数据
                original_obs = obs[:, 2:, :15, ...]  # 蓝方飞机 (B0100)，取前15维
                
                # 构建21维数据：前9维 + 6个0 + 后6维
                self.opponent_obs = np.zeros((original_obs.shape[0], original_obs.shape[1], 21, *original_obs.shape[3:]), dtype=original_obs.dtype)
                self.opponent_obs[:, :, :9, ...] = original_obs[:, :, :9, ...]  # 前9维
                self.opponent_obs[:, :, 9:15, ...] = 0  # 中间6个0
                self.opponent_obs[:, :, 15:21, ...] = original_obs[:, :, 9:15, ...]  # 后6维
                
                self.opponent_masks = masks[:, 2:, ...]
            else:
                logging.warning(f"Warning: Not enough agents for opponent data. obs shape: {obs.shape}")
                # 创建空的敌方数据
                self.opponent_obs = np.zeros((self.n_rollout_threads, 0, 21, *obs.shape[3:]), dtype=obs.dtype)
                self.opponent_masks = np.zeros((self.n_rollout_threads, 0, *masks.shape[2:]), dtype=masks.dtype)
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

        # [Fixed Opponent] Use fixed opponent policy for evaluation
        # if self.use_selfplay:
        #     logging.info(f" Using fixed opponent model for evaluation")
        #     # reset obs/rnn/mask
        #     eval_obs, eval_share_obs = self.eval_envs.reset()
        #     eval_masks = np.ones_like(eval_masks, dtype=np.float32)
        #     eval_rnn_states = np.zeros_like(eval_rnn_states, dtype=np.float32)
        #     eval_opponent_obs = eval_obs[:, 2:, :15, ...]  # 蓝方飞机，只取前15维
        #     eval_obs = eval_obs[:, :2, ...]           # 红方飞机
        #     eval_opponent_masks = np.ones((self.n_eval_rollout_threads, 1, 1), dtype=np.float32)
        #     eval_opponent_rnn_states = np.zeros((self.n_eval_rollout_threads, 1, 1, 128), dtype=np.float32)
        if self.use_selfplay:
            logging.info(f" Using fixed opponent model for evaluation")
            # reset obs/rnn/mask
            eval_obs, eval_share_obs = self.eval_envs.reset()
            eval_masks = np.ones_like(eval_masks, dtype=np.float32)
            eval_rnn_states = np.zeros_like(eval_rnn_states, dtype=np.float32)
            
            # 构建21维观察数据：前9维 + 6个0 + 后6维
            if eval_obs.shape[1] >= 3:
                original_eval_obs = eval_obs[:, 2:, :15, ...]  # 蓝方飞机，取前15维
                eval_opponent_obs = np.zeros((original_eval_obs.shape[0], original_eval_obs.shape[1], 21, *original_eval_obs.shape[3:]), dtype=original_eval_obs.dtype)
                eval_opponent_obs[:, :, :9, ...] = original_eval_obs[:, :, :9, ...]  # 前9维
                eval_opponent_obs[:, :, 9:15, ...] = 0  # 中间6个0
                eval_opponent_obs[:, :, 15:21, ...] = original_eval_obs[:, :, 9:15, ...]  # 后6维
            else:
                eval_opponent_obs = np.zeros((eval_obs.shape[0], 0, 21, *eval_obs.shape[3:]), dtype=eval_obs.dtype)
            
            eval_obs = eval_obs[:, :2, ...]           # 红方飞机
            eval_opponent_masks = np.ones((self.n_eval_rollout_threads, 1, 1), dtype=np.float32)
            eval_opponent_rnn_states = np.zeros((self.n_eval_rollout_threads, 1, 1, 128), dtype=np.float32)

        while total_episodes < self.eval_episodes:
            self.policy.prep_rollout()
            eval_actions, eval_rnn_states = self.policy.act(np.concatenate(eval_obs),
                                                            np.concatenate(eval_rnn_states),
                                                            np.concatenate(eval_masks), deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # [Fixed Opponent] get actions of fixed opponent policy
            if self.use_selfplay:
                eval_opponent_rnn_concat = np.concatenate(eval_opponent_rnn_states)  # (1, 1, 128)
                eval_opponent_mask_concat = np.concatenate(eval_opponent_masks)      # (1, 1)
                
                # 重塑为正确的维度 - 与训练阶段保持一致
                eval_opponent_rnn_reshaped = eval_opponent_rnn_concat.reshape(-1, 1, eval_opponent_rnn_concat.shape[-1])  # (1, 1, 128)
                eval_opponent_mask_reshaped = eval_opponent_mask_concat.reshape(-1, 1)  # (1, 1)
                
                eval_opponent_actions, eval_opponent_rnn_states \
                    = self.eval_opponent_policy.act(np.concatenate(eval_opponent_obs),
                                                    eval_opponent_rnn_reshaped,
                                                    eval_opponent_mask_reshaped)
                eval_opponent_rnn_states = np.array(np.split(_t2n(eval_opponent_rnn_states), self.n_eval_rollout_threads))
                eval_opponent_actions = np.array(np.split(_t2n(eval_opponent_actions), self.n_eval_rollout_threads))
                eval_actions = np.concatenate((eval_actions, eval_opponent_actions), axis=1)
            
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)

            # [Fixed Opponent] get ego reward
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
            # [Fixed Opponent] reset opponent mask/rnn_states
            # if self.use_selfplay:
            #     eval_opponent_obs = eval_obs[:, 2:, :15, ...]  # 蓝方飞机，只取前15维
            #     eval_obs = eval_obs[:, :2, ...]           # 红方飞机
            #     eval_opponent_masks[eval_dones_env == True] = \
            #         np.zeros(((eval_dones_env == True).sum(), *eval_opponent_masks.shape[1:]), dtype=np.float32)
            #     eval_opponent_rnn_states[eval_dones_env == True] = \
            #         np.zeros(((eval_dones_env == True).sum(), *eval_opponent_rnn_states.shape[1:]), dtype=np.float32)
            if self.use_selfplay:
                # 构建21维观察数据：前9维 + 6个0 + 后6维
                if eval_obs.shape[1] >= 3:
                    original_eval_obs = eval_obs[:, 2:, :15, ...]  # 蓝方飞机，取前15维
                    eval_opponent_obs = np.zeros((original_eval_obs.shape[0], original_eval_obs.shape[1], 21, *original_eval_obs.shape[3:]), dtype=original_eval_obs.dtype)
                    eval_opponent_obs[:, :, :9, ...] = original_eval_obs[:, :, :9, ...]  # 前9维
                    eval_opponent_obs[:, :, 9:15, ...] = 0  # 中间6个0
                    eval_opponent_obs[:, :, 15:21, ...] = original_eval_obs[:, :, 9:15, ...]  # 后6维
                else:
                    eval_opponent_obs = np.zeros((eval_obs.shape[0], 0, 21, *eval_obs.shape[3:]), dtype=eval_obs.dtype)
                
                eval_obs = eval_obs[:, :2, ...]           # 红方飞机
                eval_opponent_masks[eval_dones_env == True] = \
                    np.zeros(((eval_dones_env == True).sum(), *eval_opponent_masks.shape[1:]), dtype=np.float32)
                eval_opponent_rnn_states[eval_dones_env == True] = \
                    np.zeros(((eval_dones_env == True).sum(), *eval_opponent_rnn_states.shape[1:]), dtype=np.float32)
        eval_infos = {}
        eval_infos['eval_average_episode_rewards'] = np.concatenate(eval_episode_rewards).mean() 
        logging.info(" eval average episode rewards: " + str(eval_infos['eval_average_episode_rewards']))
        self.log_info(eval_infos, total_num_steps)

        logging.info("...End evaluation")

    @torch.no_grad()
    def render(self):
        logging.info("\nStart render ...")
        render_episode_rewards = 0
        render_obs, render_share_obs = self.envs.reset()
        render_masks = np.ones((1, *self.buffer.masks.shape[2:]), dtype=np.float32)
        render_rnn_states = np.zeros((1, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
        self.envs.render(mode='txt', filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi')
        
        # if self.use_selfplay:
        #     # reset obs/rnn/mask
        #     render_obs, render_share_obs = self.envs.reset()
        #     render_masks = np.ones_like(render_masks, dtype=np.float32)
        #     render_rnn_states = np.zeros_like(render_rnn_states, dtype=np.float32)
        #     render_opponent_obs = render_obs[:, 2:, :15, ...]  # 蓝方飞机，只取前15维
        #     render_obs = render_obs[:, :2, ...]           # 红方飞机
        #     render_opponent_masks = np.ones_like(render_masks, dtype=np.float32)
        #     render_opponent_rnn_states = np.zeros_like(render_rnn_states, dtype=np.float32)
        if self.use_selfplay:
            # reset obs/rnn/mask
            render_obs, render_share_obs = self.envs.reset()
            render_masks = np.ones_like(render_masks, dtype=np.float32)
            render_rnn_states = np.zeros_like(render_rnn_states, dtype=np.float32)
            
            # 构建21维观察数据：前9维 + 6个0 + 后6维
            if render_obs.shape[1] >= 3:
                original_render_obs = render_obs[:, 2:, :15, ...]  # 蓝方飞机，取前15维
                render_opponent_obs = np.zeros((original_render_obs.shape[0], original_render_obs.shape[1], 21, *original_render_obs.shape[3:]), dtype=original_render_obs.dtype)
                render_opponent_obs[:, :, :9, ...] = original_render_obs[:, :, :9, ...]  # 前9维
                render_opponent_obs[:, :, 9:15, ...] = 0  # 中间6个0
                render_opponent_obs[:, :, 15:21, ...] = original_render_obs[:, :, 9:15, ...]  # 后6维
            else:
                render_opponent_obs = np.zeros((render_obs.shape[0], 0, 21, *render_obs.shape[3:]), dtype=render_obs.dtype)
            
            render_obs = render_obs[:, :2, ...]           # 红方飞机
            render_opponent_masks = np.ones_like(render_masks, dtype=np.float32)
            render_opponent_rnn_states = np.zeros_like(render_rnn_states, dtype=np.float32)            
        while True:
            self.policy.prep_rollout()
            render_actions, render_rnn_states = self.policy.act(np.concatenate(render_obs),
                                                                np.concatenate(render_rnn_states),
                                                                np.concatenate(render_masks),
                                                                deterministic=True)
            render_actions = np.expand_dims(_t2n(render_actions), axis=0)
            render_rnn_states = np.expand_dims(_t2n(render_rnn_states), axis=0)
            
            # [Fixed Opponent] get actions of fixed opponent policy
            if self.use_selfplay:
                render_opponent_actions, render_opponent_rnn_states \
                    = self.eval_opponent_policy.act(np.concatenate(render_opponent_obs),
                                                    np.concatenate(render_opponent_rnn_states),
                                                    np.concatenate(render_opponent_masks),
                                                    deterministic=True)
                render_opponent_actions = np.expand_dims(_t2n(render_opponent_actions), axis=0)
                render_opponent_rnn_states = np.expand_dims(_t2n(render_opponent_rnn_states), axis=0)
                render_actions = np.concatenate((render_actions, render_opponent_actions), axis=1)
            
            # Obser reward and next obs
            render_obs, render_share_obs, render_rewards, render_dones, render_infos = self.envs.step(render_actions)
            if self.use_selfplay:
                render_rewards = render_rewards[:, :2, ...]  # 只保留红方奖励
            render_episode_rewards += render_rewards
            self.envs.render(mode='txt', filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi')
            if render_dones.all():
                break
            # if self.use_selfplay:
            #     render_opponent_obs = render_obs[:, 2:, :15, ...]  # 蓝方飞机，只取前15维
            #     render_obs = render_obs[:, :2, ...]           # 红方飞机
            if self.use_selfplay:
                # 构建21维观察数据：前9维 + 6个0 + 后6维
                if render_obs.shape[1] >= 3:
                    original_render_obs = render_obs[:, 2:, :15, ...]  # 蓝方飞机，取前15维
                    render_opponent_obs = np.zeros((original_render_obs.shape[0], original_render_obs.shape[1], 21, *original_render_obs.shape[3:]), dtype=original_render_obs.dtype)
                    render_opponent_obs[:, :, :9, ...] = original_render_obs[:, :, :9, ...]  # 前9维
                    render_opponent_obs[:, :, 9:15, ...] = 0  # 中间6个0
                    render_opponent_obs[:, :, 15:21, ...] = original_render_obs[:, :, 9:15, ...]  # 后6维
                else:
                    render_opponent_obs = np.zeros((render_obs.shape[0], 0, 21, *render_obs.shape[3:]), dtype=render_obs.dtype)
                
                render_obs = render_obs[:, :2, ...]           # 红方飞机           

        render_infos = {}
        render_infos['render_episode_reward'] = render_episode_rewards
        logging.info("render episode reward of agent: " + str(render_infos['render_episode_reward']))

    def save(self, episode):
        policy_actor_state_dict = self.policy.actor.state_dict()
        torch.save(policy_actor_state_dict, str(self.save_dir) + '/actor_latest.pt')
        policy_critic_state_dict = self.policy.critic.state_dict()
        torch.save(policy_critic_state_dict, str(self.save_dir) + '/critic_latest.pt')
        # [Fixed Opponent] save policy only for red aircraft
        if self.use_selfplay:
            torch.save(policy_actor_state_dict, str(self.save_dir) + f'/actor_{episode}.pt')

