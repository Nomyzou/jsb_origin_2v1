import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.JSBSim.envs import MultipleCombatEnv
from algorithms.ppo.ppo_actor import PPOActor
import time
from datetime import datetime
import logging
logging.basicConfig(level=logging.DEBUG)

# Helper: robust weight loading with strict->partial fallback and first-layer adaptation
def load_actor_weights_with_adaptation(policy: torch.nn.Module, weights_path: str, logger_prefix: str) -> None:
    try:
        try:
            loaded_sd = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True)
        except TypeError:
            loaded_sd = torch.load(weights_path, map_location=torch.device('cpu'))
        try:
            policy.load_state_dict(loaded_sd)
            logging.info(f"{logger_prefix} weights strictly loaded from: {weights_path}")
            return
        except Exception as e_strict:
            current_sd = policy.state_dict()
            adapted_sd = {}
            # keep all matching-shaped params
            for key, value in loaded_sd.items():
                if key in current_sd and current_sd[key].shape == value.shape:
                    adapted_sd[key] = value
            # try to adapt first linear layer width if mismatched
            first_w_keys = [
                'base.mlp.fc.0.weight',
            ]
            first_b_keys = [
                'base.mlp.fc.0.bias',
            ]
            for key in list(loaded_sd.keys()):
                # weight adaptation (column slice/pad)
                if any(k in key for k in first_w_keys) and key in current_sd:
                    src = loaded_sd[key]
                    dst = current_sd[key]
                    if src.dim() == 2 and dst.dim() == 2 and src.shape[0] == dst.shape[0] and src.shape[1] != dst.shape[1]:
                        if src.shape[1] >= dst.shape[1]:
                            adapted = src[:, :dst.shape[1]]
                        else:
                            pad_cols = dst.shape[1] - src.shape[1]
                            adapted = torch.cat([src, torch.zeros(src.shape[0], pad_cols, dtype=src.dtype)], dim=1)
                        adapted_sd[key] = adapted
                # bias adaptation (row slice/pad)
                if any(k in key for k in first_b_keys) and key in current_sd:
                    src = loaded_sd[key]
                    dst = current_sd[key]
                    if src.dim() == 1 and dst.dim() == 1 and src.shape[0] != dst.shape[0]:
                        if src.shape[0] >= dst.shape[0]:
                            adapted = src[:dst.shape[0]]
                        else:
                            pad_rows = dst.shape[0] - src.shape[0]
                            adapted = torch.cat([src, torch.zeros(pad_rows, dtype=src.dtype)], dim=0)
                        adapted_sd[key] = adapted
            missing_keys = [k for k in current_sd.keys() if k not in adapted_sd]
            unexpected_keys = [k for k in loaded_sd.keys() if k not in current_sd]
            policy.load_state_dict(adapted_sd, strict=False)
            logging.warning(
                f"{logger_prefix} weights partially loaded from: {weights_path}. "
                f"matched(adapted)={len(adapted_sd)} missing={len(missing_keys)} unexpected={len(unexpected_keys)}; "
                f"strict load error: {e_strict}"
            )
    except Exception as e:
        logging.warning(f"Failed to load {logger_prefix.lower()} weights; using randomly initialized parameters. Error: {e}")

class Args:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = True

def _t2n(x):
    return x.detach().cpu().numpy()

# --- 关键：明确 2v1 ---
num_ego = 2
num_enm = 1
num_agents = num_ego + num_enm

render = True
ego_policy_index = 1040
enm_policy_index = 0
episode_rewards = 0

# 建议切换到 2v1 的权重目录
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ego_run_dir = os.path.join(root_dir, "scripts/results/MultipleCombat/2v2/NoWeapon/HierarchySelfplay/mappo/v1/wandb/latest-run/files")
enm_run_dir = os.path.join(root_dir, "scripts/results/SingleCombat/1v1/NoWeapon/HierarchySelfplay/ppo/v1/wandb/latest-run/files")
experiment_name_r = ego_run_dir.split('/')[-4]
date_name = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f"{experiment_name_r}_{date_name}"
# --- 关键：切换环境到 2v1 ---
env = MultipleCombatEnv("2v1/NoWeapon/HierarchySelfplay")
env.seed(0)
args = Args()

ego_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("mps"))
enm_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("mps"))
ego_policy.eval()
enm_policy.eval()
# Replace manual loading with robust helper
load_actor_weights_with_adaptation(ego_policy, ego_run_dir + f"/actor_latest.pt", logger_prefix="Ego")
load_actor_weights_with_adaptation(enm_policy, enm_run_dir + f"/actor_latest.pt", logger_prefix="Enemy")

print("Start render")
obs, _ = env.reset()
if render:
    env.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')

# --- 关键：按人数创建 RNN states 和 masks ---
ego_rnn_states = np.zeros((num_ego, 1, 128), dtype=np.float32)
enm_rnn_states = np.zeros((num_enm, 1, 128), dtype=np.float32)
ego_masks = np.ones((num_ego, 1), dtype=np.float32)
enm_masks = np.ones((num_enm, 1), dtype=np.float32)

# --- 关键：按 [前2个为我方，最后1个为敌方] 的顺序切片 ---
ego_obs = obs[:num_ego, ...]
enm_obs = obs[num_ego:num_ego + num_enm, ...]

while True:
    start = time.time()
    ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, ego_masks, deterministic=True)
    enm_actions, _, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states, enm_masks, deterministic=True)
    ego_actions = _t2n(ego_actions)
    enm_actions = _t2n(enm_actions)
    ego_rnn_states = _t2n(ego_rnn_states)
    enm_rnn_states = _t2n(enm_rnn_states)

    # --- 关键：按相同顺序拼接 ---
    actions = np.concatenate((ego_actions, enm_actions), axis=0)

    # Env step
    obs, _, rewards, dones, infos = env.step(actions)

    # 只统计我方奖励
    rewards = rewards[:num_ego, ...]
    episode_rewards += rewards

    if render:
        env.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')

    if dones.all():
        print(infos)
        break

    # 可选打印
    # bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
    # print(f"step:{env.current_step}, bloods:{bloods}")

    # --- 关键：继续按人数切片 ---
    ego_obs = obs[:num_ego, ...]
    enm_obs = obs[num_ego:num_ego + num_enm, ...]