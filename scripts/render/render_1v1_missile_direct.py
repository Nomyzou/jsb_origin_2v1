#!/usr/bin/env python3
"""
直接修改任务类的导弹发射脚本
通过修改任务类的step方法来强制发射导弹
"""

import os
import sys
import torch
import random
import logging
import numpy as np
from pathlib import Path
import setproctitle
from datetime import datetime

# Deal with import error
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from config import get_config
from runner.share_jsbsim_runner import ShareJSBSimRunner
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import DummyVecEnv, ShareDummyVecEnv
from envs.JSBSim.core.simulatior import MissileSimulator


class ForcedShootTask:
    """强制发射导弹的任务包装器"""
    
    def __init__(self, original_task):
        self.original_task = original_task
        self.shoot_step = 10  # 在第10步发射导弹
        
    def __getattr__(self, name):
        return getattr(self.original_task, name)
    
    def step(self, env):
        # 调用原始任务的step方法
        self.original_task.step(env)
        
        # 在指定步数强制发射导弹
        if env.current_step == self.shoot_step:
            logging.info(f"强制发射导弹，当前步数: {env.current_step}")
            for agent_id, agent in env.agents.items():
                if agent.is_alive and hasattr(self.original_task, 'remaining_missiles') and self.original_task.remaining_missiles[agent_id] > 0:
                    target = agent.enemies[0]
                    new_missile_uid = f"M{agent_id[1:]}1"
                    missile = MissileSimulator.create(
                        parent=agent,
                        target=target,
                        uid=new_missile_uid,
                        missile_model="AIM-9L"
                    )
                    env.add_temp_simulator(missile)
                    self.original_task.remaining_missiles[agent_id] -= 1
                    logging.info(f"智能体 {agent_id} 发射导弹 {new_missile_uid}")


def make_render_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SingleCombat":
                env = SingleCombatEnv(all_args.scenario_name)
                # 包装任务类，强制发射导弹
                env.task = ForcedShootTask(env.task)
            elif all_args.env_name == "SingleControl":
                env = SingleControlEnv(all_args.scenario_name)
            elif all_args.env_name == "MultipleCombat":
                env = MultipleCombatEnv(all_args.scenario_name)
            else:
                logging.error("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.env_name == "MultipleCombat":
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return DummyVecEnv([get_env_fn(0)])


def parse_args(args, parser):
    group = parser.add_argument_group("JSBSim Env parameters")
    group.add_argument('--episode-length', type=int, default=1000,
                       help="the max length of an episode")
    group.add_argument('--scenario-name', type=str, default='1v1/ShootMissile/HierarchySelfplay',
                       help="scenario name for missile combat")
    group.add_argument('--num-agents', type=int, default=2,
                       help="number of agents in missile combat")
    group.add_argument('--model-path', type=str, 
                       default='scripts/results/SingleCombat/1v1/ShootMissile/HierarchySelfplay/ppo/v1/wandb/latest-run/files/actor_96.pt',
                       help="path to the trained model")
    group.add_argument('--render-episodes', type=int, default=1,
                       help="number of episodes to render")
    group.add_argument('--output-dir', type=str, default='renders/missile_combat_direct',
                       help="output directory for rendered files")
    group.add_argument('--shoot-step', type=int, default=10,
                       help="step to force missile launch")
    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    
    # 设置默认参数
    all_args.env_name = "SingleCombat"
    all_args.algorithm_name = "ppo"
    all_args.experiment_name = "v1"
    all_args.user_name = "missile_render"
    all_args.use_selfplay = True
    all_args.use_eval = True
    all_args.cuda = True
    all_args.n_training_threads = 1
    all_args.n_rollout_threads = 1
    all_args.seed = 1
    all_args.model_dir = str(Path(all_args.model_path).parent)
    all_args.use_prior = True
    
    # 检查模型文件是否存在
    if not os.path.exists(all_args.model_path):
        logging.error(f"Model file not found: {all_args.model_path}")
        return
    
    # seed
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        logging.info("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        logging.info("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # 创建输出目录
    output_dir = Path(all_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # run dir
    run_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/results") \
        / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    curr_run = f'render_{timestamp}'
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(all_args.env_name)
                              + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # env init
    envs = make_render_env(all_args)
    num_agents = all_args.num_agents
    
    config = {
        "all_args": all_args,
        "eval_envs": None,
        "envs": envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
        "model_path": all_args.model_path,
        "model_dir": str(Path(all_args.model_path).parent),
        "render_mode": "file",
        "render_episodes": all_args.render_episodes,
        "output_dir": output_dir
    }

    # run experiments
    if all_args.env_name == "MultipleCombat":
        runner = ShareJSBSimRunner(config)
    else:
        if all_args.use_selfplay:
            from runner.selfplay_jsbsim_runner import SelfplayJSBSimRunner as Runner
        else:
            from runner.jsbsim_runner import JSBSimRunner as Runner
        runner = Runner(config)
    
    # 渲染多个episode
    for episode in range(all_args.render_episodes):
        logging.info(f"Rendering episode {episode + 1}/{all_args.render_episodes}")
        episode_output_dir = output_dir / f"episode_{episode + 1}"
        episode_output_dir.mkdir(exist_ok=True)
        
        # 设置episode输出目录
        config["episode_output_dir"] = episode_output_dir
        runner.config = config
        
        # 渲染单个episode
        runner.render()
        
        # 将生成的ACMI文件移动到episode目录
        acmi_file = Path("scripts/results") / f"{all_args.experiment_name}.txt.acmi"
        if acmi_file.exists():
            episode_acmi_file = episode_output_dir / f"episode_{episode + 1}.acmi"
            acmi_file.rename(episode_acmi_file)
            logging.info(f"ACMI file saved to: {episode_acmi_file}")
            
            # 检查是否包含导弹
            with open(episode_acmi_file, 'r') as f:
                content = f.read()
            missile_lines = [line for line in content.split('\n') if line.startswith('M')]
            explosion_lines = [line for line in content.split('\n') if 'Explosion' in line]
            logging.info(f"Episode {episode + 1}: 导弹轨迹行数={len(missile_lines)}, 爆炸效果行数={len(explosion_lines)}")

    # post process
    envs.close()
    
    logging.info(f"Rendering completed. Output saved to: {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main(sys.argv[1:]) 