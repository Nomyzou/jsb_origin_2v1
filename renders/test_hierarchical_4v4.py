#!/usr/bin/env python3
"""
测试4v4分层环境配置
验证分层模型和分层环境的兼容性
"""

import os
import sys
import torch
import numpy as np
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from algorithms.ppo.ppo_actor import PPOActor
from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Args:
    """策略网络参数类"""
    def __init__(self):
        self.use_prior = True
        self.use_centralized_V = True
        self.use_naive_recurrent_policy = True
        self.use_recurrent_policy = True
        self.recurrent_N = 1
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.gain = 0.01

def test_hierarchical_environment():
    """测试分层环境"""
    logger.info("=" * 60)
    logger.info("测试4v4分层环境配置")
    logger.info("=" * 60)
    
    # 测试不同的环境配置
    configs = [
        ("4v4/NoWeapon/Selfplay", "分层多机任务"),
        ("4v4/NoWeapon/1v1Style", "1v1风格任务")
    ]
    
    for config_name, description in configs:
        logger.info(f"\n测试配置: {config_name} ({description})")
        
        try:
            # 创建环境
            env = MultipleCombatEnv(config_name)
            env.seed(0)
            
            logger.info(f"  ✓ 环境创建成功")
            logger.info(f"    飞机数量: {env.num_agents}")
            logger.info(f"    观察空间: {env.observation_space}")
            logger.info(f"    动作空间: {env.action_space}")
            logger.info(f"    任务类型: {env.task.__class__.__name__}")
            
            # 检查是否为分层任务
            if "Hierarchical" in env.task.__class__.__name__:
                logger.info(f"    ✓ 确认使用分层任务")
            else:
                logger.info(f"    ⚠ 未使用分层任务")
            
            # 重置环境
            obs, _ = env.reset()
            logger.info(f"    ✓ 环境重置成功，观察形状: {obs.shape}")
            
            # 测试几步
            for step in range(3):
                # 随机动作
                actions = np.array([env.action_space.sample() for _ in range(env.num_agents)])
                
                # 环境步进
                obs, _, rewards, dones, infos = env.step(actions)
                
                logger.info(f"    步骤 {step+1}: 观察形状={obs.shape}, 奖励形状={rewards.shape}")
                
                if dones.all():
                    logger.info("    环境提前结束")
                    break
            
            logger.info(f"  ✓ {config_name} 配置测试通过")
            
        except Exception as e:
            logger.error(f"  ❌ {config_name} 配置测试失败: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("分层环境测试总结")
    logger.info("=" * 60)
    logger.info("1. Selfplay配置使用hierarchical_multiplecombat任务")
    logger.info("2. 1v1Style配置使用multiplecombat_1v1_style任务")
    logger.info("3. 分层模型需要与分层任务匹配")
    logger.info("4. 建议使用Selfplay配置进行分层模型渲染")

def test_hierarchical_model_compatibility():
    """测试分层模型兼容性"""
    logger.info("\n" + "=" * 60)
    logger.info("测试分层模型兼容性")
    logger.info("=" * 60)
    
    # 分层模型路径
    model_path = "scripts/results/SingleCombat/1v1/NoWeapon/HierarchySelfplay/ppo/v1/wandb/latest-run/files/actor_latest.pt"
    
    if not os.path.exists(model_path):
        logger.warning(f"模型文件不存在: {model_path}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # 创建分层环境
        env = MultipleCombatEnv("4v4/NoWeapon/Selfplay")
        env.seed(0)
        
        logger.info(f"环境配置:")
        logger.info(f"  观察空间: {env.observation_space}")
        logger.info(f"  动作空间: {env.action_space}")
        logger.info(f"  任务类型: {env.task.__class__.__name__}")
        
        # 创建策略网络
        policy = PPOActor(Args(), env.observation_space, env.action_space, device=device)
        
        # 加载模型
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        
        # 分析模型结构
        action_out_keys = [key for key in state_dict.keys() if 'action_outs' in key and 'weight' in key]
        logger.info(f"模型结构:")
        logger.info(f"  动作输出层: {action_out_keys}")
        logger.info(f"  动作输出数量: {len(action_out_keys)}")
        
        # 判断模型类型
        if len(action_out_keys) == 3:
            logger.info("  ✓ 检测到分层模型 (3个动作输出)")
            logger.info("    动作0: 高度控制 [3]")
            logger.info("    动作1: 航向控制 [5]")
            logger.info("    动作2: 速度控制 [3]")
        elif len(action_out_keys) == 4:
            logger.info("  ⚠ 检测到标准PPO模型 (4个动作输出)")
            logger.info("    动作0: 副翼 [41]")
            logger.info("    动作1: 升降舵 [41]")
            logger.info("    动作2: 方向舵 [41]")
            logger.info("    动作3: 油门 [30]")
        else:
            logger.info(f"  ❓ 未知模型类型 ({len(action_out_keys)}个动作输出)")
        
        # 尝试加载模型
        policy.load_state_dict(state_dict, strict=False)
        logger.info("  ✓ 模型加载成功")
        
        # 测试推理
        obs, _ = env.reset()
        test_obs = obs[:4, :]  # 取前4架飞机的观察
        rnn_states = np.zeros((4, 1, 128), dtype=np.float32)
        masks = np.ones((4, 1))
        
        actions, _, _ = policy(test_obs, rnn_states, masks, deterministic=True)
        logger.info(f"  ✓ 推理测试成功，动作形状: {actions.shape}")
        
        logger.info("✓ 分层模型兼容性测试通过")
        
    except Exception as e:
        logger.error(f"❌ 分层模型兼容性测试失败: {e}")

def main():
    """主函数"""
    logger.info("开始4v4分层环境配置测试...")
    
    # 测试分层环境
    test_hierarchical_environment()
    
    # 测试分层模型兼容性
    test_hierarchical_model_compatibility()
    
    logger.info("\n" + "=" * 60)
    logger.info("测试完成！")
    logger.info("=" * 60)
    logger.info("建议:")
    logger.info("1. 使用 '4v4/NoWeapon/Selfplay' 配置进行分层模型渲染")
    logger.info("2. 确保模型是分层模型 (3个动作输出)")
    logger.info("3. 分层模型会自动转换为低层动作")

if __name__ == "__main__":
    main() 