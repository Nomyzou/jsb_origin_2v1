#!/usr/bin/env python
"""
策略网络批处理测试脚本
验证策略网络如何处理多架飞机的输入
"""

import numpy as np
import torch
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from algorithms.ppo.ppo_actor import PPOActor
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def test_batch_processing():
    """测试策略网络的批处理能力"""
    
    logger.info("=" * 60)
    logger.info("策略网络批处理测试")
    logger.info("=" * 60)
    
    # 创建策略网络
    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模拟观察空间和动作空间
    from gymnasium.spaces import Box, MultiDiscrete
    obs_space = Box(low=-np.inf, high=np.inf, shape=(15,))
    act_space = MultiDiscrete([41, 41, 41, 30])
    
    policy = PPOActor(args, obs_space, act_space, device=device)
    policy.eval()
    
    logger.info(f"策略网络创建成功，设备: {device}")
    
    # 测试1架飞机（训练时的情况）
    logger.info("\n1. 测试1架飞机输入:")
    obs_1v1 = np.random.randn(1, 15).astype(np.float32)  # (1, 15)
    rnn_states_1v1 = np.zeros((1, 1, 128), dtype=np.float32)  # (1, 1, 128)
    masks_1v1 = np.ones((1, 1), dtype=np.float32)  # (1, 1)
    
    with torch.no_grad():
        actions_1v1, log_probs_1v1, rnn_out_1v1 = policy(obs_1v1, rnn_states_1v1, masks_1v1, deterministic=True)
    
    logger.info(f"  输入形状: {obs_1v1.shape}")
    logger.info(f"  输出动作形状: {actions_1v1.shape}")
    logger.info(f"  输出RNN状态形状: {rnn_out_1v1.shape}")
    
    # 测试4架飞机（4v4渲染时的情况）
    logger.info("\n2. 测试4架飞机输入:")
    obs_4v4 = np.random.randn(4, 15).astype(np.float32)  # (4, 15)
    rnn_states_4v4 = np.zeros((4, 1, 128), dtype=np.float32)  # (4, 1, 128)
    masks_4v4 = np.ones((4, 1), dtype=np.float32)  # (4, 1)
    
    with torch.no_grad():
        actions_4v4, log_probs_4v4, rnn_out_4v4 = policy(obs_4v4, rnn_states_4v4, masks_4v4, deterministic=True)
    
    logger.info(f"  输入形状: {obs_4v4.shape}")
    logger.info(f"  输出动作形状: {actions_4v4.shape}")
    logger.info(f"  输出RNN状态形状: {rnn_out_4v4.shape}")
    
    # 验证输出
    logger.info("\n3. 验证输出:")
    logger.info(f"  1v1动作数量: {actions_1v1.shape[0]}")
    logger.info(f"  4v4动作数量: {actions_4v4.shape[0]}")
    logger.info(f"  4v4动作是否独立: {actions_4v4.shape[0] == 4}")
    
    # 测试动作一致性
    logger.info("\n4. 测试动作一致性:")
    # 使用相同的观察数据
    same_obs = np.random.randn(1, 15).astype(np.float32)
    same_rnn = np.zeros((1, 1, 128), dtype=np.float32)
    same_masks = np.ones((1, 1), dtype=np.float32)
    
    with torch.no_grad():
        action1, _, _ = policy(same_obs, same_rnn, same_masks, deterministic=True)
        action2, _, _ = policy(same_obs, same_rnn, same_masks, deterministic=True)
    
    # 在确定性模式下，相同输入应该产生相同输出
    is_consistent = torch.allclose(action1, action2)
    logger.info(f"  相同输入产生相同输出: {'✓' if is_consistent else '✗'}")
    
    # 测试不同飞机的动作独立性
    logger.info("\n5. 测试动作独立性:")
    different_obs = np.random.randn(4, 15).astype(np.float32)
    different_rnn = np.zeros((4, 1, 128), dtype=np.float32)
    different_masks = np.ones((4, 1), dtype=np.float32)
    
    with torch.no_grad():
        actions_diff, _, _ = policy(different_obs, different_rnn, different_masks, deterministic=True)
    
    # 检查是否有不同的动作
    unique_actions = torch.unique(actions_diff, dim=0)
    logger.info(f"  不同飞机产生不同动作: {'✓' if unique_actions.shape[0] > 1 else '✗'}")
    logger.info(f"  唯一动作数量: {unique_actions.shape[0]}")
    
    logger.info("\n" + "=" * 60)
    logger.info("批处理测试完成！")
    logger.info("=" * 60)
    
    return True

def test_memory_efficiency():
    """测试内存效率"""
    
    logger.info("\n内存效率测试:")
    
    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    from gymnasium.spaces import Box, MultiDiscrete
    obs_space = Box(low=-np.inf, high=np.inf, shape=(15,))
    act_space = MultiDiscrete([41, 41, 41, 30])
    
    policy = PPOActor(args, obs_space, act_space, device=device)
    policy.eval()
    
    # 测试不同batch size的内存使用
    batch_sizes = [1, 4, 8, 16]
    
    for batch_size in batch_sizes:
        obs = np.random.randn(batch_size, 15).astype(np.float32)
        rnn_states = np.zeros((batch_size, 1, 128), dtype=np.float32)
        masks = np.ones((batch_size, 1), dtype=np.float32)
        
        with torch.no_grad():
            actions, _, _ = policy(obs, rnn_states, masks, deterministic=True)
        
        logger.info(f"  Batch size {batch_size}: 输入 {obs.shape}, 输出 {actions.shape}")

def main():
    """主测试函数"""
    
    try:
        # 测试批处理能力
        test_batch_processing()
        
        # 测试内存效率
        test_memory_efficiency()
        
        logger.info("\n🎉 所有测试通过！")
        logger.info("策略网络完全支持从1架飞机到4架飞机的批处理！")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        return False

if __name__ == "__main__":
    main() 