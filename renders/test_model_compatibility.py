#!/usr/bin/env python3
"""
模型兼容性测试脚本
查找和测试不同模型的兼容性
"""

import os
import sys
import torch
import numpy as np
import logging
from pathlib import Path
import argparse

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
        self.hidden_size = 128

def find_all_models(base_dir):
    """查找所有模型文件"""
    models = []
    
    if not os.path.exists(base_dir):
        return models
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.pt') and file.startswith('actor'):
                model_path = os.path.join(root, file)
                models.append(model_path)
    
    return models

def analyze_model_structure(model_path, device):
    """分析模型结构"""
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        
        # 分析动作输出层
        action_out_keys = [key for key in state_dict.keys() if 'action_outs' in key and 'weight' in key]
        
        # 分析观察输入层
        obs_keys = [key for key in state_dict.keys() if 'obs_encoder' in key or 'base' in key]
        
        # 分析RNN层
        rnn_keys = [key for key in state_dict.keys() if 'rnn' in key]
        
        return {
            'action_outs': action_out_keys,
            'obs_encoder': obs_keys,
            'rnn_layers': rnn_keys,
            'total_params': len(state_dict.keys())
        }
    except Exception as e:
        logger.error(f"分析模型结构失败: {e}")
        return None

def test_model_compatibility(model_path, env, device):
    """测试模型兼容性"""
    try:
        # 创建测试策略网络
        test_policy = PPOActor(Args(), env.observation_space, env.action_space, device=device)
        
        # 尝试加载模型
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        
        # 检查动作空间是否匹配
        expected_action_dims = env.action_space.nvec
        logger.info(f"环境期望的动作空间: {expected_action_dims}")
        
        # 检查模型中的动作输出层
        action_out_keys = [key for key in state_dict.keys() if 'action_outs' in key and 'weight' in key]
        logger.info(f"模型中的动作输出层: {action_out_keys}")
        
        # 尝试加载模型
        test_policy.load_state_dict(state_dict, strict=False)
        logger.info("模型兼容性测试通过")
        return True
        
    except Exception as e:
        logger.error(f"模型兼容性测试失败: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='测试模型兼容性')
    parser.add_argument('--search-dir', default='scripts/results', help='搜索目录')
    parser.add_argument('--test-env', action='store_true', help='是否测试环境兼容性')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 查找所有模型
    logger.info(f"在 {args.search_dir} 中查找模型文件...")
    models = find_all_models(args.search_dir)
    
    if not models:
        logger.warning(f"在 {args.search_dir} 中没有找到模型文件")
        return
    
    logger.info(f"找到 {len(models)} 个模型文件:")
    for i, model_path in enumerate(models):
        logger.info(f"  {i+1}. {model_path}")
    
    # 分析每个模型的结构
    logger.info("\n分析模型结构...")
    compatible_models = []
    
    for model_path in models:
        logger.info(f"\n分析模型: {model_path}")
        
        # 分析模型结构
        structure = analyze_model_structure(model_path, device)
        if structure:
            logger.info(f"  动作输出层: {structure['action_outs']}")
            logger.info(f"  观察编码层: {structure['obs_encoder']}")
            logger.info(f"  RNN层: {structure['rnn_layers']}")
            logger.info(f"  总参数数量: {structure['total_params']}")
            
            # 判断模型类型
            if len(structure['action_outs']) == 4:
                logger.info("  ✓ 标准PPO模型 (4个动作输出)")
                compatible_models.append((model_path, "Standard PPO"))
            elif len(structure['action_outs']) == 3:
                logger.info("  ⚠ 分层模型 (3个动作输出)")
            elif len(structure['action_outs']) == 5:
                logger.info("  ⚠ 分层射击模型 (5个动作输出)")
            else:
                logger.info(f"  ❓ 未知模型类型 ({len(structure['action_outs'])}个动作输出)")
    
    # 如果指定了测试环境，创建环境并测试兼容性
    if args.test_env:
        logger.info("\n创建4v4环境进行兼容性测试...")
        try:
            env = MultipleCombatEnv("4v4/NoWeapon/1v1Style")
            env.seed(0)
            logger.info(f"环境创建成功. 观察空间: {env.observation_space}")
            logger.info(f"动作空间: {env.action_space}")
            
            # 测试兼容的模型
            logger.info("\n测试模型兼容性...")
            for model_path, model_type in compatible_models:
                logger.info(f"\n测试 {model_type} 模型: {model_path}")
                if test_model_compatibility(model_path, env, device):
                    logger.info(f"  ✅ {model_path} 兼容")
                else:
                    logger.info(f"  ❌ {model_path} 不兼容")
                    
        except Exception as e:
            logger.error(f"环境创建失败: {e}")
    
    # 输出总结
    logger.info("\n" + "="*60)
    logger.info("模型兼容性分析总结")
    logger.info("="*60)
    
    if compatible_models:
        logger.info(f"找到 {len(compatible_models)} 个兼容的模型:")
        for model_path, model_type in compatible_models:
            logger.info(f"  - {model_path} ({model_type})")
    else:
        logger.warning("没有找到兼容的模型")
    
    logger.info("\n建议:")
    logger.info("1. 使用标准PPO模型进行4v4渲染")
    logger.info("2. 避免使用分层模型，因为动作空间不匹配")
    logger.info("3. 如果只有分层模型，可以使用基线策略进行渲染")

if __name__ == "__main__":
    main() 