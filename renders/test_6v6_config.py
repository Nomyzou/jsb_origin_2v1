import numpy as np
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 检查当前工作目录和Python路径
print(f"Current working directory: {os.getcwd()}")
print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")  # 只显示前3个路径

try:
    from envs.JSBSim.envs import MultipleCombatEnv
    print("Successfully imported MultipleCombatEnv")
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative import...")
    try:
        sys.path.insert(0, str(project_root / "envs"))
        from JSBSim.envs import MultipleCombatEnv
        print("Successfully imported with alternative path")
    except ImportError as e2:
        print(f"Alternative import also failed: {e2}")
        sys.exit(1)

import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_6v6_config():
    """测试6v6配置文件是否正确"""
    try:
        logger.info("Testing 6v6 configuration...")
        
        # 创建6v6环境
        env = MultipleCombatEnv("6v6/NoWeapon/Hierarchy1v1Style")
        env.seed(0)
        
        logger.info(f"Environment created successfully!")
        logger.info(f"Number of agents: {env.num_agents}")
        logger.info(f"Expected: 12 (6 red + 6 blue)")
        
        if env.num_agents != 12:
            logger.error(f"Wrong number of agents! Expected 12, got {env.num_agents}")
            return False
        
        # 检查飞机ID
        agent_ids = list(env.agents.keys())
        logger.info(f"Agent IDs: {agent_ids}")
        
        # 验证红方飞机
        red_agents = [aid for aid in agent_ids if aid.startswith('A')]
        logger.info(f"Red agents: {red_agents}")
        expected_red = ['A0100', 'A0200', 'A0300', 'A0400', 'A0500', 'A0600']
        
        if set(red_agents) != set(expected_red):
            logger.error(f"Red agents mismatch! Expected {expected_red}, got {red_agents}")
            return False
        
        # 验证蓝方飞机
        blue_agents = [aid for aid in agent_ids if aid.startswith('B')]
        logger.info(f"Blue agents: {blue_agents}")
        expected_blue = ['B0100', 'B0200', 'B0300', 'B0400', 'B0500', 'B0600']
        
        if set(blue_agents) != set(expected_blue):
            logger.error(f"Blue agents mismatch! Expected {expected_blue}, got {blue_agents}")
            return False
        
        # 检查观察空间
        logger.info(f"Observation space: {env.observation_space}")
        logger.info(f"Action space: {env.action_space}")
        
        # 重置环境
        obs, _ = env.reset()
        logger.info(f"Observation shape: {obs.shape}")
        logger.info(f"Expected shape: (12, 15)")
        
        if obs.shape != (12, 15):
            logger.error(f"Observation shape mismatch! Expected (12, 15), got {obs.shape}")
            return False
        
        # 检查每架飞机的初始状态
        logger.info("Checking initial aircraft states...")
        for agent_id in agent_ids:
            agent = env.agents[agent_id]
            logger.info(f"{agent_id}: alive={agent.is_alive}, bloods={agent.bloods}")
        
        logger.info("6v6 configuration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing 6v6 configuration: {e}")
        return False

if __name__ == "__main__":
    success = test_6v6_config()
    if success:
        print("✅ 6v6 configuration test passed!")
    else:
        print("❌ 6v6 configuration test failed!") 