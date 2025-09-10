import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from envs.JSBSim.envs import MultipleCombatEnv
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_hierarchy_1v1_style_env():
    """测试4v4分层1v1风格环境"""
    logger.info("Testing 4v4 Hierarchical 1v1 Style Environment...")
    
    try:
        # 创建环境
        env = MultipleCombatEnv("4v4/NoWeapon/Hierarchy1v1Style")
        env.seed(0)
        
        logger.info(f"Environment created successfully!")
        logger.info(f"Number of agents: {env.num_agents}")
        logger.info(f"Observation space: {env.observation_space}")
        logger.info(f"Action space: {env.action_space}")
        
        # 验证观察空间
        expected_obs_length = 15
        actual_obs_length = env.observation_space.shape[0]
        logger.info(f"Expected observation length: {expected_obs_length}")
        logger.info(f"Actual observation length: {actual_obs_length}")
        
        if actual_obs_length != expected_obs_length:
            logger.error(f"Observation space mismatch! Expected {expected_obs_length}, got {actual_obs_length}")
            return False
        
        # 验证动作空间
        expected_action_dims = [3, 5, 3]  # 分层动作空间
        actual_action_dims = env.action_space.nvec
        logger.info(f"Expected action dimensions: {expected_action_dims}")
        logger.info(f"Actual action dimensions: {actual_action_dims}")
        
        if not np.array_equal(actual_action_dims, expected_action_dims):
            logger.error(f"Action space mismatch! Expected {expected_action_dims}, got {actual_action_dims}")
            return False
        
        # 重置环境
        obs, _ = env.reset()
        logger.info(f"Environment reset successful")
        logger.info(f"Observation shape: {obs.shape}")
        
        # 验证观察数据
        logger.info("Testing observation data...")
        for i, agent_id in enumerate(env.agents.keys()):
            agent_obs = obs[i]
            logger.info(f"Agent {agent_id}: observation shape {agent_obs.shape}")
            
            # 检查观察数据是否在合理范围内
            if np.any(np.isnan(agent_obs)) or np.any(np.isinf(agent_obs)):
                logger.error(f"Invalid observation data for agent {agent_id}")
                return False
        
        # 测试动作执行
        logger.info("Testing action execution...")
        num_agents = env.num_agents
        
        # 生成随机动作（分层动作空间）
        actions = []
        for i in range(num_agents):
            # 为每个智能体生成分层动作
            agent_actions = []
            for dim in env.action_space.nvec:
                agent_actions.append(np.random.randint(0, dim))
            actions.append(agent_actions)
        
        actions = np.array(actions)
        logger.info(f"Generated actions shape: {actions.shape}")
        
        # 执行动作
        obs, _, rewards, dones, infos = env.step(actions)
        logger.info(f"Action execution successful")
        logger.info(f"Rewards shape: {rewards.shape}")
        logger.info(f"Dones shape: {dones.shape}")
        
        # 验证对应关系
        logger.info("Testing corresponding enemy relationships...")
        for agent_id in env.agents.keys():
            if agent_id.startswith('A'):
                corresponding_enemy = 'B' + agent_id[1:]
                logger.info(f"  {agent_id} -> {corresponding_enemy}")
            elif agent_id.startswith('B'):
                corresponding_enemy = 'A' + agent_id[1:]
                logger.info(f"  {agent_id} -> {corresponding_enemy}")
        
        logger.info("All tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        return False

def main():
    success = test_hierarchy_1v1_style_env()
    if success:
        logger.info("✅ 4v4 Hierarchical 1v1 Style Environment test PASSED")
    else:
        logger.error("❌ 4v4 Hierarchical 1v1 Style Environment test FAILED")

if __name__ == "__main__":
    main() 