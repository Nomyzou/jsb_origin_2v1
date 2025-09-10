import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from envs.JSBSim.envs import MultipleCombatEnv
from envs.JSBSim.utils.situation_assessment import get_situation_adv
from envs.JSBSim.core.catalog import Catalog as c
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_target_selection():
    """测试目标选择功能"""
    logger.info("Testing target selection functionality...")
    
    try:
        # 创建环境
        env = MultipleCombatEnv("4v4/NoWeapon/Hierarchy1v1Style")
        env.seed(0)
        
        # 重置环境
        obs, _ = env.reset()
        logger.info(f"Environment reset successful")
        
        # 测试态势优势计算
        logger.info("Testing situation advantage calculation...")
        
        # 获取我方飞机和敌方飞机
        friendly_ids = [agent_id for agent_id in env.agents.keys() if agent_id.startswith('A')]
        enemy_ids = [agent_id for agent_id in env.agents.keys() if agent_id.startswith('B')]
        
        logger.info(f"Friendly agents: {friendly_ids}")
        logger.info(f"Enemy agents: {enemy_ids}")
        
        # 计算每架我方飞机对每架敌方飞机的态势优势
        for friendly_id in friendly_ids:
            logger.info(f"\nCalculating advantages for {friendly_id}:")
            ego_sim = env.agents[friendly_id]
            
            advantages = {}
            for enemy_id in enemy_ids:
                try:
                    enemy_sim = env.agents[enemy_id]
                    
                    # 计算态势优势
                    advantage = get_situation_adv(
                        ego_sim, enemy_sim, 
                        [c.position_long_gc_deg, c.position_lat_geod_deg, c.position_h_sl_ft,
                         c.attitude_roll_rad, c.attitude_pitch_rad, c.attitude_psi_rad,
                         c.velocities_u_fps, c.velocities_v_fps, c.velocities_w_fps,
                         c.velocities_u_fps, c.velocities_v_fps, c.velocities_w_fps],  # 重复速度分量以满足12个变量的要求
                        0.0, 0.0, 0.0
                    )
                    
                    advantages[enemy_id] = advantage
                    logger.info(f"  vs {enemy_id}: {advantage:.4f}")
                    
                except Exception as e:
                    logger.error(f"Failed to calculate advantage for {friendly_id} vs {enemy_id}: {e}")
                    advantages[enemy_id] = -np.inf
            
            # 选择最优目标
            if advantages:
                best_enemy = max(advantages.keys(), key=lambda x: advantages[x])
                best_advantage = advantages[best_enemy]
                logger.info(f"  Best target: {best_enemy} (advantage: {best_advantage:.4f})")
        
        logger.info("Target selection test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Target selection test failed: {e}")
        return False

def main():
    success = test_target_selection()
    if success:
        logger.info("✅ Target selection test PASSED")
    else:
        logger.error("❌ Target selection test FAILED")

if __name__ == "__main__":
    main() 