import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from envs.JSBSim.utils.utils import LLA2NEU, get_AO_TA_R
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 伤害计算相关常量
DAMAGE_DISTANCE_THRESHOLD = 5000.0  # 伤害距离阈值 (米) - 修正为5公里
DAMAGE_ANGLE_THRESHOLD = np.pi/3     # 伤害角度阈值 (60度，弧度)
MAX_DAMAGE_PER_STEP = 1.0           # 每步最大伤害值
DAMAGE_BASE_RATE = 2              # 基础伤害率

def calculate_damage(ego_feature, enm_feature, distance_threshold=DAMAGE_DISTANCE_THRESHOLD, 
                    angle_threshold=DAMAGE_ANGLE_THRESHOLD, base_rate=DAMAGE_BASE_RATE):
    """
    计算基于角度和距离的伤害值 - 修正版本
    """
    try:
        # 计算AO和TA角度以及距离
        AO, TA, R, _ = get_AO_TA_R(ego_feature, enm_feature, return_side=True)
        
        logger.info(f"  距离: {R:.2f}m, AO: {np.degrees(AO):.2f}°, TA: {np.degrees(TA):.2f}°")
        
        # 检查是否满足伤害条件
        if R > distance_threshold:
            logger.info(f"  距离超过阈值 {distance_threshold}m，无伤害")
            return 0.0
        
        # 正确的伤害计算逻辑：
        # 当AO < angle_threshold时，我方对敌方造成伤害
        # AO越小伤害越大，TA越大伤害越大
        
        if AO < angle_threshold:  # 我方朝向敌机（AO < 60度）
            # AO角度因子：AO越小，伤害越大
            ao_factor = 1.0 - (AO / angle_threshold)
            ao_factor = np.clip(ao_factor, 0, 1)
            
            # TA角度因子：TA越大，伤害越大（敌机越难逃脱）
            # TA从0到π，我们希望在TA较大时伤害更大
            ta_factor = TA / np.pi
            ta_factor = np.clip(ta_factor, 0, 1)
            
            # 综合角度因子
            angle_factor = ao_factor * ta_factor
            
            logger.info(f"  AO因子: {ao_factor:.4f}, TA因子: {ta_factor:.4f}, 角度因子: {angle_factor:.4f}")
        else:
            angle_factor = 0.0
            logger.info(f"  AO角度 {np.degrees(AO):.2f}° 超过阈值 {np.degrees(angle_threshold):.2f}°，无伤害")
        
        # 计算基于距离的伤害 (距离越近伤害越大)
        distance_factor = 3.0 - (R / distance_threshold)
        distance_factor = np.clip(distance_factor, 0, 10)
        
        logger.info(f"  距离因子: {distance_factor:.4f}")
        
        # 综合伤害计算
        damage = base_rate * angle_factor * distance_factor
        damage = np.clip(damage, 0, MAX_DAMAGE_PER_STEP)
        
        logger.info(f"  最终伤害: {damage:.4f}")
        return damage
        
    except Exception as e:
        logger.error(f"Error calculating damage: {e}")
        return 0.0

def test_realistic_scenarios():
    """测试真实场景"""
    
    logger.info("=== 测试真实空战场景 ===")
    logger.info(f"伤害距离阈值: {DAMAGE_DISTANCE_THRESHOLD}m ({DAMAGE_DISTANCE_THRESHOLD/1000:.1f}km)")
    logger.info(f"伤害角度阈值: {np.degrees(DAMAGE_ANGLE_THRESHOLD):.1f}°")
    
    center_lon, center_lat, center_alt = 120.0, 60.0, 0.0
    
    # 创建测试飞机特征
    def create_aircraft_feature(lon, lat, alt, vn=0, ve=0, vd=0):
        ned = LLA2NEU(lon, lat, alt, center_lon, center_lat, center_alt)
        return np.array([*ned, vn, ve, vd])
    
    # 测试场景1: 飞机朝向目标（小AO角度）
    logger.info(f"\n--- 场景1: 飞机朝向目标 (AO < 60°) ---")
    
    # 两架飞机相距2km，我方飞机朝向敌方
    lon1, lat1, alt1 = 120.0, 60.0, 1000.0
    lon2, lat2, alt2 = 120.0 + 0.036, 60.0, 1000.0  # 约2km距离
    
    # 我方飞机朝向敌方（速度方向指向敌方）
    # 计算方向向量
    ned1 = LLA2NEU(lon1, lat1, alt1, center_lon, center_lat, center_alt)
    ned2 = LLA2NEU(lon2, lat2, alt2, center_lon, center_lat, center_alt)
    direction_vector = ned2 - ned1
    direction_vector = direction_vector / np.linalg.norm(direction_vector)  # 归一化
    
    # 设置速度（朝向敌方）
    speed = 200  # 200m/s
    ego_feature = np.array([*ned1, *direction_vector * speed])
    enm_feature = np.array([*ned2, 100, 0, 0])  # 敌方固定速度
    
    # 计算实际距离
    actual_distance = np.linalg.norm(enm_feature[:3] - ego_feature[:3])
    logger.info(f"实际距离: {actual_distance:.2f}m ({actual_distance/1000:.2f}km)")
    
    # 计算伤害
    damage = calculate_damage(ego_feature, enm_feature)
    logger.info(f"伤害结果: {damage:.4f}")
    
    # 测试场景2: 飞机背向目标（大AO角度）
    logger.info(f"\n--- 场景2: 飞机背向目标 (AO > 60°) ---")
    
    # 我方飞机背向敌方（速度方向背离敌方）
    ego_feature = np.array([*ned1, *(-direction_vector * speed)])
    enm_feature = np.array([*ned2, 100, 0, 0])  # 敌方固定速度
    
    # 计算实际距离
    actual_distance = np.linalg.norm(enm_feature[:3] - ego_feature[:3])
    logger.info(f"实际距离: {actual_distance:.2f}m ({actual_distance/1000:.2f}km)")
    
    # 计算伤害
    damage = calculate_damage(ego_feature, enm_feature)
    logger.info(f"伤害结果: {damage:.4f}")
    
    # 测试场景3: 近距离正面交锋
    logger.info(f"\n--- 场景3: 近距离正面交锋 ---")
    
    # 两架飞机相距500m，正面相对
    lon1, lat1, alt1 = 120.0, 60.0, 1000.0
    lon2, lat2, alt2 = 120.0 + 0.009, 60.0, 1000.0  # 约500m距离
    
    ned1 = LLA2NEU(lon1, lat1, alt1, center_lon, center_lat, center_alt)
    ned2 = LLA2NEU(lon2, lat2, alt2, center_lon, center_lat, center_alt)
    direction_vector = ned2 - ned1
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    
    # 正面相对
    ego_feature = np.array([*ned1, *direction_vector * speed])
    enm_feature = np.array([*ned2, *(-direction_vector * speed)])
    
    # 计算实际距离
    actual_distance = np.linalg.norm(enm_feature[:3] - ego_feature[:3])
    logger.info(f"实际距离: {actual_distance:.2f}m ({actual_distance/1000:.2f}km)")
    
    # 计算伤害
    damage = calculate_damage(ego_feature, enm_feature)
    logger.info(f"伤害结果: {damage:.4f}")

if __name__ == "__main__":
    test_realistic_scenarios()
