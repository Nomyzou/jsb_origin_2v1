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
DAMAGE_DISTANCE_THRESHOLD = 5.0  # 伤害距离阈值 (米)
DAMAGE_ANGLE_THRESHOLD = np.pi/3     # 伤害角度阈值 (60度，弧度)
MAX_DAMAGE_PER_STEP = 1.0           # 每步最大伤害值
DAMAGE_BASE_RATE = 2              # 基础伤害率

def calculate_damage(ego_feature, enm_feature, distance_threshold=DAMAGE_DISTANCE_THRESHOLD, 
                    angle_threshold=DAMAGE_ANGLE_THRESHOLD, base_rate=DAMAGE_BASE_RATE):
    """
    计算基于角度和距离的伤害值 - 修正版本
    
    Args:
        ego_feature: 己方飞机特征 (north, east, down, vn, ve, vd)
        enm_feature: 敌方飞机特征 (north, east, down, vn, ve, vd)
        distance_threshold: 伤害距离阈值 (米)
        angle_threshold: 伤害角度阈值 (弧度，默认60度)
        base_rate: 基础伤害率
    
    Returns:
        damage: 我方对敌方的伤害值 (0-MAX_DAMAGE_PER_STEP之间)
    """
    try:
        # 计算AO和TA角度以及距离
        AO, TA, R, _ = get_AO_TA_R(ego_feature, enm_feature, return_side=True)
        
        # 检查是否满足伤害条件
        if R > distance_threshold:
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
        else:
            angle_factor = 0.0
        
        # 计算基于距离的伤害 (距离越近伤害越大)
        distance_factor = 3.0 - (R / distance_threshold)
        distance_factor = np.clip(distance_factor, 0, 10)
        
        # 综合伤害计算
        damage = base_rate * angle_factor * distance_factor
        damage = np.clip(damage, 0, MAX_DAMAGE_PER_STEP)
        
        return damage
        
    except Exception as e:
        logger.error(f"Error calculating damage: {e}")
        return 0.0

def test_distance_calculation():
    """测试距离计算和坐标转换"""
    
    # 测试参数
    center_lon = 120.0  # 经度
    center_lat = 60.0   # 纬度  
    center_alt = 0.0    # 高度
    
    logger.info("=== 距离计算测试 ===")
    logger.info(f"战场中心坐标: 经度={center_lon}°, 纬度={center_lat}°, 高度={center_alt}m")
    
    # 测试案例1: 两架飞机在相同位置
    logger.info("\n--- 测试案例1: 相同位置 ---")
    lon1, lat1, alt1 = 120.0, 60.0, 1000.0
    lon2, lat2, alt2 = 120.0, 60.0, 1000.0
    
    logger.info(f"飞机1: 经度={lon1}°, 纬度={lat1}°, 高度={alt1}m")
    logger.info(f"飞机2: 经度={lon2}°, 纬度={lat2}°, 高度={alt2}m")
    
    # 转换为NEU坐标
    ned1 = LLA2NEU(lon1, lat1, alt1, center_lon, center_lat, center_alt)
    ned2 = LLA2NEU(lon2, lat2, alt2, center_lon, center_lat, center_alt)
    
    logger.info(f"飞机1 NEU坐标: North={ned1[0]:.2f}m, East={ned1[1]:.2f}m, Up={ned1[2]:.2f}m")
    logger.info(f"飞机2 NEU坐标: North={ned2[0]:.2f}m, East={ned2[1]:.2f}m, Up={ned2[2]:.2f}m")
    
    # 计算距离
    distance = np.linalg.norm([ned2[0] - ned1[0], ned2[1] - ned1[1], ned2[2] - ned1[2]])
    logger.info(f"计算距离: {distance:.2f}m")
    
    # 测试案例2: 两架飞机相距1度经度
    logger.info("\n--- 测试案例2: 相距1度经度 ---")
    lon1, lat1, alt1 = 120.0, 60.0, 1000.0
    lon2, lat2, alt2 = 121.0, 60.0, 1000.0  # 向东1度
    
    logger.info(f"飞机1: 经度={lon1}°, 纬度={lat1}°, 高度={alt1}m")
    logger.info(f"飞机2: 经度={lon2}°, 纬度={lat2}°, 高度={alt2}m")
    
    ned1 = LLA2NEU(lon1, lat1, alt1, center_lon, center_lat, center_alt)
    ned2 = LLA2NEU(lon2, lat2, alt2, center_lon, center_lat, center_alt)
    
    logger.info(f"飞机1 NEU坐标: North={ned1[0]:.2f}m, East={ned1[1]:.2f}m, Up={ned1[2]:.2f}m")
    logger.info(f"飞机2 NEU坐标: North={ned2[0]:.2f}m, East={ned2[1]:.2f}m, Up={ned2[2]:.2f}m")
    
    distance = np.linalg.norm([ned2[0] - ned1[0], ned2[1] - ned1[1], ned2[2] - ned1[2]])
    logger.info(f"计算距离: {distance:.2f}m")
    
    # 理论计算：在60度纬度处，1度经度约等于55.8km
    theoretical_distance = 55800  # 约55.8km
    logger.info(f"理论距离(60°纬度处1度经度): {theoretical_distance}m")
    logger.info(f"误差: {abs(distance - theoretical_distance):.2f}m")
    
    # 测试案例3: 两架飞机相距1度纬度
    logger.info("\n--- 测试案例3: 相距1度纬度 ---")
    lon1, lat1, alt1 = 120.0, 60.0, 1000.0
    lon2, lat2, alt2 = 120.0, 61.0, 1000.0  # 向北1度
    
    logger.info(f"飞机1: 经度={lon1}°, 纬度={lat1}°, 高度={alt1}m")
    logger.info(f"飞机2: 经度={lon2}°, 纬度={lat2}°, 高度={alt2}m")
    
    ned1 = LLA2NEU(lon1, lat1, alt1, center_lon, center_lat, center_alt)
    ned2 = LLA2NEU(lon2, lat2, alt2, center_lon, center_lat, center_alt)
    
    logger.info(f"飞机1 NEU坐标: North={ned1[0]:.2f}m, East={ned1[1]:.2f}m, Up={ned1[2]:.2f}m")
    logger.info(f"飞机2 NEU坐标: North={ned2[0]:.2f}m, East={ned2[1]:.2f}m, Up={ned2[2]:.2f}m")
    
    distance = np.linalg.norm([ned2[0] - ned1[0], ned2[1] - ned1[1], ned2[2] - ned1[2]])
    logger.info(f"计算距离: {distance:.2f}m")
    
    # 理论计算：1度纬度约等于111km
    theoretical_distance = 111000  # 约111km
    logger.info(f"理论距离(1度纬度): {theoretical_distance}m")
    logger.info(f"误差: {abs(distance - theoretical_distance):.2f}m")
    
    # 测试案例4: 两架飞机相距10km
    logger.info("\n--- 测试案例4: 相距约10km ---")
    lon1, lat1, alt1 = 120.0, 60.0, 1000.0
    # 在60度纬度处，10km约等于0.179度经度
    lon2, lat2, alt2 = 120.0 + 0.179, 60.0, 1000.0
    
    logger.info(f"飞机1: 经度={lon1}°, 纬度={lat1}°, 高度={alt1}m")
    logger.info(f"飞机2: 经度={lon2:.3f}°, 纬度={lat2}°, 高度={alt2}m")
    
    ned1 = LLA2NEU(lon1, lat1, alt1, center_lon, center_lat, center_alt)
    ned2 = LLA2NEU(lon2, lat2, alt2, center_lon, center_lat, center_alt)
    
    logger.info(f"飞机1 NEU坐标: North={ned1[0]:.2f}m, East={ned1[1]:.2f}m, Up={ned1[2]:.2f}m")
    logger.info(f"飞机2 NEU坐标: North={ned2[0]:.2f}m, East={ned2[1]:.2f}m, Up={ned2[2]:.2f}m")
    
    distance = np.linalg.norm([ned2[0] - ned1[0], ned2[1] - ned1[1], ned2[2] - ned1[2]])
    logger.info(f"计算距离: {distance:.2f}m")
    logger.info(f"预期距离: 10000m")
    logger.info(f"误差: {abs(distance - 10000):.2f}m")
    
    # 测试案例5: 高度差测试
    logger.info("\n--- 测试案例5: 高度差1000m ---")
    lon1, lat1, alt1 = 120.0, 60.0, 1000.0
    lon2, lat2, alt2 = 120.0, 60.0, 2000.0  # 高度差1000m
    
    logger.info(f"飞机1: 经度={lon1}°, 纬度={lat1}°, 高度={alt1}m")
    logger.info(f"飞机2: 经度={lon2}°, 纬度={lat2}°, 高度={alt2}m")
    
    ned1 = LLA2NEU(lon1, lat1, alt1, center_lon, center_lat, center_alt)
    ned2 = LLA2NEU(lon2, lat2, alt2, center_lon, center_lat, center_alt)
    
    logger.info(f"飞机1 NEU坐标: North={ned1[0]:.2f}m, East={ned1[1]:.2f}m, Up={ned1[2]:.2f}m")
    logger.info(f"飞机2 NEU坐标: North={ned2[0]:.2f}m, East={ned2[1]:.2f}m, Up={ned2[2]:.2f}m")
    
    distance = np.linalg.norm([ned2[0] - ned1[0], ned2[1] - ned1[1], ned2[2] - ned1[2]])
    logger.info(f"计算距离: {distance:.2f}m")
    logger.info(f"预期距离: 1000m")
    logger.info(f"误差: {abs(distance - 1000):.2f}m")

def test_damage_distance_threshold():
    """测试伤害距离阈值"""
    
    logger.info("\n=== 伤害距离阈值测试 ===")
    
    # 从原代码中获取的阈值
    DAMAGE_DISTANCE_THRESHOLD = 5.0  # 米
    logger.info(f"当前伤害距离阈值: {DAMAGE_DISTANCE_THRESHOLD}m")
    
    # 测试不同距离下的伤害计算
    center_lon, center_lat, center_alt = 120.0, 60.0, 0.0
    
    # 创建测试飞机特征
    def create_aircraft_feature(lon, lat, alt, vn=0, ve=0, vd=0):
        ned = LLA2NEU(lon, lat, alt, center_lon, center_lat, center_alt)
        return np.array([*ned, vn, ve, vd])
    
    # 测试距离: 1m, 5m, 10m, 100m, 1000m, 10000m
    test_distances = [1, 5, 10, 100, 1000, 10000]
    
    for distance in test_distances:
        logger.info(f"\n--- 测试距离: {distance}m ---")
        
        # 创建两架飞机，相距指定距离
        lon1, lat1, alt1 = 120.0, 60.0, 1000.0
        # 在60度纬度处，距离转换为经度差
        lon_diff = distance / 55800  # 约55.8km/度
        lon2 = lon1 + lon_diff
        
        ego_feature = create_aircraft_feature(lon1, lat1, alt1)
        enm_feature = create_aircraft_feature(lon2, lat1, alt1)
        
        # 计算实际距离
        actual_distance = np.linalg.norm(enm_feature[:3] - ego_feature[:3])
        logger.info(f"实际计算距离: {actual_distance:.2f}m")
        
        # 计算伤害
        damage = calculate_damage(ego_feature, enm_feature)
        logger.info(f"计算伤害: {damage:.4f}")
        
        # 检查是否超过阈值
        if actual_distance > DAMAGE_DISTANCE_THRESHOLD:
            logger.info(f"距离超过阈值，应该无伤害: {damage == 0}")
        else:
            logger.info(f"距离在阈值内，可能有伤害: {damage > 0}")

def test_ao_ta_calculation():
    """测试AO和TA角度计算"""
    
    logger.info("\n=== AO/TA角度计算测试 ===")
    
    center_lon, center_lat, center_alt = 120.0, 60.0, 0.0
    
    # 测试案例1: 两架飞机相对飞行
    logger.info("\n--- 测试案例1: 相对飞行 ---")
    lon1, lat1, alt1 = 120.0, 60.0, 1000.0
    lon2, lat2, alt2 = 120.001, 60.0, 1000.0  # 相距约55.8m
    
    ned1 = LLA2NEU(lon1, lat1, alt1, center_lon, center_lat, center_alt)
    ned2 = LLA2NEU(lon2, lat2, alt2, center_lon, center_lat, center_alt)
    
    ego_feature = np.array([*ned1, 100, 0, 0])  # 向东飞行
    enm_feature = np.array([*ned2, -100, 0, 0])  # 向西飞行
    
    AO, TA, R, side_flag = get_AO_TA_R(ego_feature, enm_feature, return_side=True)
    
    logger.info(f"AO角度: {AO:.4f} rad ({np.degrees(AO):.2f}°)")
    logger.info(f"TA角度: {TA:.4f} rad ({np.degrees(TA):.2f}°)")
    logger.info(f"距离: {R:.2f}m")
    logger.info(f"侧向标志: {side_flag}")
    
    # 测试案例2: 两架飞机同向飞行
    logger.info("\n--- 测试案例2: 同向飞行 ---")
    ego_feature = np.array([*ned1, 100, 0, 0])  # 向东飞行
    enm_feature = np.array([*ned2, 100, 0, 0])  # 也向东飞行
    
    AO, TA, R, side_flag = get_AO_TA_R(ego_feature, enm_feature, return_side=True)
    
    logger.info(f"AO角度: {AO:.4f} rad ({np.degrees(AO):.2f}°)")
    logger.info(f"TA角度: {TA:.4f} rad ({np.degrees(TA):.2f}°)")
    logger.info(f"距离: {R:.2f}m")
    logger.info(f"侧向标志: {side_flag}")

if __name__ == "__main__":
    test_distance_calculation()
    test_damage_distance_threshold()
    test_ao_ta_calculation()
