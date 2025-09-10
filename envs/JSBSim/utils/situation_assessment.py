import numpy as np
from math import radians, cos, sin, exp, pi

# ##################################################################
# This file is adapted from calculate_angles_corrected.py
# It is refactored to calculate situation assessment between two
# aircraft simulators within the JSBSim environment.
# ##################################################################


def body_to_earth_rotation_matrix(roll, pitch, yaw):
    sin_roll, cos_roll = sin(roll), cos(roll)
    sin_pitch, cos_pitch = sin(pitch), cos(pitch)
    sin_yaw, cos_yaw = sin(yaw), cos(yaw)

    rot_matrix = np.array([
        [cos_pitch * sin_yaw, cos_roll * cos_yaw + sin_roll * sin_pitch * sin_yaw,
         sin_roll * cos_yaw - cos_roll * sin_pitch * sin_yaw],
        [cos_pitch * cos_yaw, -cos_roll * sin_yaw + sin_roll * sin_pitch * cos_yaw,
         -sin_roll * sin_yaw - cos_roll * sin_pitch * cos_yaw],
        [sin_pitch, -sin_roll * cos_pitch, cos_roll * cos_pitch]
    ])
    return rot_matrix

def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def angle_advantage(phi_u, phi_q):
    return 1 - (abs(phi_u) + abs(phi_q)) / (2 * pi)

def distance_advantage(R, Rw=5.0, sigma=2.0):
    if R <= Rw:
        return 1.0
    else:
        return exp(-((R - Rw) ** 2) / (2 * sigma ** 2))

def calc_vop(R, vT, vmax=0.3, Rw=5.0):
    if R > Rw:
        return vT + (vmax - vT) * (1 - exp(-(R - Rw) / Rw))
    else:
        return vT

def speed_advantage(v, v_op):
    if v_op == 0:
        return 0.0
    return (v / v_op) * exp(-2 * abs(v - v_op) / v_op)

def height_advantage(delta_h, sigma_h=0.5):
    if delta_h <= 0:
        return exp(-(delta_h ** 2) / (2 * sigma_h ** 2))
    elif 0 < delta_h <= sigma_h:
        return 1.0
    else:
        return exp(-((delta_h - sigma_h) ** 2) / (2 * sigma_h ** 2))

def judge_battle_situation(azimuth_rad, aspect_rad):
    az = abs(azimuth_rad)
    asp = abs(aspect_rad)
    half_pi = pi / 2
    if 0 <= az <= half_pi and 0 <= asp <= half_pi:
        return 'a'
    elif 0 <= az <= half_pi and half_pi < asp <= pi:
        return 'c'
    elif half_pi < az <= pi and 0 <= asp <= half_pi:
        return 'b'
    else:
        return 'd'

def calc_overall_situation_value(f_ang, f_d, f_v, f_h, situation):
    DYNAMIC_WEIGHTS = {
        'a': [0.332, 0.291, 0.209, 0.168],
        'b': [0.325, 0.210, 0.278, 0.287],
        'c': [0.239, 0.328, 0.278, 0.155],
        'd': [0.111, 0.313, 0.403, 0.173],
    }
    STATIC_WEIGHTS = [0.8, 0.1, 0.05, 0.05]

    dynamic_w = DYNAMIC_WEIGHTS[situation]
    final_w = [0.5 * dw + 0.5 * sw for dw, sw in zip(dynamic_w, STATIC_WEIGHTS)]
    vals = [f_ang, f_d, f_v, f_h]
    return sum([v * w for v, w in zip(vals, final_w)])


def get_situation_adv(ego_state, enm_state, center_lon=120.0, center_lat=60.0, center_alt=0.0):
    """
    Calculates the situational advantage of an ego aircraft against an enemy aircraft.
    
    Args:
        ego_state: List of ego aircraft state variables [lon, lat, alt, roll, pitch, yaw, vx, vy, vz]
        enm_state: List of enemy aircraft state variables [lon, lat, alt, roll, pitch, yaw, vx, vy, vz]
        center_lon: Center longitude (default: 120.0)
        center_lat: Center latitude (default: 60.0)
        center_alt: Center altitude (default: 0.0)
    
    Returns:
        float: Situational advantage value
    """
    R_earth = 6371.0
    origin_lon_rad = radians(center_lon)
    origin_lat_rad = radians(center_lat)

    # Ego position
    lat0_rad = radians(ego_state[1])
    lon0_rad = radians(ego_state[0])
    y0 = R_earth * (lat0_rad - origin_lat_rad)
    x0 = R_earth * (lon0_rad - origin_lon_rad) * cos(origin_lat_rad)
    z0 = ego_state[2] / 1000.0

    # Enemy position
    lat1_rad = radians(enm_state[1])
    lon1_rad = radians(enm_state[0])
    y1 = R_earth * (lat1_rad - origin_lat_rad)
    x1 = R_earth * (lon1_rad - origin_lon_rad) * cos(origin_lat_rad)
    z1 = enm_state[2] / 1000.0

    rel_pos_vector = np.array([x1 - x0, y1 - y0, z1 - z0])
    rel_pos_2d = rel_pos_vector[:2]

    # Ego velocity
    body_vel0 = np.array([ego_state[9], ego_state[10], ego_state[11]]) / 1000.0
    rot0 = body_to_earth_rotation_matrix(ego_state[3], ego_state[4], ego_state[5])
    v0 = rot0.dot(body_vel0)
    v0_2d = v0[:2]

    # Enemy velocity
    body_vel1 = np.array([enm_state[9], enm_state[10], enm_state[11]]) / 1000.0
    rot1 = body_to_earth_rotation_matrix(enm_state[3], enm_state[4], enm_state[5])
    v1 = rot1.dot(body_vel1)
    v1_2d = v1[:2]

    # Angles
    rel_pos_norm = np.linalg.norm(rel_pos_2d)
    if rel_pos_norm == 0: return -np.inf
    rel_pos_unit = rel_pos_2d / rel_pos_norm

    v0_norm = np.linalg.norm(v0_2d)
    if v0_norm == 0: return -np.inf
    v0_unit = v0_2d / v0_norm

    v1_norm = np.linalg.norm(v1_2d)
    if v1_norm == 0: return -np.inf
    v1_unit = v1_2d / v1_norm

    cos_azimuth = np.clip(np.dot(v0_unit, rel_pos_unit), -1.0, 1.0)
    azimuth_rad = np.arccos(cos_azimuth)
    if np.cross(v0_unit, rel_pos_unit) < 0:
        azimuth_rad = -azimuth_rad

    cos_aspect = np.clip(np.dot(v1_unit, rel_pos_unit), -1.0, 1.0)
    aspect_rad = np.arccos(cos_aspect)
    if np.cross(v1_unit, rel_pos_unit) < 0:
        aspect_rad = -aspect_rad

    # Advantage functions
    distance = np.linalg.norm(rel_pos_vector)
    azimuth_rad_norm = normalize_angle(azimuth_rad)
    aspect_rad_norm = normalize_angle(aspect_rad)

    f_ang = angle_advantage(azimuth_rad_norm, aspect_rad_norm)
    f_d = distance_advantage(distance)
    vop = calc_vop(distance, v1_norm)
    f_v = speed_advantage(v0_norm, vop)
    delta_h = z0 - z1
    f_h = height_advantage(delta_h)

    situation = judge_battle_situation(azimuth_rad, aspect_rad)
    overall_adv = calc_overall_situation_value(f_ang, f_d, f_v, f_h, situation)

    return overall_adv 