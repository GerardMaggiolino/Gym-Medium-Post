import pybullet as p
import os
import math


class Rib:
    def __init__(self, client, base):
        f_name = os.path.join(os.path.dirname(__file__), 'rib.urdf')
        roll = 0
        pitch = 0
        yaw = 0  # Example: 90-degree rotation around Z-axis

        # Convert Euler angles to radians
        roll_rad = math.radians(roll)
        pitch_rad = math.radians(pitch)
        yaw_rad = math.radians(yaw)

        # Calculate the quaternion from Euler angles
        orientation = p.getQuaternionFromEuler([roll_rad, pitch_rad, yaw_rad])
        p.loadURDF(fileName=f_name,
                   basePosition=[base[0], base[1], 0],
                   baseOrientation=orientation,
                   physicsClientId=client)