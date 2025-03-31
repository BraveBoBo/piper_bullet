# calculate the camera parameters for the camera
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
import cv2
import json
import os


camera ={
    "view_matrix": [ ],
    "proj_matrix": [ ],
    "cam_forward": [ ],
    "cam_horiz": [ ],
    "cam_vert": [ ],
    "cam_dist": 0.,
    "cam_tgt": [ ],
    "cam_resolution": (200,200),
}

camerapos=[0,0,8]

def cal_d435(cameraPos = camerapos):
    global viewMatrix, rot
    
    width = 1280     # 图像宽度 (px)
    height = 720     # 图像高度 (px)
    fov = 69.4       # RealSense D435 标称水平FOV约69.4度
    aspect = width / height
    near = 0.01
    far = 20.0
    
    # 相机默认无旋转
    qq = np.array([0, 0, 0, 1])
    rot = R.from_quat(qq).as_matrix()
    
    # 相机朝向（forward为负Z轴）
    forward_vector = np.array([0, 0, -1])
    up_vector = np.array([0, -1, 0])  # Y轴负方向向上
    
    targetPos = np.matmul(rot, forward_vector) + cameraPos
    cameraupPos = np.matmul(rot, up_vector)
    
    # 计算视图矩阵
    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=cameraPos,
        cameraTargetPosition=targetPos,
        cameraUpVector=cameraupPos
    )
    
    # 计算投影矩阵
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=aspect,
        nearVal=near,
        farVal=far
    )
    
    print("Projection Matrix:\n", projection_matrix)
    print("View Matrix:\n", viewMatrix)

    # 转换为NumPy矩阵格式
    # viewMatrix = np.array(viewMatrix).reshape(4, 4).T
    # projection_matrix = np.array(projection_matrix).reshape(4, 4).T
    
    return viewMatrix, projection_matrix


# d435
camera["cam_dist"] = np.linalg.norm(camerapos)
camera['view_matrix'] = cal_d435()[0]
camera['proj_matrix'] = cal_d435()[1]
camera['cam_forward'] = [0, 0, -1]
camera['cam_horiz'] = [1, 0, 0]
camera['cam_vert'] = [0, 1, 0]
camera['cam_tgt'] = [0, 0, 0]
camera['cam_resolution'] = (1280, 720)

save_path = os.path.join(os.path.dirname(__file__))
json.dump(camera, open(os.path.join(save_path, 'camview_d435.json'), 'w'))
