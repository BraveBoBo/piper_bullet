import numpy as np

SCENE_INFO ={
    'rigid_grasp': {
        'entities': {
            # 'urdf/borders.urdf': {
            #     'basePosition': [-1.5, 2.0, 0.5],
            #     'baseOrientation': [0, 0, 0],
            #     'globalScaling': 4.0,
            #     'mass': 0,
            #     'useTexture': True,
            # },
            # 'ycb/004_sugar_box/google_16k/textured.obj': {
            # 'ycb/009_gelatin_box/google_16k/textured.obj': {
            'ycb/003_cracker_box/google_16k/textured.obj': {
                'basePosition': [1.8, 1.7, 0.25],
                'baseOrientation': [0, 0, 0],
                'globalScaling': 7.0,
                'mass': 0.01,
                'rgbaColor': (0.9, 0.75, 0.65, 1),
            },
            # 'ycb/005_tomato_soup_can/google_16k/textured.obj': {
            # 'ycb/007_tuna_fish_can/google_16k/textured.obj': {
            # 'ycb/002_master_chef_can/google_16k/textured.obj': {
            #     'basePosition': [0.9, 1.5, 0.25],
            #     'baseOrientation': [0, 0, 0],
            #     'globalScaling': 8.0,
            #     'mass': 0.01,
            #     'rgbaColor': (0.9, 0.75, 0.65, 1),
            # },
        },
        'goal_pos': [[-2.5, 2.0, 0.5]],
    },
}
scene_name = 'rigid_grasp'


# robot info
ROBOT_INFO ={
        'piper': {# piper_description/urdf/piper_description.urdf
        'file_name':'piper_description/urdf/piper_description.urdf',
        'ee_joint_name': 'joint6', # arm end effector joint
        'ee_link_name': 'gripper_base',
        'global_scaling': 10.0,
        'use_fixed_base': True,
        'base_pos': np.array([5.0, 1.5, 0]),
        'rest_arm_qpos': None
    },
}

# Default camera values: [distance, pitch, yaw, posX, posY, posZ]
DEFAULT_CAM = [11.4, -22.4, 257, -0.08, -0.29, 1.8]

# Info for camera rendering without debug visualizer.
# projectionMatrix is output 3 from pybullet.getDebugVisualizerCamera()
DEFAULT_CAM_PROJECTION = {
    # Camera info for {cameraDistance: 11.0, cameraYaw: 140,
    # cameraPitch: -40, cameraTargetPosition: array([0., 0., 0.])}
    'projectionMatrix': (1.0, 0.0, 0.0, 0.0,
                         0.0, 1.0, 0.0, 0.0,
                         0.0, 0.0, -1.0000200271606445, -1.0,
                         0.0, 0.0, -0.02000020071864128, 0.0)
}