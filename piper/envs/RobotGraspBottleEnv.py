import os
import time
import sys
sys.path.append("/home/libo/project/simulator/piper_bullet")
import wandb

import cv2
from matplotlib import cm  # for colors
import numpy as np



import gym
import pybullet
import pybullet_utils.bullet_client as bclient
import pybullet_data

from piper.constant import SCENE_INFO,scene_name,ROBOT_INFO,DEFAULT_CAM, DEFAULT_CAM_PROJECTION
from piper.bullet_manipulator import BulletManipulator
from piper.camera_utils import cameraConfig
# --------------------
# Environment for rigid body grasping
# --------------------



class RigidGrasping(gym.Env):
    MAX_OBS_VEL = 20.0  # max vel (in m/s) for the anchor observations
    MAX_ACT_VEL = 10.0  # max vel (in m/s) for the anchor actions
    WORKSPACE_BOX_SIZE = 200.0  # workspace box limits (needs to be >=1)
    STEPS_AFTER_DONE = 500     # steps after releasing anchors at the end
    FORCE_REWARD_MULT = 1e-4   # scaling for the force penalties
    FINAL_REWARD_MULT = 400    # multiply the final reward (for sparse rewards)
    SUCESS_REWARD_TRESHOLD = 2.5  # approx. threshold for task success/failure
    ORI_SIZE = 3 * 2  # 3D position + sin,cos for 3 Euler angles
    FING_DIST = 0.01  # default finger distance



    def __init__(self, args):
        self.args = args
        self.cam_on = args.cam_resolution > 0
        self.sim = bclient.BulletClient(
            connection_mode=pybullet.GUI if args.viz else pybullet.DIRECT)
        if self.args.viz:  # no rendering during load
            self.sim.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
            
        reset_bullet(self.args, self.sim, args.plane_texture_file, args.debug)

        self.num_anchors = 1  

        # load the scene
        res = self.load_objects(self.sim, args, args.debug)
        self.rigid_ids ,self.goal_poses = res

        # load the floor
        self.load_floor(self.sim, debug=args.debug)


        # load the robot
        self.load_robot(self.sim, args, debug=args.debug)


        self.max_episode_len = self.args.max_episode_len
                # Define sizes of observation and action spaces.
        self.gripper_lims = np.tile(np.concatenate(
            [RigidGrasping.WORKSPACE_BOX_SIZE * np.ones(3),  # 3D pos
             np.ones(3)]), self.num_anchors)             # 3D linvel/MAX_OBS_VEL   

        if args.cam_resolution <= 0:  # report gripper positions as low-dim obs
            self.observation_space = gym.spaces.Box(
                -1.0 * self.gripper_lims, self.gripper_lims)
            
        else:  # RGB WxHxC
            shape = (args.cam_resolution, args.cam_resolution, 3)
            if args.flat_obs:
                shape = (np.prod(shape),)
            self.observation_space = gym.spaces.Box(
                low=0, high=255 if args.uint8_pixels else 1.0,
                dtype=np.uint8 if args.uint8_pixels else np.float16,
                shape=shape)            


        act_sz = 3+1 # 3D pos + 1D finger distance
        act_sz += RigidGrasping.ORI_SIZE # sin,cos for 3 Euler angles
        self.action_space = gym.spaces.Box(  # [-1, 1]
            -1.0 * np.ones(self.num_anchors * act_sz),
            np.ones(self.num_anchors * act_sz))
        if self.args.debug:
            print('Wrapped as DeformEnvRobot with act', self.action_space)

    def seed(self, seed):
        np.random.seed(seed)
        
    @property
    def _cam_viewmat(self):
        dist, pitch, yaw, pos_x, pos_y, pos_z = self.args.cam_viewmat
        cam = {
            'distance': dist,
            'pitch': pitch,
            'yaw': yaw,
            'cameraTargetPosition': [pos_x, pos_y, pos_z],
            'upAxisIndex': 2,
            'roll': 0,
        }
        view_mat = self.sim.computeViewMatrixFromYawPitchRoll(**cam)

        return view_mat
    @property
    def anchor_ids(self):
        return list(self.anchors.keys())
        
    def load_objects(self,sim, args, debug=True):

        data_path = os.path.join(os.path.split(__file__)[0], '..', 'data')
        args.data_path = data_path
        sim.setAdditionalSearchPath(data_path)
        # Load rigid objects.
        rigid_ids = []
        for name, kwargs in SCENE_INFO[scene_name]['entities'].items():
            rgba_color = kwargs['rgbaColor'] if 'rgbaColor' in kwargs else None
            texture_file = None
            if 'useTexture' in kwargs and kwargs['useTexture']:
                texture_file = self.get_texture_path(args.rigid_texture_file)
            id = self.load_rigid_object(
                sim, os.path.join(data_path, name), kwargs['globalScaling'],
                kwargs['basePosition'], kwargs['baseOrientation'],
                kwargs.get('mass', 0.0), texture_file, rgba_color)
            rigid_ids.append(id)
        goal_poses = SCENE_INFO[scene_name]['goal_pos']

        return rigid_ids,np.array(goal_poses)

    def load_rigid_object(self, sim, obj_file_name, scale, init_pos, init_ori,
                        mass=0.0, texture_file=None, rgba_color=None):
        """Load a rigid object from file, create visual and collision shapes."""
        if obj_file_name.endswith('.obj'):  # mesh info
            xyz_scale = [scale, scale, scale]
            viz_shape_id = sim.createVisualShape(
                shapeType=pybullet.GEOM_MESH, rgbaColor=rgba_color,
                fileName=obj_file_name, meshScale=xyz_scale)
            col_shape_id = sim.createCollisionShape(
                shapeType=pybullet.GEOM_MESH,
                fileName=obj_file_name, meshScale=xyz_scale)
            rigid_id = sim.createMultiBody(
                baseMass=mass,  # mass==0 => fixed at position where it is loaded
                basePosition=init_pos,
                baseCollisionShapeIndex=col_shape_id,
                baseVisualShapeIndex=viz_shape_id,
                baseOrientation=pybullet.getQuaternionFromEuler(init_ori))
        elif obj_file_name.endswith('.urdf'):  # URDF file
            rigid_id = sim.loadURDF(
                obj_file_name, init_pos, pybullet.getQuaternionFromEuler(init_ori),
                useFixedBase=True if mass <= 0 else False, globalScaling=scale)
        else:
            print('Unknown file extension', obj_file_name)
            assert(False), 'load_rigid_object supports only obj and URDF files'
        sim.changeDynamics(rigid_id, -1, mass, lateralFriction=1.0,
                        spinningFriction=1.0, rollingFriction=1.0,
                        restitution=0.0)
        n_jt = sim.getNumJoints(rigid_id)

        if texture_file is not None:
            texture_id = sim.loadTexture(texture_file)
            kwargs = {}
            if hasattr(pybullet, 'VISUAL_SHAPE_DOUBLE_SIDED'):
                kwargs['flags'] = pybullet.VISUAL_SHAPE_DOUBLE_SIDED

            if obj_file_name.endswith('figure_headless.urdf'):
                sim.changeVisualShape(  # only changing the body of the figure
                    rigid_id, 0, rgbaColor=[1, 1, 1, 1],
                    textureUniqueId=texture_id, **kwargs)
            else:
                for i in range(-1, n_jt):
                    sim.changeVisualShape(
                        rigid_id, i, rgbaColor=[1,1,1,1],
                        textureUniqueId=texture_id, **kwargs)

        return rigid_id
    
    def load_floor(self,sim, plane_texture=None, debug=False):
        sim.setAdditionalSearchPath(pybullet_data.getDataPath())
        floor_id = sim.loadURDF('plane.urdf')
        if plane_texture is not None:
            if debug: print('texture file', plane_texture)
            texture_id = sim.loadTexture(plane_texture)
            sim.changeVisualShape(
                floor_id, -1, rgbaColor=[1,1,1,1], textureUniqueId=texture_id, )
        # assert(floor_id == 1)  # camera assumes floor/ground is loaded second, AFTER THE DEFORMABLE
        return sim
    
    def reset(self):
        self.stepnum = 0
        self.episode_reward = 0.0
        self.anchors = {}
        self.camera_config = cameraConfig.from_file(self.args.cam_config_path)

        if self.args.viz:  # no rendering during load
            self.sim.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        # Reset pybullet sim to clear out deformables and reload objects.
        plane_texture_path = os.path.join(
            self.args.data_path,  self.get_texture_path(
                self.args.plane_texture_file))
        reset_bullet(self.args, self.sim, plane_texture=plane_texture_path)
        res = self.load_objects(self.sim, self.args, self.args.debug)
        self.rigid_ids, self.goal_pos = res

        self.load_floor(self.sim, plane_texture=plane_texture_path,
                        debug=self.args.debug)
        
        self.load_robot(self.sim, self.args, debug=self.args.debug)
        
        self.sim.stepSimulation()  # step once to get initial state

        # Set up viz.
        if self.args.viz:  # loading done, so enable debug rendering if needed
            time.sleep(0.1)  # wait for debug visualizer to catch up
            self.sim.stepSimulation()  # step once to get initial state
            self.sim.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)


        obs, _ = self.get_obs()
        return obs

    def load_robot(self, sim,args,debug=False):
        """Load a robot from file, create visual and collision shapes."""
        data_path = os.path.join(os.path.split(__file__)[0], '..', 'data')
        sim.setAdditionalSearchPath(data_path)
        robot_info = ROBOT_INFO.get(f'piper', None)
        assert(robot_info is not None)  # make sure robot_info is ok
        robot_path = os.path.join(data_path, 'robots',
                                  robot_info['file_name'])
        if debug:
            print('Loading robot from', robot_path)
        self.robot = BulletManipulator(
            sim, robot_path, control_mode='velocity',
            ee_joint_name=robot_info['ee_joint_name'],
            ee_link_name=robot_info['ee_link_name'],
            base_pos=robot_info['base_pos'],
            base_quat=pybullet.getQuaternionFromEuler([0, 0, np.pi]),
            global_scaling=robot_info['global_scaling'],
            use_fixed_base=robot_info['use_fixed_base'],
            rest_arm_qpos=robot_info['rest_arm_qpos'],
            left_ee_joint_name=robot_info.get('left_ee_joint_name', None),
            left_ee_link_name=robot_info.get('left_ee_link_name', None),
            left_fing_link_prefix='panda_hand_l_', left_joint_suffix='_l',
            left_rest_arm_qpos=robot_info.get('left_rest_arm_qpos', None),
            debug=debug)
        
    def get_texture_path(self, file_path):
        # Get either pre-specified texture file or a random one.
        if self.args.use_random_textures:
            parent = os.path.dirname(file_path)
            full_parent_path = os.path.join(self.args.data_path, parent)
            randfile = np.random.choice(list(os.listdir(full_parent_path)))
            file_path = os.path.join(parent, randfile)
        return file_path


    
    def get_obs(self):
        done = False
        obs = self.render(mode='rgb_array', width=self.args.cam_resolution,
                            height=self.args.cam_resolution)
        if self.args.uint8_pixels:
            obs = obs.astype(np.uint8)  # already in [0,255]
        else:
            obs = obs.astype(np.float32)/255.0  # to [0,1]
            obs = np.clip(obs, 0, 1)
        if self.args.flat_obs:
            obs = obs.reshape(-1)
        atol = 0.0001
        if ((obs < self.observation_space.low-atol).any() or
            (obs > self.observation_space.high+atol).any()):
            print('obs', obs.shape, f'{np.min(obs):e}, n{np.max(obs):e}')
            assert self.observation_space.contains(obs)

        return obs, done


    def render(self, mode='rgb_array', width=1280, height=720):
        assert (mode == 'rgb_array')
        # w, h, rgba_px, _, _ = self.sim.getCameraImage(
        #     width=width, height=height,
        #     renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
        #     viewMatrix=self._cam_viewmat, **DEFAULT_CAM_PROJECTION)
        
        # if self.args.debug:
        #     print('Camera image', w, h, rgba_px.shape)
        CAM_PROJECTION = {
        # Camera info for {cameraDistance: 11.0, cameraYaw: 140,
        # cameraPitch: -40, cameraTargetPosition: array([0., 0., 0.])}
        'projectionMatrix': self.camera_config.proj_matrix,
        }
        w, h, rgba_px, _, _ = self.sim.getCameraImage(
            width=width, height=height,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=self.camera_config.view_matrix,**CAM_PROJECTION)
        
        # If getCameraImage() returns a tuple instead of numpy array that
        # means that pybullet was installed without numpy being present.
        # Uninstall pybullet, then install numpy, then install pybullet.
        assert (isinstance(rgba_px, np.ndarray)), 'Install numpy, then pybullet'
        img = rgba_px[:, :, 0:3]
        return img
    



def reset_bullet(args, sim, plane_texture=None, debug=False):
    """Reset/initialize pybullet simulation."""
    dist, pitch, yaw, pos_x, pos_y, pos_z = args.cam_viewmat
    cam_args = {
            'cameraDistance': dist,
            'cameraPitch': pitch,
            'cameraYaw': yaw,
            'cameraTargetPosition': np.array([pos_x, pos_y, pos_z])
    }
    if args.viz:
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, False)
        sim.resetDebugVisualizerCamera(**cam_args)
        if debug:
            res = sim.getDebugVisualizerCamera()
            print('Camera info for', cam_args)
            print('viewMatrix', res[2])
            print('projectionMatrix', res[3])
    sim.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)  # FEM deform sim
    sim.setGravity(0, 0, args.sim_gravity)
    sim.setTimeStep(1.0/args.sim_freq)
    sim.setAdditionalSearchPath(pybullet_data.getDataPath())
    # sim.setRealTimeSimulation(1)
    return
