import os
import time
import sys
sys.path.append("/home/libo/project/simulator/piper_bullet")
import wandb

import cv2
from matplotlib import cm  # for colors
import numpy as np
import open3d as o3d



import gym
import pybullet
import pybullet_utils.bullet_client as bclient
import pybullet_data

from piper.constant import SCENE_INFO,scene_name,ROBOT_INFO,DEFAULT_CAM, DEFAULT_CAM_PROJECTION
from piper.bullet_manipulator import BulletManipulator
from piper.camera_utils import cameraConfig
from piper.gripper import PiperGripper
from piper.grasp import Grasp
from piper.sampling import AntipodalGraspSampler
from piper.util import position_and_quaternion_from_tf
from piper.visualization import show_grasp_set ,create_plane
from piper.sampling import as_trimesh

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

    SOLVER_STEPS = 100  # a bit more than default helps in contact-rich tasks
    dt = 1. / 240.  # this is the default and should not be changed light-heartedly
    TIME_SLEEP = dt * 3  # for visualization
    SPINNING_FRICTION = 0.1
    SPINNING_FRICTION = 0.003  # this setting is used in AdaGrasp
    ROLLING_FRICTION = 0.0001
    MIN_OBJ_MASS = 0.05  # small masses will be replaced by this (could tune a bit more, in combo with solver)
    JOINT_TYPES = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
    antipodgraspsample_k = 10
    def __init__(self, args):
        self.args = args
        self.cam_on = args.cam_resolution > 0
        self.sim = bclient.BulletClient(
            connection_mode=pybullet.GUI if args.viz else pybullet.DIRECT)
        self.antipodgraspsample_k = 100
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
        # self.load_robot(self.sim, args, debug=args.debug)

        # use the robot gripper
        # load the gripper and the grasp and the sample the grasp
        # self.load_gripper(self.sim, args, debug=args.debug)


        self.gripper = PiperGripper()
        self.grasp = Grasp()
        self.sample = AntipodalGraspSampler()



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

        # Define grasp and the griper
        # make the gripper and the grasp

        act_sz = 3+1 # 3D pos + 1D finger distance
        act_sz += RigidGrasping.ORI_SIZE # sin,cos for 3 Euler angles
        self.action_space = gym.spaces.Box(  # [-1, 1]
            -1.0 * np.ones(self.num_anchors * act_sz),
            np.ones(self.num_anchors * act_sz))
        if self.args.debug:
            print('Wrapped as DeformEnvRobot with act', self.action_space)
    
    @staticmethod
    def unscale_pos(act, unscaled):
        if unscaled:
            return act
        return act * RigidGrasping.WORKSPACE_BOX_SIZE

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
        self.floor_id = sim.loadURDF('plane.urdf')
        if plane_texture is not None:
            if debug: print('texture file', plane_texture)
            texture_id = sim.loadTexture(plane_texture)
            sim.changeVisualShape(
                self.floor_id, -1, rgbaColor=[1,1,1,1], textureUniqueId=texture_id, )
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
        
        # self.load_robot(self.sim, self.args, debug=self.args.debug)
        
        # self.load_gripper(self.sim, self.args, debug=self.args.debug)
        
        # load the gripper and the grasp and the sample the grasp
        self.gripper = PiperGripper()
        self.grasp = Grasp()
        self.antipodgraspsample = AntipodalGraspSampler()
        self.antipodgraspsample = self.setup_grasp_sampler()


        self.sim.stepSimulation()  # step once to get initial state

        # Set up viz.
        if self.args.viz:  # loading done, so enable debug rendering if needed
            time.sleep(0.1)  # wait for debug visualizer to catch up
            self.sim.stepSimulation()  # step once to get initial state
            self.sim.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)


        obs, _ = self.get_obs()
        return obs
    
    def setup_grasp_sampler(self):
        self.antipodgraspsample.mesh = load_o3d_mesh(self.args)
        self.antipodgraspsample.gripper = self.gripper
        self.antipodgraspsample.n_orientations = 18
        self.antipodgraspsample.verbose = True
        self.antipodgraspsample.max_targets_per_ref_point = 1
        return self.antipodgraspsample


    def grasp_pose_from_antipodal(self,with_plane=True,debug=True):
        graspset, contacts = self.antipodgraspsample.sample(self.antipodgraspsample_k)
        additional_objects = []
        if with_plane:
            plane = create_plane(size=(1.5,1.5), centered=False)
            additional_objects.append(plane)
            # plane = as_trimesh(plane)
        collision = self.antipodgraspsample.check_collisions(graspset,additional_objects=additional_objects)
        graspset = graspset[~collision]
        contacts = contacts[~collision]
        if debug:
            # visualize the grasps
            show_grasp_set([self.antipodgraspsample.mesh], graspset, gripper=self.gripper, use_width=False,
                                      score_color_func=lambda s: [s, 1-s, 0], with_plane=True)

        return graspset, contacts

    def load_gripper(self, sim, args=None, position=None, orientation=None, fixed_base=False, friction=None, debug=False):
       
        data_path = os.path.join(os.path.split(__file__)[0], '..', 'data')
        sim.setAdditionalSearchPath(data_path)
        robot_info = ROBOT_INFO.get(f'piper_gripper', None)
        assert(robot_info is not None)  # make sure robot_info is ok
        robot_path = os.path.join(data_path, 'robots',
                                  robot_info['file_name'])
        
        if position is None:
            position = [0, 0, 0]
        if orientation is None:
            orientation = [0, 0, 0, 1]

        body_id = self.sim.loadURDF(robot_path, basePosition=position, baseOrientation=orientation,
                                   useFixedBase=int(fixed_base))
        
        num_joints = self.sim.getNumJoints(body_id)
        joint_infos = {}
        for joint_idx in range(num_joints):
            if friction is not None:
                self.sim.changeDynamics(
                    body_id,
                    joint_idx,
                    lateralFriction=friction,
                    spinningFriction=self.SPINNING_FRICTION,
                    rollingFriction=self.ROLLING_FRICTION,
                    frictionAnchor=False  # todo: not sure whether we really want the friction anchor
                    # documentation says:
                    # enable or disable a friction anchor: friction drift correction
                    # (disabled by default, unless set in the URDF contact section)
                )
            joint_info = self._get_joint_info(body_id, joint_idx)
            joint_infos[joint_info['link_name']] = joint_info

        return body_id, joint_infos


    def load_robot(self, sim,
                   robot_mode="piper",
                     position=None,
                     orientation=None,
                    args= None,
                   debug=False):
        """Load a robot from file, create visual and collision shapes."""
        data_path = os.path.join(os.path.split(__file__)[0], '..', 'data')
        sim.setAdditionalSearchPath(data_path)
        robot_info = ROBOT_INFO.get(f'piper', None) if robot_mode == 'piper' else ROBOT_INFO.get(f'piper_gripper', None)
        assert(robot_info is not None)  # make sure robot_info is ok
        robot_path = os.path.join(data_path, 'robots',
                                  robot_info['file_name'])
        if debug:
            print('Loading robot from', robot_path)
        self.robot = BulletManipulator(
            sim, robot_path,
            robot_mode=robot_mode,
            control_mode='velocity',
            ee_joint_name=robot_info['ee_joint_name'],
            ee_link_name=robot_info['ee_link_name'],
            base_pos=robot_info['base_pos'] if robot_mode == 'piper' else position,
            base_quat=pybullet.getQuaternionFromEuler([0, 0, np.pi]) if robot_mode == 'piper' else orientation,
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
    

    def _get_joint_info(self, body_id, joint_id):
        """returns a dict with some joint info"""
        # todo: make joint_info a class so we don't have to memorise the keys
        info = self.sim.getJointInfo(body_id, joint_id)
        joint_info = {
            'id': info[0],
            'link_name': info[12].decode("utf-8"),
            'joint_name': info[1].decode("utf-8"),
            'type': self.JOINT_TYPES[info[2]],
            'friction': info[7],
            'lower_limit': info[8],
            'upper limit': info[9],
            'max_force': info[10],
            'max_velocity': info[11],
            'joint_axis': info[13],
            'parent_pos': info[14],
            'parent_orn': info[15]
        }
        return joint_info
    
    def _are_in_collision(self, body_id_1, body_id_2):
        """
        checks if two bodies are in collision with each other.

        :return: bool, True if the two bodies are in collision
        """
        max_distance = 0.01  # 1cm for now, might want to choose a more reasonable value
        points = self.sim.getClosestPoints(body_id_1, body_id_2, max_distance)

        if self.args.debug:
            print(f'checking collision between {self.sim.getBodyInfo(body_id_1)} and {self._p.getBodyInfo(body_id_2)}')
            print(f'found {len(points)} points')

        n_colliding_points = 0
        distances = []
        for point in points:
            distance = point[8]
            distances.append(distance)
            if distance < 0:
                n_colliding_points += 1

        if self.args.debug:
            print(f'of which {n_colliding_points} have a negative distance (i.e. are in collision)')
            print(f'distances are: {distances}')

        return n_colliding_points > 0

    def step(self, action, unscaled=False):
        if self.args.debug:
            print('action', action)
        if not unscaled:
            assert self.action_space.contains(action)
            assert ((np.abs(action) <= 1.0).all()), 'action must be in [-1, 1]'
        action = action.reshape(self.num_anchors, -1)

        # Step through physics simulation.
        for sim_step in range(self.args.sim_steps_per_action):
            self.do_action(action, unscaled)
            self.sim.stepSimulation()

        return None
    

    def do_action(self, action, unscaled=False):
        # Note: action is in [-1,1], so we unscale pos (ori is sin,cos so ok).
        action = action.reshape(self.num_anchors, -1)
        ee_pos, ee_ori, _, _ = self.robot.get_ee_pos_ori_vel()
        tgt_pos = RigidGrasping.unscale_pos(action[0, :3], unscaled)
        tgt_ee_ori = ee_ori if action.shape[-1] == 3 else action[0, 3:-1]
        tgt_kwargs = {'ee_pos': tgt_pos, 'ee_ori': tgt_ee_ori,
                      'fing_dist': RigidGrasping.FING_DIST}

        tgt_qpos = self.robot.ee_pos_to_qpos(**tgt_kwargs)
        n_slack = 1  # use > 1 if robot has trouble reaching the pose
        sub_i = 0
        max_diff = 0.02
        diff = self.robot.get_qpos() - tgt_qpos
        while (np.abs(diff) > max_diff).any():
            self.robot.move_to_qpos(
                tgt_qpos, mode=pybullet.POSITION_CONTROL, kp=0.1, kd=1.0)
            self.sim.stepSimulation()
            diff = self.robot.get_qpos() - tgt_qpos
            sub_i += 1
            if sub_i >= n_slack:
                diff = np.zeros_like(diff)  # set while loop to done


    def _simulate_grasp(self, g,debug=True):
        # Step through physics simulation.
        print('************** physics engine parameters **************')
        print(self.sim.getPhysicsEngineParameters())
        print('*******************************************************')

        ########################################
        # PHASE 0: PLACING GRIPPER IN GRASP POSE
        # we have TCP grasp representation, hence need to transform gripper to TCP-oriented pose as well

        tf = np.matmul(g.pose, self.gripper.tf_base_to_TCP)
        grasp_pos, grasp_quat = position_and_quaternion_from_tf(g.pose, convention='pybullet')
        gripper_pos, gripper_quat = position_and_quaternion_from_tf(tf, convention='pybullet')
        print('grasp_pos', grasp_pos)
        print('grasp_quat', grasp_quat)
        # load a dummy robot which we can move everywhere and connect the gripper to it
        self.piper_gripper,gripper_joints = self.load_gripper(self.sim, args=self.args,
            position=gripper_pos, orientation=gripper_quat,
            fixed_base=False, friction=None, debug=debug)
            
        


        ###################################
        # PHASE:  OPEN THE GRIPPER
        # open the gripper  
        # 



        ###################################
        # PHASE 1: CHECK GRIPPER COLLISIONS
        # checking collisions against ground plane and target object
        if self._are_in_collision(self.piper_gripper, self.floor_id):
            if debug:
                print('gripper and plane are in collision')
            return None
        
        if self._are_in_collision(self.piper_gripper, self.rigid_ids[0]):
            if debug:
                print('gripper and target object are in collision')
            return None

        if debug:
            print('COLLISION CHECKS PASSED... press enter to continue')

        ##############################
        # PHASE 2: CLOSING FINGER TIPS
        # now we need to link the finger tips together, so they mimic their movement
        # this variant is by https://github.com/lzylucy/graspGripper
        # using link 1 as master with velocity control, and all other links use position control to follow 1
        # self._p.setJointMotorControl2(self._body_ids['gripper'], 1, p.VELOCITY_CONTROL, targetVelocity=1, force=50)
        # seconds = 1.0
        # for i in range(int(seconds/self.dt)):
        #     self._control_follower_joints()
        #     self._p.stepSimulation()

        #     if self.verbose:
        #         time.sleep(self.TIME_SLEEP)

        #     # checking contact
        #     if self._both_fingers_touch_object(
        #             gripper_joints['robotiq_2f_85_left_pad']['id'],
        #             gripper_joints['robotiq_2f_85_right_pad']['id']):
        #         if self.verbose:
        #             print('CONTACT ESTABLISHED')
        #             print('proceeding to hold grasp for 0.25 seconds')
        #         break
        



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

def load_o3d_mesh(args,tr=None,texture_fn=None,debug=False):
    data_path = os.path.join(os.path.split(__file__)[0], '..', 'data')
    args.data_path = data_path
    meshes = []
    for name, kwargs in SCENE_INFO[scene_name]['entities'].items():
        # load the object
        mesh_fn = os.path.join(args.data_path, name)
        assert os.path.exists(mesh_fn), f'{mesh_fn} not found'
        mesh = o3d.io.read_triangle_mesh(mesh_fn, enable_post_processing=True)
        
        if tr is None:
            # set the transform as the same as the rigid object
            tr = np.eye(4)
            tr[:3, 3] = kwargs['basePosition']  # object position
            tr[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz(
                kwargs['baseOrientation'])
            
        mesh.transform(tr)

        #scale the mesh
        mesh.scale(kwargs['globalScaling'], center=mesh.get_center())

        # set the color of the mesh
        if texture_fn is not None:
            mesh.textures = [o3d.io.read_image(texture_fn)]

        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        meshes.append(mesh)
        if debug:
            from piper.visualization import _colorize_o3d_objects
            # show the mesh
        
            _colorize_o3d_objects([o3d.geometry.TriangleMesh.create_coordinate_frame(0.02),mesh])
            o3d.visualization.draw_geometries([o3d.geometry.TriangleMesh.create_coordinate_frame(0.02),mesh])
            # show the base frame

            # time.sleep(10)

    return meshes[0] if len(meshes) == 1 else meshes
