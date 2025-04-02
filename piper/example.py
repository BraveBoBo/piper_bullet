import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive
from __init__ import __file__ as piper_path

interactive(True)

import gym
import pybullet

from args import get_args_parser
from common import init_wandb, init_writer
from bullet_manipulator import convert_all
import util


def play(env, num_episodes=500):

    vidwriter = init_writer('logs', args)
    env.reset()
    graspset, contacts = env.grasp_pose_from_antipodal()
    for g in graspset:
        print('g', g)
        re = env._simulate_grasp(g)


def main(args):    
    kwargs = {'args': args}
    env = gym.make(args.env,**kwargs)
    env.seed(env.args.seed)
    print('Created', args.task, 'with observation_space',
        env.observation_space.shape, 'action_space', env.action_space.shape)
    obs = env.reset()
    if args.debug:
        import cv2
        cv2.imshow('obs', obs)
        cv2.waitKey(0)
        print('obs', obs)
        print('obs shape', obs.shape)
    play(env, num_episodes=5)



def _set_action(tf): # set 
    finger_dist= [[0.7]]
    gripper_pos, gripper_quat = util.position_and_quaternion_from_tf(tf, convention='pybullet')
    gripper_pos = np.array(gripper_pos)[np.newaxis]
    gripper_orn = np.array(pybullet.getEulerFromQuaternion(gripper_quat))[np.newaxis]
    print('gripper_pos', gripper_pos)   
    print('gripper_orn', gripper_orn) 
    gripper_orn = convert_all(gripper_orn, 'theta_to_sin_cos')
    return np.concatenate([gripper_pos, gripper_orn, finger_dist], axis=-1)
  


if __name__ == '__main__':
    args,_ = get_args_parser()
    args.debug = False
    main(args)