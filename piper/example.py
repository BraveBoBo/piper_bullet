import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive
from __init__ import __file__ as piper_path

interactive(True)

import gym
# import pybullet

from args import get_args_parser
from common import init_wandb, init_writer
# from bullet_manipulator import convert_all
# import util


def play(env, num_episodes=500):

    env.reset()
    graspset, _ = env.grasp_pose_from_antipodal()
    for g in graspset:
        print('g', g)
        env._simulate_grasp(g)
        env.reset()


def main(args):    
    kwargs = {'args': args}
    env = gym.make(args.env,**kwargs)
    env.seed(env.args.seed)
    print('Created', args.task, 'with observation_space',
        env.observation_space.shape, 'action_space', env.action_space.shape)
    env.reset()
    play(env, num_episodes=5)


if __name__ == '__main__':
    args,_ = get_args_parser()
    args.debug = True
    main(args)