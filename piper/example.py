import matplotlib.pyplot as plt
import os
from matplotlib import interactive
from __init__ import __file__ as piper_path

interactive(True)
import numpy as np

import gym

from args import get_args_parser
from common import init_wandb, init_writer

def play(env, num_episodes=500):

    # logdir = os.path.join(piper_path, 'logs')
    vidwriter = init_writer('logs', args)
    # calculate the target pose
    graspset, contacts = env.grasp_pose_from_antipodal()
    print('Grasp set:', graspset)

    for episode in range(num_episodes):

        obs = env.reset()


        if args.cam_resolution > 0:
            img = env.render(mode='rgb_array', width=args.cam_resolution,
                             height=args.cam_resolution)
            if vidwriter is not None:
                vidwriter.write(img[..., ::-1])

        step = 0
        ctrl_freq = args.sim_freq / args.sim_steps_per_action
        robot = env.robot

        gif_frames = []
        rwds = []
        print(f'# {args.env}:')

        env.sim.stepSimulation()


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




if __name__ == '__main__':
    args,_ = get_args_parser()
    args.debug = False
    main(args)