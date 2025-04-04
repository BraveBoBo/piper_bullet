"""
Command line arguments.


Note: this code is for research i.e. quick experimentation; it has minimal
comments for now, but if we see further interest from the community -- we will
add further comments, unify the style, improve efficiency and add unittests.

@contactrika

"""
import argparse

import sys
import re
import os

def get_args_parser():
    parser = argparse.ArgumentParser(description='args', add_help=True)
    #
    # Main/demo args.
    parser.add_argument('--task', type=str, default='rigid_grasp',
                         choices=['rigid_grasp', 'test'],
                         help='Task to perform (train/test)')
    parser.add_argument('--env', type=str,
                        default='RigidGrasping-v1', help='Env name')
    parser.add_argument('--max_episode_len', type=int,
                        default=200, help='Max steps per episode')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--logdir', type=str, default="logs",
                        help='Path for logs')
    parser.add_argument('--load_checkpt', type=str, default=None,
                        help='Path to a saved model e.g. '
                             '/tmp/dedo/PPO_210825_204955_HangGarment-v1'
                             '(used to re-start training or'
                             'load model for play if --play is set)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Name/ID of the device for training.')
    parser.add_argument('--rl_algo', type=str, default=None,
                        choices=['ApexDDPG', 'A2C', 'DDPG', 'Impala',
                                 'PPO', 'SAC', 'TD3'],
                        help='Name of RL algo from Stable Baselines to train')
    parser.add_argument('--play', action='store_true',
                        help='Load saved model from --load_checkpt and'
                             'play (no training)')
    parser.add_argument('--total_env_steps', type=int, default=int(10e6),
                        help='Total number of env steps for training')
    parser.add_argument('--num_envs', type=int, default=8,
                        help='Number of parallel envs.')
    parser.add_argument('--log_save_interval', type=int, default=100,
                        help='Interval for logging and saving.')
    parser.add_argument('--disable_logging_video', action='store_true',
                        help='Whether to disable dumping video to logger')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Whether to enable logging to wandb.ai')
    parser.add_argument('--viz', type=bool, default=True,
                        help='Whether to visualize')
    parser.add_argument('--debug', type=bool, default=True,
                        help='Whether to print debug info')
    #
    # Simulation args. Note: turn up frequency when deform stiffness is high.
    parser.add_argument('--sim_gravity', type=float, default=-9.8,
                        help='Gravity constant for PyBullet simulation.')
    parser.add_argument('--sim_freq', type=int, default=500,  # 250-1K
                        help='PyBullet simulation frequency.')
    parser.add_argument('--sim_steps_per_action', type=int, default=8,
                        help='Number of sim steps per action.')
    #
    # Anchor/grasping args.
    parser.add_argument('--anchor_init_pos', type=float, nargs=3,
                        default=[-0.04, 0.40, 0.70],
                        help='Initial position for an anchor')
    parser.add_argument('--other_anchor_init_pos', type=float, nargs=3,
                        default=[0.04, 0.40, 0.70],
                        help='Initial position for another anchors')
    #
    # SoftBody args.
    parser.add_argument('--override_deform_obj', type=str, default=None,
                        help='Load custom deformable (note that you have to'
                             'fill in DEFORM_INFO entry for new items)')
    parser.add_argument('--deform_init_pos', type=float, nargs=3,
                        default=[0, 0, 0.42],
                        help='Initial pos for the center of the deform object')
    parser.add_argument('--deform_init_ori', type=float, nargs=3,
                        default=[0, 0, 0],
                        help='Initial orientation for deform (in Euler angles)')
    parser.add_argument('--deform_scale', type=float, default=1.0,
                        help='Scaling for the deform object')
    parser.add_argument('--deform_bending_stiffness', type=float, default=1.0,
                        help='deform spring elastic stiffness')  # 1.0-300.0
    parser.add_argument('--deform_damping_stiffness', type=float, default=0.1,
                        help='deform spring damping stiffness')
    parser.add_argument('--deform_elastic_stiffness', type=float, default=1.0,
                        help='deform spring elastic stiffness')  # 1.0-300.0
    parser.add_argument('--deform_friction_coeff', type=float, default=0.1,
                        help='deform friction coefficient')
    parser.add_argument('--disable_self_collision', action='store_true',
                        help='Disables self collision in the deformable object')
    #
    # Texture args
    parser.add_argument('--deform_texture_file', type=str,
                        default="textures/deform/orange_pattern.png",
                        help='Texture file for the deformable objects')
    parser.add_argument('--rigid_texture_file', type=str,
                        default="textures/rigid/red_marble.png",
                        help='Texture file for the rigid objects')
    parser.add_argument('--plane_texture_file', type=str,
                        default="textures/plane/lightwood.jpg",
                        help='Texture file for the plane (floor)')
    parser.add_argument('--use_random_textures', action='store_true',
                        help='Randomly selecting a texture for the rigid obj, '
                             'deformable obj and floor from the texture folder')
    #
    # Camera args.
    parser.add_argument('--pcd', action='store_true',
                        help='Extracts depth image from sim and creates a PCD')
    parser.add_argument('--cam_config_path', type=str,
                        default="/home/libo/project/simulator/piper_bullet/piper/cam_configs/camview_d435.json",
                        help="Camera configuration file")
    parser.add_argument('--cam_resolution', type=int, default=200,
                        help='RGB camera resolution in pixels (both with and '
                             'height). Use none to get only anchor poses.')
    parser.add_argument('--cam_viewmat', type=float, nargs=6,
                        default=(1, 0, 90, 0, 0, 0.5),
                        help='Generate the view matrix for rendering camera'
                             '(not the debug camera). '
                             '[distance, pitch, yaw, posX, posY, posZ]')
    #
    # Training args.
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for training')
    parser.add_argument('--uint8_pixels', action='store_true',
                        help='Use CNNs for RL and uint8 in [0,255] for pixels')
    parser.add_argument('--flat_obs', action='store_true',
                        help='Flat observations instead of WxHxC')
    parser.add_argument('--rllib_use_torch', action='store_true',
                        help='Whether to use torch models for RLlib')
    parser.add_argument('--rollout_len', type=int, default=64,
                        help='Episode rollout length')
    parser.add_argument('--replay_size', type=int, default=10000,
                        help='Number of observations to store in replay buffer'
                             '10K 200x200 frames take ~20GBs of CPU RAM')
    parser.add_argument('--unsup_algo', type=str, default=None,
                        choices=['VAE', 'SVAE', 'PRED', 'DSA'],
                        help='Unsupervised learner (e.g. for run_svae.py)')
    # Parse args and do sanity checks.
    args, _ = parser.parse_known_args()

    return args, parser


