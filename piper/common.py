import wandb

import cv2
import os

def init_writer(savepath,
                    args,
                    k=24,):
    savepath = os.path.join(args.logdir, f'{args.env}_{args.seed}.mp4')
    vidwriter = cv2.VideoWriter(
        savepath, cv2.VideoWriter_fourcc(*'mp4v'), k,
        (args.cam_resolution, args.cam_resolution))
    print('saving to ', savepath)
    return vidwriter

def init_wandb(args,owner="libo"):
    wandb.init(project='dedo', name=f'{args.env}-preset',
                config={'env': f'{args.task}-preset'},
                tags=['preset', args.env])
