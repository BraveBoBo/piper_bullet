import os
from timeit import default_timer as timer
import numpy as np
import configparser
import copy

import open3d as o3d
import sys  
sys.path.append('/home/libo/project/simulator/piper_bullet/')
import thirdparty.burg_toolkit.burg_toolkit as burg   
# burg_toolkit as burg

SAVE_FILE = os.path.join('..', 'sampled_grasps.npy')



def get_grasp():
    gripper_model = burg.gripper.ParallelJawGripper(finger_length=0.07,opening_width=0.07,
                                                    finger_thickness=0.003)
    mesh_fn = '/home/libo/project/simulator/piper_bullet/dedo_piper/dedo/data/ycb/015_peach/google_16k/nontextured.ply'

    ags = burg.sampling.AntipodalGraspSampler()
    ags.mesh = burg.io.load_mesh(mesh_fn)
    ags.gripper = gripper_model
    ags.n_orientations = 18
    ags.verbose = True
    ags.max_targets_per_ref_point = 1
    graspset, contacts = ags.sample(100)

    scores = np.array([g.score for g in graspset])
    print('scores', scores)
    index = int(np.argmax(scores))
    return graspset[index].translation, graspset[index].quaternion




if __name__ == "__main__":
    # print('hi')
    # test_rotation_to_align_vectors()
    # test_angles()
    # test_cone_sampling()
    trs =get_grasp()
    print(trs)
    # show_grasp_pose_definition()
    # visualise_perturbations()
    print('bye')
