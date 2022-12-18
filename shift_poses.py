import os
import glob
import numpy as np
from src.bbox_3D_estimation.utils import read_list_poses_orig
import re
from scipy.spatial.transform import Rotation as R


# r = R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])
# print(r.as_matrix())

regex = re.compile('[^0-9]')

DIR = "data/costum_datasets/test/demo_bottle_sfm/bottle-1/"

poses = sorted(glob.glob(f"{DIR}backup-poses/poses/*.txt"))
names = []
for pose in poses:
    name = regex.sub("", pose)[1:]
    names.append(name)
poses = read_list_poses_orig(poses)

M = np.empty((4, 4))
M[:3, :3] = np.array([[  -1.0000000,  0.0000000,  0.0000000],
                      [  0.0000000, -1.0000000, -0.0000000],
                      [ 0.0000000,  0.0000000, 1.0000000 ]])
M[:3, 3] =  np.array([+0, 0, 0])
M[3, :] = [0, 0, 0, 1]

shifted_poses = []
for pose in poses:

    pose = np.dot(M, pose)
    inverted = np.linalg.inv(pose)

    # Extract rotation matrix and translation vector from left-handed pose T
    # R = pose[:3, :3]
    # t = pose[:3, 3]

    # # Negate last element of translation vector
    # t[2] = -t[2]

    # # Convert the pose to a right-handed coordinate system
    # R_right = np.array([[R[0,0], R[0,1], -R[0,2]],
    #                     [R[1,0], R[1,1], -R[1,2]],
    #                     [-R[2,0], -R[2,1], R[2,2]]])

    # # Construct right-handed pose T'
    # T_prime = np.eye(4)
    # T_prime[:3, :3] = R_right
    # T_prime[:3, 3] = t

    
    # inverted = np.dot(M, inverted)
    # T_prime = pose
    # original = np.linalg.inv(inverted)

    shifted_poses.append(inverted)


shift_pose_dir = f"{DIR}poses/"
os.makedirs(shift_pose_dir, exist_ok=True)
for pose, name in zip(shifted_poses, names):
    np.savetxt(f"{shift_pose_dir}{name}.txt", pose, delimiter=" ")
